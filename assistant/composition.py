"""Racine de composition : le seul endroit qui connaît toutes les couches.

Elle lit la configuration, instancie les adaptateurs et les injecte dans les
cas d'usage. Changer de moteur, de stockage ou de modèle se fait ici (ou
dans les fichiers de configuration), jamais dans le domaine.
"""

from __future__ import annotations

import json
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.guards import OutputValidatingGenerator
from assistant.application.index_corpus import IndexCorpus
from assistant.application.ports import (
    Embedder, Generator, IndexManifest, PromptRepository, SnapshotStore, VectorIndex,
)
from assistant.application.search_passages import SearchPassages
from assistant.application.snapshots import RecordSnapshot
from assistant.application.status import CheckStatus
from assistant.domain.access import PUBLIC_GROUP
from assistant.domain.model import User
from assistant.infrastructure.clock import SystemClock
from assistant.infrastructure.decorators import (
    CachedEmbedder, LoggingEmbedder, LoggingGenerator, RetryingEmbedder, RetryingGenerator,
)
from assistant.infrastructure.http_ai_client import HttpEmbedder, HttpGenerator
from assistant.infrastructure.json_text import MAX_DIGITS, parse as parse_json
from assistant.infrastructure.markdown_corpus import MarkdownCorpus
from assistant.infrastructure.prompt_files import FilePromptRepository
from assistant.infrastructure.snapshot_files import JsonSnapshotStore
from assistant.infrastructure.splitter import ParagraphSplitter
from assistant.infrastructure.text_files import decode_utf8, read_utf8
from assistant.infrastructure.vector_index import JsonVectorIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def config_file(path: str | Path | None = None) -> Path:
    """Le fichier de configuration à lire : `path` (--config), sinon la variable ASSISTANT_CONFIG, sinon
    config/app.toml sous la racine du projet."""
    return Path(path or os.environ.get("ASSISTANT_CONFIG", PROJECT_ROOT / "config/app.toml"))


class UnknownUserError(LookupError):
    pass


class ConfigError(ValueError):
    """Configuration absente, incomplète ou hors bornes : le message nomme le fichier et la clé."""


# Décorateurs connus et type attendu : une faute de frappe (« validate_ouput ») ou une valeur
# comme « oui » ne doit pas couper un garde-fou en silence.
DECORATOR_TYPES: dict[str, type] = {
    "validate_output": bool, "max_output_chars": int, "cache_embeddings": bool, "log": bool, "retries": int,
}

# Clés attendues, section par section (None : le premier niveau). Même raison : « topk = 8 »
# ou « max_char = 300 » ne doivent pas laisser la valeur par défaut s'appliquer sans rien dire.
# TOML a de vrais commentaires : « _note = … » est une clé inconnue comme une autre (les jeux de
# questions, en JSON, faute de commentaires, acceptent les clés « _ »).
KNOWN_KEYS: dict[str | None, set[str]] = {
    None: {"ai_service", "corpus", "index", "snapshots", "splitter", "retrieval", "generation", "decorators", "users"},
    "ai_service": {"base_url", "embedding_model", "generation_model", "timeout_seconds"},
    "corpus": {"directory"},
    "index": {"path"},
    "snapshots": {"directory"},
    "splitter": {"max_chars", "overlap_chars", "include_title"},
    "retrieval": {"top_k", "min_score"},
    "generation": {"prompt", "temperature", "max_tokens", "max_attempts", "seed"},
    "decorators": set(DECORATOR_TYPES),
}

# Délai maximal d'un appel au service IA : un jour, bien au-delà d'une génération lente. Sans
# borne, une valeur absurde ferait échouer le client HTTP plus tard, sans nommer fichier ni clé.
MAX_TIMEOUT_SECONDS = 86_400

# Le type attendu, nommé en français. La valeur fautive est citée en JSON compact (_show) : true et non True,
# une chaîne entre guillemets, qu'on distingue d'un nombre.
KIND_NAMES: dict[type, str] = {bool: "un booléen", int: "un entier", float: "un nombre", str: "une chaîne"}


def _check_known_keys(raw: dict[str, Any], path: Path) -> None:
    sections = [(name, raw if name is None else raw.get(name), known) for name, known in KNOWN_KEYS.items()]
    users = raw.get("users")
    if isinstance(users, dict):
        sections += [(f"users.{name}", user, {"groups"}) for name, user in users.items()]
    for name, values, known in sections:
        if not isinstance(values, dict):
            continue   # absente ou mal formée : _section le dira
        unknown = sorted(set(values) - known)
        if unknown:
            where = path if name is None else f"{path} [{name}]"
            raise ConfigError(f"{where} : clé(s) inconnue(s) [{', '.join(unknown)}] "
                              f"(connues : {', '.join(sorted(known))})")


def _section(raw: dict[str, Any], name: str, path: Path, required: bool = True,
             parent: str | None = None) -> dict[str, Any]:
    section = raw.get(name, {} if not required else None)
    if not isinstance(section, dict):
        label = name if parent is None else f"{parent}.{name}"
        raise ConfigError(f"{path} : section [{label}] {'absente' if section is None else 'mal formée'}")
    return section


# Un entier de plus de 4300 chiffres : int() refuse de le convertir, et tomllib lève alors une ValueError de CPython,
# en anglais, sans dire où ; lu en hexadécimal, en octal ou en binaire, tomllib le convertit, mais str() refuse de
# l'écrire. Refusé avec un message en français, comme une erreur de syntaxe (« TOML invalide — … »).
_HUGE = 10 ** MAX_DIGITS


def _load_toml(text: str) -> dict[str, Any]:
    """tomllib.loads ; ValueError, avec un message en français, pour un entier de plus de 4300 chiffres."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        raise
    except ValueError:   # hors TOMLDecodeError, seul int() en lève
        raw = None
    if raw is None or _has_huge_integer(raw):
        raise ValueError(f"TOML invalide — nombre entier de plus de {MAX_DIGITS} chiffres")
    return raw


def _has_huge_integer(value: Any) -> bool:
    """Un entier lu en hexadécimal, en octal ou en binaire, de plus de 4300 chiffres décimaux. Une boucle, pas
    any(…) : un appel par niveau, pour que des tableaux ou des tables en ligne butent sur la limite de tomllib avant
    celle de cette fonction (sous 3.11 aussi). Des en-têtes [a.a.…], que tomllib lit sans récursion, butent ici : load
    le dit de même (trop de niveaux d'imbrication)."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            if _has_huge_integer(item):
                return True
        return False
    return isinstance(value, int) and value >= _HUGE   # un booléen vaut au plus 1


def _real(value: int) -> float:
    """Un entier là où un réel est attendu : l'infini, du signe de l'entier, au-delà du plus grand réel
    (float() lèverait OverflowError, une trace d'erreur)."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _show(value: Any) -> str:
    """La valeur en JSON compact, accents écrits tels quels (ensure_ascii=False)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _range(minimum: float | None, maximum: float | None) -> str:
    """Toute borne haute va avec une borne basse (délai, recouvrement, seuils, température)."""
    if maximum is None:
        return f"doit valoir au moins {minimum}"
    return f"doit être compris entre {minimum} et {maximum}"


def _value(section: dict[str, Any], key: str, where: str, kind: type, default: Any = ...,
           minimum: float | None = None, maximum: float | None = None) -> Any:
    """Valeur typée et bornée. La valeur par défaut est bornée elle aussi : le recouvrement par
    défaut ne convient pas à tout max_chars. `not value >= minimum` refuse aussi nan (TOML)."""
    if key in section:
        value, shown = section[key], _show(section[key])
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = _real(value)
        if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
            raise ConfigError(f"{where} : « {key} » doit être {KIND_NAMES[kind]}, pas {shown}")
    elif default is ...:
        raise ConfigError(f"{where} : clé « {key} » obligatoire")
    elif default is None:
        return None
    else:
        value, shown = default, f"{_show(default)} (valeur par défaut)"
    if (minimum is not None and not value >= minimum) or (maximum is not None and not value <= maximum):
        raise ConfigError(f"{where} : « {key} » = {shown}, {_range(minimum, maximum)}")
    return value


def _groups(user: dict[str, Any], where: str) -> frozenset[str]:
    groups = user.get("groups", [PUBLIC_GROUP])
    if not isinstance(groups, list) or not all(isinstance(group, str) for group in groups):
        raise ConfigError(f"{where} : « groups » doit être une liste de chaînes, pas {_show(groups)}")
    return frozenset(groups)


@dataclass
class AppConfig:
    ai_base_url: str
    embedding_model: str
    generation_model: str
    timeout: float
    corpus_dir: Path
    index_path: Path
    snapshots_dir: Path
    splitter: dict[str, Any]
    top_k: int
    min_scores: dict[str, float]
    temperature: float
    max_tokens: int
    max_attempts: int
    seed: int | None
    prompt_name: str = "answer"
    decorators: dict[str, Any] = field(default_factory=dict)
    users: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        path = config_file(path)
        try:
            # UTF-8 strict, BOM accepté, comme les autres fichiers lus (text_files).
            raw = _load_toml(decode_utf8(path.read_bytes()))
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path} : TOML invalide — {error}") from error
        except RecursionError:   # des centaines de niveaux d'imbrication : tomllib, ou _has_huge_integer, abandonne
            raise ConfigError(f"{path} : TOML invalide — trop de niveaux d'imbrication") from None
        except ValueError as error:   # pas en UTF-8 (l'octet fautif et sa position), ou entier de plus de 4300 chiffres
            raise ConfigError(f"{path} : {error}") from error
        base = PROJECT_ROOT  # chemins relatifs à la racine du projet (ou absolus)
        _check_known_keys(raw, path)
        ai = _section(raw, "ai_service", path)
        gen = _section(raw, "generation", path, required=False)
        retrieval = _section(raw, "retrieval", path, required=False)
        decorators = _section(raw, "decorators", path, required=False)
        for key, kind in DECORATOR_TYPES.items():
            _value(decorators, key, f"{path} [decorators]", kind, default=None,
                   minimum={"max_output_chars": 1, "retries": 0}.get(key))
        splitter = _section(raw, "splitter", path, required=False)
        # Bornes de ParagraphSplitter, vérifiées ici pour nommer le fichier et la clé ; le
        # recouvrement, même par défaut, doit rester sous la moitié de max_chars.
        max_chars = _value(splitter, "max_chars", f"{path} [splitter]", int, 800, minimum=100)
        splitter_settings = {
            "max_chars": max_chars,
            "overlap_chars": _value(splitter, "overlap_chars", f"{path} [splitter]", int, 120,
                                    minimum=0, maximum=max_chars // 2 - 1),
            "include_title": _value(splitter, "include_title", f"{path} [splitter]", bool, True),
        }
        min_scores = _section(retrieval, "min_score", path, required=False, parent="retrieval")
        users = _section(raw, "users", path, required=False)
        seed = _value(gen, "seed", f"{path} [generation]", int, default=None)
        # AI_SERVICE_URL remplace l'adresse, mais base_url reste vérifiée : même verdict sur le fichier
        # quel que soit l'environnement.
        base_url = _value(ai, "base_url", f"{path} [ai_service]", str)
        return cls(
            ai_base_url=os.environ.get("AI_SERVICE_URL", base_url),
            embedding_model=_value(ai, "embedding_model", f"{path} [ai_service]", str),
            generation_model=_value(ai, "generation_model", f"{path} [ai_service]", str),
            timeout=_value(ai, "timeout_seconds", f"{path} [ai_service]", float, 300.0,
                           minimum=1, maximum=MAX_TIMEOUT_SECONDS),
            corpus_dir=base / _value(_section(raw, "corpus", path), "directory", f"{path} [corpus]", str),
            index_path=base / _value(_section(raw, "index", path), "path", f"{path} [index]", str),
            snapshots_dir=base / _value(_section(raw, "snapshots", path, required=False), "directory",
                                        f"{path} [snapshots]", str, "eval/instantanes"),
            splitter=splitter_settings,
            top_k=_value(retrieval, "top_k", f"{path} [retrieval]", int, 4, minimum=1),
            min_scores={k: _value(min_scores, k, f"{path} [retrieval.min_score]", float, minimum=-1, maximum=1)
                        for k in min_scores},
            temperature=_value(gen, "temperature", f"{path} [generation]", float, 0.2, minimum=0, maximum=2),
            max_tokens=_value(gen, "max_tokens", f"{path} [generation]", int, 400, minimum=1),
            max_attempts=_value(gen, "max_attempts", f"{path} [generation]", int, 2, minimum=1),
            seed=seed,
            prompt_name=_value(gen, "prompt", f"{path} [generation]", str, "answer"),
            decorators=dict(decorators),
            users={name: _groups(_section(users, name, path, parent="users"), f"{path} [users.{name}]")
                   for name in users},
        )

    def min_score_for(self, embedding_model: str) -> float:
        """Seuil configuré pour cet alias, sinon la valeur `default` (ADR 0004 : à calibrer)."""
        return self.min_scores.get(embedding_model, self.min_scores.get("default", 0.4))

    def has_threshold_for(self, embedding_model: str) -> bool:
        """Un seuil est-il configuré pour cet alias ? Configuré ne veut pas dire mesuré : les
        commentaires de `[retrieval.min_score]` disent lesquels le sont."""
        return embedding_model in self.min_scores

    def user(self, name: str) -> User:
        if name not in self.users:
            raise UnknownUserError(   # « (connus : alice, bruno, claire) » : les noms, sans crochets ni guillemets
                f"utilisateur inconnu : {name} (connus : {', '.join(sorted(self.users))})"
            )
        return User(id=name, groups=self.users[name])


def read_json(path: str | Path) -> Any:
    """Un fichier JSON que lit l'interface (jeu de questions), lu comme ceux de l'infrastructure :
    UTF-8 strict (BOM accepté) et JSON strict (clé en double, NaN, entier de plus de 4300 chiffres,
    plus de 900 niveaux : ValueError), chaînes comprises : un « \\ud800 » isolé n'est pas du texte (il ne
    s'écrit pas en UTF-8). L'interface reçoit des valeurs, sans nommer l'infrastructure ;
    le message ne nomme pas le fichier : l'appelant le préfixe."""
    return parse_json(read_utf8(Path(path)), strings=True)


def read_json_text(text: str) -> Any:
    """Un texte JSON que lit l'interface (corps d'une requête à l'API HTTP, réponse du service IA à GET
    /v1/models), lu aussi strictement que les fichiers, par le même contrôleur : clé en double, NaN,
    entier de plus de 4300 chiffres, plus de 900 niveaux, chaîne qui n'est pas du texte : ValueError
    (json_text). `text` vient d'un décodage UTF-8 strict des octets reçus (decode("utf-8-sig")), comme
    le suppose json_text.parse, qui ne parcourt les chaînes que si le texte contient un échappement
    « \\ud800 » à « \\udfff »."""
    return parse_json(text, strings=True)


@dataclass
class Container:
    config: AppConfig
    index_corpus: IndexCorpus
    ask_question: AskQuestion
    search_passages: SearchPassages
    check_status: CheckStatus
    record_snapshot: RecordSnapshot
    snapshots: SnapshotStore
    index: VectorIndex
    prompts: PromptRepository
    settings: AskSettings


def build(config: AppConfig, embedding_model: str | None = None,
          generation_model: str | None = None, index_path: str | Path | None = None,
          min_score: float | None = None,
          splitter_overrides: dict[str, Any] | None = None,
          prompt_name: str | None = None) -> Container:
    embedding_model = embedding_model or config.embedding_model
    generation_model = generation_model or config.generation_model

    index = JsonVectorIndex(index_path or config.index_path)
    # L'adaptateur nu sert à `status` : un cache d'embeddings masquerait un changement de modèle servi.
    raw_embedder = HttpEmbedder(config.ai_base_url, embedding_model, config.timeout)
    embedder, generator = decorate(
        raw_embedder,
        HttpGenerator(config.ai_base_url, generation_model, config.timeout),
        config.decorators,
        current_index=index.manifest,
    )
    source = MarkdownCorpus(config.corpus_dir)
    splitter = ParagraphSplitter(**{**config.splitter, **(splitter_overrides or {})})
    prompts = FilePromptRepository(PROMPTS_DIR)
    clock = SystemClock()

    index_corpus = IndexCorpus(
        source=source,
        splitter=splitter,
        embedder=embedder,
        index=index,
        clock=clock,
    )
    settings = AskSettings(
        top_k=config.top_k,
        min_score=min_score if min_score is not None else config.min_score_for(embedding_model),
        max_attempts=config.max_attempts,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
        prompt_name=prompt_name or config.prompt_name,
    )
    ask_question = AskQuestion(
        embedder=embedder,
        index=index,
        generator=generator,
        prompts=prompts,
        settings=settings,
    )
    check_status = CheckStatus(source, splitter, raw_embedder, index, prompts, settings.prompt_name)
    snapshots = JsonSnapshotStore(config.snapshots_dir)
    # Empreinte de configuration d'un instantané : tout ce qui change les réponses côté application
    # (mêmes clés que la version C#). Les réglages propres au service IA (budget de réflexion…) n'y
    # sont pas : ils vivent dans l'autre déployable. Deux instantanés de même configuration peuvent
    # donc venir de deux réglages du service : c'est une limite à nommer (S4.2), pas un oubli.
    configuration = {
        "corpus": config.corpus_dir.name,
        "embedding_model": embedding_model,
        "generation_model": generation_model,
        "splitter": splitter.describe(),
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "prompt": settings.prompt_name,
        "max_attempts": settings.max_attempts,
        "validate_output": bool(config.decorators.get("validate_output", True)),
        "max_output_chars": int(config.decorators.get("max_output_chars", 1500)),
    }
    record_snapshot = RecordSnapshot(ask_question, snapshots, clock, configuration)
    return Container(config, index_corpus, ask_question, SearchPassages(embedder, index), check_status,
                     record_snapshot, snapshots, index, prompts, settings)


def decorate(embedder: Embedder, generator: Generator, options: dict[str, Any], *,
             current_index: Callable[[], IndexManifest | None]) -> tuple[Embedder, Generator]:
    """Empile les décorateurs choisis dans `[decorators]` (séquence 4.1).

    Ordre, de l'intérieur vers l'extérieur : nouvelles tentatives (au plus près
    du réseau), journal des embeddings, cache (pour ne pas journaliser les
    réponses servies depuis le cache), validation de la sortie (règle métier),
    puis journal des générations (pour voir aussi les rejets). Les cas d'usage
    ne voient que les ports. `current_index` renvoie le manifeste de l'index
    courant : le cache d'embeddings ne sert que cet index (ADR 0009).
    """
    retries = int(options.get("retries", 0))
    if retries > 0:
        embedder = RetryingEmbedder(embedder, attempts=retries)
        generator = RetryingGenerator(generator, attempts=retries)
    if options.get("log", False):
        embedder = LoggingEmbedder(embedder)
    if options.get("cache_embeddings", False):
        embedder = CachedEmbedder(embedder, current_index)
    if options.get("validate_output", True):
        generator = OutputValidatingGenerator(generator, int(options.get("max_output_chars", 1500)))
    if options.get("log", False):
        generator = LoggingGenerator(generator)  # journalise aussi les rejets de la validation
    return embedder, generator
