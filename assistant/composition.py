"""Racine de composition : le seul endroit qui connaît toutes les couches.

Elle lit la configuration, instancie les adaptateurs et les injecte dans les
cas d'usage. Changer de moteur, de stockage ou de modèle se fait ici (ou
dans les fichiers de configuration), jamais dans le domaine.
"""

from __future__ import annotations

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
from assistant.domain.model import User
from assistant.infrastructure.clock import SystemClock
from assistant.infrastructure.decorators import (
    CachedEmbedder, LoggingEmbedder, LoggingGenerator, RetryingEmbedder, RetryingGenerator,
)
from assistant.infrastructure.http_ai_client import HttpEmbedder, HttpGenerator
from assistant.infrastructure.markdown_corpus import MarkdownCorpus
from assistant.infrastructure.prompt_files import FilePromptRepository
from assistant.infrastructure.snapshot_files import JsonSnapshotStore
from assistant.infrastructure.splitter import ParagraphSplitter
from assistant.infrastructure.vector_index import JsonVectorIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


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
            raise ConfigError(f"{where} : clé(s) inconnue(s) {unknown} (connues : {sorted(known)})")


def _section(raw: dict[str, Any], name: str, path: Path, required: bool = True) -> dict[str, Any]:
    section = raw.get(name, {} if not required else None)
    if not isinstance(section, dict):
        raise ConfigError(f"{path} : section [{name}] {'absente' if section is None else 'mal formée'}")
    return section


def _value(section: dict[str, Any], key: str, where: str, kind: type, default: Any = ...,
           minimum: float | None = None, maximum: float | None = None) -> Any:
    if key not in section:
        if default is ...:
            raise ConfigError(f"{where} : clé « {key} » obligatoire")
        return default
    value = section[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ConfigError(f"{where} : « {key} » doit être de type {kind.__name__}, pas {value!r}")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ConfigError(f"{where} : « {key} » = {value} hors de [{minimum}, {maximum}]")
    return value


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
        path = Path(path or os.environ.get("ASSISTANT_CONFIG", PROJECT_ROOT / "config/app.toml"))
        try:
            with open(path, "rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path} : TOML invalide — {error}") from error
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
        for key, kind, minimum in (("max_chars", int, 1), ("overlap_chars", int, 0), ("include_title", bool, None)):
            _value(splitter, key, f"{path} [splitter]", kind, default=None, minimum=minimum)
        min_scores = _section(retrieval, "min_score", path, required=False)
        users = _section(raw, "users", path, required=False)
        seed = _value(gen, "seed", f"{path} [generation]", int, default=None)
        return cls(
            ai_base_url=os.environ.get("AI_SERVICE_URL", _value(ai, "base_url", f"{path} [ai_service]", str)),
            embedding_model=_value(ai, "embedding_model", f"{path} [ai_service]", str),
            generation_model=_value(ai, "generation_model", f"{path} [ai_service]", str),
            timeout=_value(ai, "timeout_seconds", f"{path} [ai_service]", float, 300.0, minimum=1),
            corpus_dir=base / _value(_section(raw, "corpus", path), "directory", f"{path} [corpus]", str),
            index_path=base / _value(_section(raw, "index", path), "path", f"{path} [index]", str),
            snapshots_dir=base / _value(_section(raw, "snapshots", path, required=False), "directory",
                                        f"{path} [snapshots]", str, "eval/instantanes"),
            splitter=dict(splitter),
            top_k=_value(retrieval, "top_k", f"{path} [retrieval]", int, 4, minimum=1),
            min_scores={k: _value(min_scores, k, f"{path} [retrieval.min_score]", float, minimum=-1, maximum=1)
                        for k in min_scores},
            temperature=_value(gen, "temperature", f"{path} [generation]", float, 0.2, minimum=0, maximum=2),
            max_tokens=_value(gen, "max_tokens", f"{path} [generation]", int, 400, minimum=1),
            max_attempts=_value(gen, "max_attempts", f"{path} [generation]", int, 2, minimum=1),
            seed=seed,
            prompt_name=_value(gen, "prompt", f"{path} [generation]", str, "answer"),
            decorators=dict(decorators),
            users={
                name: frozenset(_value(_section(users, name, path), "groups", f"{path} [users.{name}]", list, ["tous"]))
                for name in users
            },
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
            raise UnknownUserError(
                f"utilisateur inconnu : {name} (connus : {sorted(self.users)})"
            )
        return User(id=name, groups=self.users[name])


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
