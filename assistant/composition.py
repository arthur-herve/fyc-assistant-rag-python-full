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
from typing import Any

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.guards import OutputValidatingGenerator
from assistant.application.index_corpus import IndexCorpus
from assistant.application.ports import Embedder, Generator
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
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        base = PROJECT_ROOT  # chemins relatifs à la racine du projet (ou absolus)
        ai, gen, retrieval = raw["ai_service"], raw.get("generation", {}), raw.get("retrieval", {})
        return cls(
            ai_base_url=os.environ.get("AI_SERVICE_URL", ai["base_url"]),
            embedding_model=ai["embedding_model"],
            generation_model=ai["generation_model"],
            timeout=float(ai.get("timeout_seconds", 300)),
            corpus_dir=base / raw["corpus"]["directory"],
            index_path=base / raw["index"]["path"],
            snapshots_dir=base / raw.get("snapshots", {}).get("directory", "eval/instantanes"),
            splitter=dict(raw.get("splitter", {})),
            top_k=int(retrieval.get("top_k", 4)),
            min_scores={k: float(v) for k, v in retrieval.get("min_score", {}).items()},
            temperature=float(gen.get("temperature", 0.2)),
            max_tokens=int(gen.get("max_tokens", 400)),
            max_attempts=int(gen.get("max_attempts", 2)),
            seed=gen.get("seed"),
            prompt_name=str(gen.get("prompt", "answer")),
            decorators=dict(raw.get("decorators", {})),
            users={
                name: frozenset(entry.get("groups", ["tous"]))
                for name, entry in raw.get("users", {}).items()
            },
        )

    def min_score_for(self, embedding_model: str) -> float:
        return self.min_scores.get(embedding_model, self.min_scores.get("default", 0.4))

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
    check_status: CheckStatus
    record_snapshot: RecordSnapshot
    snapshots: JsonSnapshotStore
    index: JsonVectorIndex
    embedder: Embedder
    prompts: FilePromptRepository
    settings: AskSettings


def build(config: AppConfig, embedding_model: str | None = None,
          generation_model: str | None = None, index_path: str | Path | None = None,
          min_score: float | None = None,
          splitter_overrides: dict[str, Any] | None = None,
          prompt_name: str | None = None) -> Container:
    embedding_model = embedding_model or config.embedding_model
    generation_model = generation_model or config.generation_model

    embedder, generator = decorate(
        HttpEmbedder(config.ai_base_url, embedding_model, config.timeout),
        HttpGenerator(config.ai_base_url, generation_model, config.timeout),
        config.decorators,
    )
    index = JsonVectorIndex(index_path or config.index_path)
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
    check_status = CheckStatus(source, splitter, embedder, index, prompts, settings.prompt_name)
    snapshots = JsonSnapshotStore(config.snapshots_dir)
    # Empreinte de configuration d'un instantané : tout ce qui change les réponses.
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
    }
    record_snapshot = RecordSnapshot(ask_question, snapshots, clock, configuration)
    return Container(config, index_corpus, ask_question, check_status, record_snapshot,
                     snapshots, index, embedder, prompts, settings)


def decorate(embedder: Embedder, generator: Generator,
             options: dict[str, Any]) -> tuple[Embedder, Generator]:
    """Empile les décorateurs choisis dans `[decorators]` (séquence 4.1).

    Ordre, de l'intérieur vers l'extérieur : nouvelles tentatives (au plus près
    du réseau), journal, cache (pour ne pas journaliser les réponses servies
    depuis le cache), validation de la sortie (règle métier, au plus près du
    cas d'usage). Les cas d'usage ne voient que les ports.
    """
    retries = int(options.get("retries", 0))
    if retries > 0:
        embedder = RetryingEmbedder(embedder, attempts=retries)
        generator = RetryingGenerator(generator, attempts=retries)
    if options.get("log", False):
        embedder = LoggingEmbedder(embedder)
        generator = LoggingGenerator(generator)
    if options.get("cache_embeddings", False):
        embedder = CachedEmbedder(embedder)
    if options.get("validate_output", True):
        generator = OutputValidatingGenerator(generator, int(options.get("max_output_chars", 1500)))
    return embedder, generator
