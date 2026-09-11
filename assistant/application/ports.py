"""Ports : ce dont les cas d'usage ont besoin, sans dire comment c'est fait.

Les adaptateurs de la couche infrastructure implémentent ces protocoles.
Les cas d'usage ne connaissent que ces signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol, Sequence

from assistant.domain.model import Chunk, Document, Passage


# --- Embeddings --------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingBatch:
    """Vecteurs produits par un modèle.

    `model` est l'identifiant du modèle *réellement* utilisé, tel que renvoyé
    par le service IA, et non le nom demandé dans la configuration.
    """

    model: str
    dimension: int
    vectors: list[list[float]]


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...

    def embed_query(self, text: str) -> EmbeddingBatch: ...


# --- Génération --------------------------------------------------------------

@dataclass(frozen=True)
class GenerationRequest:
    system: str
    prompt: str
    temperature: float
    max_tokens: int
    seed: int | None = None


@dataclass(frozen=True)
class Generation:
    model: str
    text: str


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> Generation: ...


# --- Index vectoriel ---------------------------------------------------------

@dataclass(frozen=True)
class IndexManifest:
    """Carte d'identité d'un index : sans elle, un index est inexplicable."""

    index_id: str
    embedding_model: str
    dimension: int
    corpus_fingerprint: str
    splitter: dict
    document_count: int
    chunk_count: int
    created_at: str


class VectorIndex(Protocol):
    def manifest(self) -> IndexManifest | None: ...

    def replace(
        self,
        manifest: IndexManifest,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        predicate: Callable[[Chunk], bool],
    ) -> list[Passage]: ...


# --- Corpus et découpage -----------------------------------------------------

class DocumentSource(Protocol):
    def load(self) -> list[Document]: ...


class TextSplitter(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...

    def describe(self) -> dict: ...


# --- Prompts -----------------------------------------------------------------

@dataclass(frozen=True)
class PromptTemplate:
    """Prompt versionné. Sa version fait partie de la trace de chaque réponse."""

    name: str
    version: str
    system: str
    user: str

    def render(self, **values: str) -> str:
        return self.user.format(**values)


class PromptRepository(Protocol):
    def get(self, name: str) -> PromptTemplate: ...


# --- Horloge -----------------------------------------------------------------

class Clock(Protocol):
    """Le plus petit port du projet : l'heure est une entrée-sortie déguisée.

    Sans lui, deux instantanés ou deux manifestes ne pourraient jamais être
    comparés octet pour octet dans un test (séquence 4.2).
    """

    def now(self) -> datetime: ...


# --- Instantanés -------------------------------------------------------------

@dataclass(frozen=True)
class SnapshotEntry:
    """Ce qu'a répondu l'assistant à une question, réduit à ce qui se compare."""

    question_id: str
    user_id: str
    question: str
    status: str
    cited_documents: tuple[str, ...]
    text: str
    attempts: int = 0


@dataclass(frozen=True)
class Snapshot:
    """Comportement du système figé à un instant, avec la configuration qui l'a produit.

    `configuration` est l'empreinte de tout ce qui change les réponses : index,
    modèles, prompt, découpage, seuil… Comparer deux instantanés sans comparer
    leurs configurations n'aurait aucun sens (séquence 3.2).
    """

    name: str
    created_at: str
    configuration: dict[str, Any] = field(default_factory=dict)
    entries: tuple[SnapshotEntry, ...] = ()


class SnapshotStore(Protocol):
    def save(self, snapshot: Snapshot) -> None: ...

    def load(self, name: str) -> Snapshot: ...

    def names(self) -> list[str]: ...
