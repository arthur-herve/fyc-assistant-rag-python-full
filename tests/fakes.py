"""Doubles de test : le cœur métier se teste sans IA, sans réseau, sans disque."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Callable, Sequence

from assistant.application.errors import IndexReplacedError
from assistant.application.ports import (
    EmbeddingBatch, Generation, GenerationRequest, IndexManifest, PromptTemplate, Snapshot,
)
from assistant.domain.model import Chunk, Document, Passage

VOCABULARY = ["télétravail", "jours", "frais", "repas", "salaire", "senior", "congés", "australie"]


def make_document(doc_id: str, text: str, groups: Sequence[str] = ("tous",),
                  title: str | None = None) -> Document:
    return Document(doc_id, title or doc_id.capitalize(), text, frozenset(groups))


class KeywordEmbedder:
    """Un axe par mot du vocabulaire : les scores sont prévisibles à la main."""

    def __init__(self, model: str = "fake-keywords") -> None:
        self.model = model
        self.calls: list[tuple[str, list[str]]] = []

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if word in lowered else 0.0 for word in VOCABULARY]

    def embed_documents(self, texts):
        self.calls.append(("document", list(texts)))
        return EmbeddingBatch(self.model, len(VOCABULARY), [self._vector(t) for t in texts])

    def embed_query(self, text):
        self.calls.append(("query", [text]))
        return EmbeddingBatch(self.model, len(VOCABULARY), [self._vector(text)])


class ScriptedGenerator:
    """Renvoie des réponses écrites à l'avance et enregistre les requêtes reçues."""

    def __init__(self, *outputs: str, model: str = "fake-llm") -> None:
        self.outputs = list(outputs)
        self.model = model
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> Generation:
        self.requests.append(request)
        text = self.outputs[min(len(self.requests), len(self.outputs)) - 1]
        return Generation(self.model, text)


class FakeIndex:
    def __init__(self) -> None:
        self._manifest: IndexManifest | None = None
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def manifest(self):
        return self._manifest

    def replace(self, manifest, chunks, vectors):
        self._manifest, self.chunks, self.vectors = manifest, list(chunks), [list(v) for v in vectors]

    def search(self, vector, top_k, predicate: Callable[[Chunk], bool], index_id=None):
        if index_id is not None and (self._manifest is None or self._manifest.index_id != index_id):
            raise IndexReplacedError()   # contrat du port : jamais chercher dans un autre index que celui contrôlé

        def cosine(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return 0.0 if not na or not nb else sum(x * y for x, y in zip(a, b)) / (na * nb)
        passages = [Passage(c, cosine(vector, v)) for c, v in zip(self.chunks, self.vectors)
                    if predicate(c)]
        return sorted(passages, key=lambda p: p.score, reverse=True)[:top_k]


class ListSource:
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)

    def load(self):
        return list(self.documents)


class WholeDocumentSplitter:
    def split(self, document: Document) -> list[Chunk]:
        return [Chunk(f"{document.id}#0", document.id, document.title, document.text, 0,
                      document.allowed_groups)] if document.text else []

    def describe(self):
        return {"type": "whole"}


class FixedClock:
    """Toujours la même heure : deux manifestes ou instantanés se comparent octet pour octet."""

    def __init__(self, moment: datetime | None = None) -> None:
        self.moment = moment or datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.moment


class MemorySnapshotStore:
    def __init__(self) -> None:
        self.saved: dict[str, Snapshot] = {}

    def save(self, snapshot: Snapshot) -> None:
        self.saved[snapshot.name] = snapshot

    def load(self, name: str) -> Snapshot:
        return self.saved[name]

    def names(self) -> list[str]:
        return sorted(self.saved)


class StaticPrompts:
    def get(self, name):
        return PromptTemplate(name, "test-v1", "Cite tes sources.",
                              "Passages :\n{passages}\n\nQuestion : {question}")


def count_markers(text: str) -> int:
    return len(re.findall(r"^\[\d+\]", text, re.MULTILINE))
