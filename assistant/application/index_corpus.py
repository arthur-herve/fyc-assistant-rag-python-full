"""Cas d'usage : indexer le corpus documentaire."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from assistant.domain.model import Chunk, Document

from .errors import EmptyCorpusError, InconsistentEmbeddingsError
from .ports import Clock, DocumentSource, Embedder, IndexManifest, TextSplitter, VectorIndex


def corpus_fingerprint(documents: Sequence[Document]) -> str:
    """Empreinte du corpus : change dès qu'un texte ou un droit d'accès change.

    Chaque champ est précédé de sa longueur en octets : sans séparateur, « a » + « bc » et
    « ab » + « c » donneraient la même empreinte. Même formule dans la version C#."""
    digest = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.id):
        for field in (doc.id, doc.title, doc.text, ",".join(sorted(doc.allowed_groups))):
            data = field.encode("utf-8")
            digest.update(f"{len(data)}:".encode("ascii"))
            digest.update(data)
    return digest.hexdigest()


class IndexCorpus:
    def __init__(
        self,
        source: DocumentSource,
        splitter: TextSplitter,
        embedder: Embedder,
        index: VectorIndex,
        clock: Clock,
        batch_size: int = 32,
    ) -> None:
        self._source = source
        self._splitter = splitter
        self._embedder = embedder
        self._index = index
        self._batch_size = batch_size
        self._clock = clock

    def execute(self) -> IndexManifest:
        documents = self._source.load()
        chunks: list[Chunk] = [c for d in documents for c in self._splitter.split(d)]
        if not chunks:
            raise EmptyCorpusError()

        vectors: list[list[float]] = []
        model: str | None = None
        dimension: int | None = None
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            result = self._embedder.embed_documents([c.text for c in batch])
            if len(result.vectors) != len(batch):
                raise InconsistentEmbeddingsError(
                    f"{len(batch)} textes envoyés, {len(result.vectors)} vecteurs reçus"
                )
            if model is None:
                model, dimension = result.model, result.dimension
            elif (result.model, result.dimension) != (model, dimension):
                # Le service IA a changé de modèle en cours d'indexation.
                raise InconsistentEmbeddingsError(
                    f"modèle {model} puis {result.model} pendant la même indexation"
                )
            vectors.extend(result.vectors)

        assert model is not None and dimension is not None
        fingerprint = corpus_fingerprint(documents)
        splitter = self._splitter.describe()
        index_id = hashlib.sha256(
            json.dumps(
                [fingerprint, model, dimension, splitter], sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:12]

        manifest = IndexManifest(
            index_id=index_id,
            embedding_model=model,
            dimension=dimension,
            corpus_fingerprint=fingerprint,
            splitter=splitter,
            document_count=len(documents),
            chunk_count=len(chunks),
            created_at=self._clock.now().isoformat(timespec="seconds"),
        )
        self._index.replace(manifest, chunks, vectors)
        return manifest
