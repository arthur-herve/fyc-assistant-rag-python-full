"""Décorateurs techniques sur les ports `Embedder` et `Generator` (séquence 4.1).

Chacun enveloppe une implémentation du même port et ajoute une chose :
cache, journal, nouvelles tentatives. Les cas d'usage n'en savent rien ;
`composition.py` choisit lesquels empiler, et dans quel ordre.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

from assistant.application.errors import AIServiceError
from assistant.application.ports import (
    Embedder, EmbeddingBatch, Generation, GenerationRequest, Generator,
)

log = logging.getLogger("assistant")


# --- Cache -------------------------------------------------------------------

class CachedEmbedder:
    """Mémorise les vecteurs déjà calculés dans le processus.

    Attention, leçon de la séquence 4.2 : un cache d'embeddings est lui-même
    un artefact lié au modèle. Il est ici en mémoire, donc perdu à l'arrêt ;
    un cache persistant devrait porter l'identifiant du modèle dans sa clé.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._queries: dict[str, EmbeddingBatch] = {}
        self._documents: dict[tuple[str, ...], EmbeddingBatch] = {}
        self.hits = 0
        self.misses = 0

    def embed_query(self, text: str) -> EmbeddingBatch:
        if text in self._queries:
            self.hits += 1
            return self._queries[text]
        self.misses += 1
        batch = self._inner.embed_query(text)
        self._queries[text] = batch
        return batch

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        key = tuple(texts)
        if key in self._documents:
            self.hits += 1
            return self._documents[key]
        self.misses += 1
        batch = self._inner.embed_documents(texts)
        self._documents[key] = batch
        return batch


# --- Journal -----------------------------------------------------------------

class LoggingEmbedder:
    def __init__(self, inner: Embedder, logger: logging.Logger = log) -> None:
        self._inner = inner
        self._log = logger

    def embed_query(self, text: str) -> EmbeddingBatch:
        start = time.perf_counter()
        batch = self._inner.embed_query(text)
        self._log.info("embeddings requête · %s · %d dim. · %.0f ms",
                       batch.model, batch.dimension, (time.perf_counter() - start) * 1000)
        return batch

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        start = time.perf_counter()
        batch = self._inner.embed_documents(texts)
        self._log.info("embeddings %d documents · %s · %.0f ms",
                       len(texts), batch.model, (time.perf_counter() - start) * 1000)
        return batch


class LoggingGenerator:
    def __init__(self, inner: Generator, logger: logging.Logger = log) -> None:
        self._inner = inner
        self._log = logger

    def generate(self, request: GenerationRequest) -> Generation:
        start = time.perf_counter()
        try:
            generation = self._inner.generate(request)
        except Exception as error:
            self._log.warning("génération échouée après %.0f ms : %s",
                              (time.perf_counter() - start) * 1000, error)
            raise
        self._log.info("génération · %s · %d caractères · %.0f ms",
                       generation.model, len(generation.text), (time.perf_counter() - start) * 1000)
        return generation


# --- Nouvelles tentatives ----------------------------------------------------

def _retry(action: Callable[[], object], attempts: int, delay: float, sleep: Callable[[float], None]):
    last: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return action()
        except AIServiceError as error:
            last = error
            if attempt < attempts:
                log.warning("service IA en erreur (%s), nouvelle tentative %d/%d",
                            error, attempt + 1, attempts)
                sleep(delay * (attempt + 1))
    assert last is not None
    raise last


class RetryingEmbedder:
    """Réessaie quand le service IA est injoignable ou en erreur (pas quand la
    réponse est invalide : ce n'est pas un problème de transport)."""

    def __init__(self, inner: Embedder, attempts: int = 1, delay_seconds: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._inner = inner
        self._attempts = attempts
        self._delay = delay_seconds
        self._sleep = sleep

    def embed_query(self, text: str) -> EmbeddingBatch:
        return _retry(lambda: self._inner.embed_query(text), self._attempts, self._delay, self._sleep)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return _retry(lambda: self._inner.embed_documents(texts), self._attempts, self._delay, self._sleep)


class RetryingGenerator:
    def __init__(self, inner: Generator, attempts: int = 1, delay_seconds: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._inner = inner
        self._attempts = attempts
        self._delay = delay_seconds
        self._sleep = sleep

    def generate(self, request: GenerationRequest) -> Generation:
        return _retry(lambda: self._inner.generate(request), self._attempts, self._delay, self._sleep)
