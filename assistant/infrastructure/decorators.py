"""Décorateurs techniques sur les ports `Embedder` et `Generator` (séquence 4.1).

Chacun enveloppe une implémentation du même port et ajoute une chose :
cache, journal, nouvelles tentatives. Les cas d'usage n'en savent rien ;
`composition.py` choisit lesquels empiler, et dans quel ordre.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Sequence

from assistant.application.errors import AIServiceError
from assistant.application.ports import (
    Embedder, EmbeddingBatch, Generation, GenerationRequest, Generator, IndexManifest,
)

log = logging.getLogger("assistant")


# --- Cache -------------------------------------------------------------------

class CachedEmbedder:
    """Mémorise, dans le processus, les vecteurs des questions déjà posées.

    Leçon de la séquence 4.2 : un cache d'embeddings est un artefact dérivé de
    l'index, comme l'index est dérivé du modèle (ADR 0009). D'où trois règles :

    - il ne sert que l'index courant (`current_index`) : un nouvel index le vide,
      même si le nom du modèle n'a pas changé (préfixes modifiés, moteur qui ne
      fournit pas d'empreinte des poids…) ;
    - il ne garde que des vecteurs du modèle et de la dimension de cet index :
      sinon, une question posée pendant que le service servait un autre modèle
      resterait en erreur même après le retour du service au bon modèle ;
    - les documents ne sont jamais mis en cache : une (ré)indexation doit refléter
      le modèle servi *maintenant*, pas celui d'une indexation précédente.

    Garanti : le cache ne mélange jamais deux index. Pas garanti : une question déjà
    posée ne repart pas au service, donc c'est une question nouvelle (ou `status`, qui
    n'utilise pas ce cache) qui révèle un changement de modèle servi (ADR 0009).
    """

    def __init__(self, inner: Embedder, current_index: Callable[[], IndexManifest | None],
                 max_entries: int = 10_000) -> None:
        self._inner = inner
        self._current_index = current_index
        self._index: IndexManifest | None = None   # l'index que les entrées servent
        self._queries: dict[str, EmbeddingBatch] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()   # le serveur HTTP de l'application est multi-fil
        self.hits = 0
        self.misses = 0

    def embed_query(self, text: str) -> EmbeddingBatch:
        index = self._current_index()
        with self._lock:
            if index is not self._index:   # un nouvel index : un nouvel objet manifeste
                self._index = index
                self._queries.clear()
            cached = self._queries.get(text)
            if cached is not None:
                self.hits += 1
                return cached
            self.misses += 1
        batch = self._inner.embed_query(text)
        if index is not None and (batch.model, batch.dimension) == (index.embedding_model, index.dimension):
            with self._lock:
                if index is self._index and text not in self._queries:   # index inchangé pendant l'appel
                    if len(self._queries) >= self._max_entries:
                        self._queries.pop(next(iter(self._queries)))  # le plus ancien sort
                    self._queries[text] = batch
        return batch

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._inner.embed_documents(texts)


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
            self._log.warning("génération rejetée ou échouée après %.0f ms : %s",
                              (time.perf_counter() - start) * 1000, error)
            raise
        self._log.info("génération · %s · %d caractères · %.0f ms",
                       generation.model, len(generation.text), (time.perf_counter() - start) * 1000)
        return generation


# --- Nouvelles tentatives ----------------------------------------------------

def _retry(action: Callable[[], object], attempts: int, delay: float,
           sleep: Callable[[float], None], logger: logging.Logger = log):
    last: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return action()
        except AIServiceError as error:
            last = error
            if not error.transient:
                raise  # modèle inconnu, requête refusée : réessayer ne changera rien
            if attempt < attempts:
                logger.warning("service IA en erreur (%s), nouvelle tentative %d/%d",
                               error, attempt + 1, attempts)
                sleep(delay * (attempt + 1))
    assert last is not None
    raise last


class RetryingEmbedder:
    """Réessaie quand le service IA est injoignable ou en erreur passagère (5xx),
    pas quand la requête est refusée (4xx) ni quand la réponse est invalide."""

    def __init__(self, inner: Embedder, attempts: int = 1, delay_seconds: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep, logger: logging.Logger = log) -> None:
        self._inner = inner
        self._attempts = attempts
        self._delay = delay_seconds
        self._sleep = sleep
        self._log = logger

    def embed_query(self, text: str) -> EmbeddingBatch:
        return _retry(lambda: self._inner.embed_query(text), self._attempts, self._delay, self._sleep, self._log)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return _retry(lambda: self._inner.embed_documents(texts), self._attempts, self._delay, self._sleep, self._log)


class RetryingGenerator:
    def __init__(self, inner: Generator, attempts: int = 1, delay_seconds: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep, logger: logging.Logger = log) -> None:
        self._inner = inner
        self._attempts = attempts
        self._delay = delay_seconds
        self._sleep = sleep
        self._log = logger

    def generate(self, request: GenerationRequest) -> Generation:
        return _retry(lambda: self._inner.generate(request), self._attempts, self._delay, self._sleep, self._log)
