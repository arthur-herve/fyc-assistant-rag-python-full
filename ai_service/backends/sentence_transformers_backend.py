"""Backend optionnel : modèles d'embeddings Hugging Face exécutés en Python.

Nécessite `pip install sentence-transformers` (installe PyTorch, ~1 à 2 Go).
Fonctionne sur CPU. Le modèle est téléchargé au premier appel.
"""

from __future__ import annotations

import threading
from typing import Sequence

from .base import BackendError, Vectors


class SentenceTransformersEmbeddingBackend:
    def __init__(self, model: str, device: str | None = None, batch_size: int = 16) -> None:
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise BackendError(
                        "sentence-transformers n'est pas installé : "
                        "pip install -r requirements-ai-optional.txt"
                    ) from error
                self._model = SentenceTransformer(self._model_name, device=self._device)
            return self._model

    def embed(self, texts: Sequence[str]) -> Vectors:
        model = self._load()
        vectors = model.encode(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return Vectors(
            model_id=f"st:{self._model_name}", dimension=len(vectors[0]), vectors=vectors
        )
