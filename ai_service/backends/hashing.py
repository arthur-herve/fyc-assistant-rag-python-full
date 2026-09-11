"""Embeddings hors-ligne : sac de mots haché, déterministe, sans dépendance.

Ce n'est pas un modèle appris : il ne capte que le vocabulaire commun
entre la question et le texte. Il sert au mode hors-ligne et aux tests, et
de point de comparaison « plancher » dans le banc d'essai.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Sequence

from .base import Vectors

_STOPWORDS = frozenset(
    "au aux avec ce ces cette dans de des du elle en est et il ils je la le les leur "
    "lui ma mais me mes mon ne nos notre nous on ou par pas pour qu que qui sa se ses "
    "son sont sur ta te tes ton tu un une vos votre vous est-ce quel quelle quels "
    "quelles combien comment faut peut dois doit".split()
)


def _features(text: str, stem: int) -> list[str]:
    ascii_text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    words = [w for w in re.findall(r"[a-z0-9]+", ascii_text) if len(w) > 1 and w not in _STOPWORDS]
    return [w[:stem] for w in words]


class HashingEmbeddingBackend:
    def __init__(self, dimension: int = 256, stem: int = 6) -> None:
        self._dimension = dimension
        self._stem = stem

    def embed(self, texts: Sequence[str]) -> Vectors:
        return Vectors(
            model_id=f"hashing-{self._dimension}-stem{self._stem}",
            dimension=self._dimension,
            vectors=[self._embed_one(t) for t in texts],
        )

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for feature in _features(text, self._stem):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "little") % self._dimension
            vector[slot] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
