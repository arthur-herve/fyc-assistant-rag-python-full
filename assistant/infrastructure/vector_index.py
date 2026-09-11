"""Adaptateurs : index vectoriel en mémoire et index persisté en JSON.

Volontairement simple (quelques centaines de morceaux) : la recherche est
une similarité cosinus exhaustive. Une vraie base vectorielle se brancherait
derrière le même port.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from assistant.application.ports import IndexManifest
from assistant.domain.model import Chunk, Passage


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._manifest: IndexManifest | None = None
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def manifest(self) -> IndexManifest | None:
        return self._manifest

    def replace(self, manifest: IndexManifest, chunks: Sequence[Chunk],
                vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("autant de vecteurs que de morceaux sont attendus")
        if any(len(v) != manifest.dimension for v in vectors):
            raise ValueError(f"tous les vecteurs doivent avoir {manifest.dimension} dimensions")
        self._manifest = manifest
        self._chunks = list(chunks)
        self._vectors = [_normalize(v) for v in vectors]

    def search(self, vector: Sequence[float], top_k: int,
               predicate: Callable[[Chunk], bool]) -> list[Passage]:
        query = _normalize(vector)
        scored = [
            Passage(chunk=chunk, score=sum(a * b for a, b in zip(query, stored)))
            for chunk, stored in zip(self._chunks, self._vectors)
            if predicate(chunk)
        ]
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:top_k]


class JsonVectorIndex(InMemoryVectorIndex):
    """Index persisté dans un fichier JSON lisible (pratique pour l'enseignement)."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        manifest = IndexManifest(**data["manifest"])
        chunks = [
            Chunk(**{**c, "allowed_groups": frozenset(c["allowed_groups"])})
            for c in data["chunks"]
        ]
        super().replace(manifest, chunks, data["vectors"])

    def manifest(self) -> IndexManifest | None:
        self._load()
        return super().manifest()

    def search(self, vector, top_k, predicate):
        self._load()
        return super().search(vector, top_k, predicate)

    def replace(self, manifest, chunks, vectors) -> None:
        super().replace(manifest, chunks, vectors)
        self._loaded = True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": asdict(manifest),
            "chunks": [
                {**asdict(c), "allowed_groups": sorted(c.allowed_groups)} for c in chunks
            ],
            "vectors": [[round(x, 6) for x in v] for v in self._vectors],
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)
