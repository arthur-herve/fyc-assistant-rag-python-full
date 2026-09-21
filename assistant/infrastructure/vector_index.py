"""Adaptateurs : index vectoriel en mémoire et index persisté en JSON.

Volontairement simple (quelques centaines de morceaux) : la recherche est
une similarité cosinus exhaustive. Une vraie base vectorielle se brancherait
derrière le même port.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

from assistant.application.ports import IndexManifest
from assistant.domain.model import Chunk, Passage


class IndexUnreadableError(ValueError):
    """Le fichier d'index existe mais ne se lit pas : ce n'est pas « aucun index »."""


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


class _State(NamedTuple):
    manifest: IndexManifest | None
    chunks: list[Chunk]
    vectors: list[list[float]]


class InMemoryVectorIndex:
    def __init__(self) -> None:
        # Un seul attribut, remplacé d'un coup : une recherche en cours ne voit jamais
        # le manifeste d'un index et les vecteurs d'un autre.
        self._state = _State(None, [], [])

    def manifest(self) -> IndexManifest | None:
        return self._state.manifest

    def replace(self, manifest: IndexManifest, chunks: Sequence[Chunk],
                vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("autant de vecteurs que de morceaux sont attendus")
        if any(len(v) != manifest.dimension for v in vectors):
            raise ValueError(f"tous les vecteurs doivent avoir {manifest.dimension} dimensions")
        self._state = _State(manifest, list(chunks), [_normalize(v) for v in vectors])

    def search(self, vector: Sequence[float], top_k: int,
               predicate: Callable[[Chunk], bool]) -> list[Passage]:
        state = self._state
        if state.manifest is not None and len(vector) != state.manifest.dimension:
            raise ValueError(f"vecteur de requête de {len(vector)} dimensions, "
                             f"l'index en attend {state.manifest.dimension}")
        query = _normalize(vector)
        scored = [
            Passage(chunk=chunk, score=sum(a * b for a, b in zip(query, stored)))
            for chunk, stored in zip(state.chunks, state.vectors)
            if predicate(chunk)
        ]
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:top_k]


class JsonVectorIndex(InMemoryVectorIndex):
    """Index persisté dans un fichier JSON lisible (pratique pour l'enseignement).

    Le fichier est relu quand il a changé, à la lecture du manifeste (que les cas
    d'usage consultent avant de chercher) : un processus long (`serve`) voit une
    réindexation faite par la ligne de commande, droits d'accès compris.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        # Version du fichier chargé : date, taille et numéro de fichier. La date seule ne suffit pas :
        # deux réécritures rapides peuvent tomber dans la même tranche d'horloge du disque.
        self._version: tuple[int, int, int] | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            try:
                version = self._file_version()
            except FileNotFoundError:
                return
            if version == self._version:
                return
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                manifest = IndexManifest(**data["manifest"])
                chunks = [
                    Chunk(**{**c, "allowed_groups": frozenset(c["allowed_groups"])})
                    for c in data["chunks"]
                ]
                super().replace(manifest, chunks, data["vectors"])
            except (ValueError, KeyError, TypeError) as error:
                raise IndexUnreadableError(f"index illisible ({self._path}) : {error}") from error
            self._version = version

    def _file_version(self) -> tuple[int, int, int]:
        stat = self._path.stat()   # os.replace donne un nouveau fichier, donc un nouveau numéro
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def manifest(self) -> IndexManifest | None:
        self._load()
        return super().manifest()

    def search(self, vector, top_k, predicate):
        if self._version is None:   # jamais chargé ; sinon, pas de relecture : SearchPassages vérifie après coup
            self._load()
        return super().search(vector, top_k, predicate)

    def replace(self, manifest, chunks, vectors) -> None:
        # En mémoire, exactement ce qui est écrit : mêmes scores avant et après un redémarrage.
        stored = [[round(x, 6) for x in _normalize(v)] for v in vectors]
        with self._lock:
            super().replace(manifest, chunks, stored)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "manifest": asdict(manifest),
                "chunks": [
                    {**asdict(c), "allowed_groups": sorted(c.allowed_groups)} for c in chunks
                ],
                "vectors": stored,
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)
            self._version = self._file_version()
