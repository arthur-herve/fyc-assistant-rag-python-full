"""Adaptateurs : index vectoriel en mémoire et index persisté en JSON.

Volontairement simple (quelques centaines de morceaux) : la recherche est
une similarité cosinus exhaustive. Une vraie base vectorielle se brancherait
derrière le même port.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, NamedTuple, Sequence, TypeVar

from assistant.application.errors import IndexReplacedError, IndexWriteError
from assistant.application.ports import IndexManifest
from assistant.domain.model import Chunk, Passage

from .json_text import parse
from .text_files import decode_utf8

_T = TypeVar("_T")

# Sous Windows, un fichier ouvert par un autre processus ne se remplace pas, et ne se lit pas pendant
# qu'on le remplace : ces collisions durent quelques millisecondes, on réessaie brièvement.
_SHARING_DELAYS = (0.01, 0.02, 0.05, 0.1, 0.2)
# Un rédacteur renomme son fichier temporaire moins d'une seconde après l'avoir écrit, nouvelles tentatives
# comprises. Plus vieux qu'une heure, c'est celui d'un rédacteur tué au milieu d'une écriture : la prochaine
# écriture réussie le supprime.
_DELETED_AFTER = 3600.0   # secondes


class IndexUnreadableError(ValueError):
    """Le fichier d'index existe mais ne se lit pas : ce n'est pas « aucun index »."""


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def _check(manifest: IndexManifest, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("autant de vecteurs que de morceaux sont attendus")
    if any(len(v) != manifest.dimension for v in vectors):
        raise ValueError(f"tous les vecteurs doivent avoir {manifest.dimension} dimensions")


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
        _check(manifest, chunks, vectors)
        self._state = _State(manifest, list(chunks), [_normalize(v) for v in vectors])

    def search(self, vector: Sequence[float], top_k: int,
               predicate: Callable[[Chunk], bool], index_id: str | None = None) -> list[Passage]:
        state = self._state
        if index_id is not None and (state.manifest is None or state.manifest.index_id != index_id):
            # Vérifié sur l'état même qu'on va interroger : jamais un autre index que celui contrôlé.
            raise IndexReplacedError()
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
    réindexation faite par la ligne de commande, droits d'accès compris, et un fichier
    supprimé (plus d'index), comme la ligne de commande.
    """

    def __init__(self, path: str | Path, sleep: Callable[[float], None] = time.sleep) -> None:
        super().__init__()
        self._path = Path(path)
        self._sleep = sleep
        # Version du fichier chargé : date, taille et numéro de fichier. La date seule ne suffit pas :
        # deux réécritures rapides peuvent tomber dans la même tranche d'horloge du disque.
        # Limite connue, théorique, laissée telle quelle : sous Linux, le numéro d'un fichier remplacé peut
        # être réattribué aussitôt. Si deux réécritures suivent l'index chargé dans sa tranche d'horloge et
        # que la seconde, de même taille, reprend son numéro, la version ne change pas et cette réécriture
        # n'est pas relue. Comparer aussi le début du fichier (le manifeste) s'en garderait.
        self._version: tuple[int, int, int] | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            try:
                version = self._shared(lambda: self._file_version(self._path))
                if version == self._version:
                    return
                raw = self._shared(self._path.read_bytes)
            except FileNotFoundError:
                # Fichier supprimé : plus d'index, comme le voit la ligne de commande.
                self._state, self._version = _State(None, [], []), None
                return
            try:
                # Illisible : octets qui ne sont pas de l'UTF-8 strict (BOM accepté), JSON qui n'est pas strict
                # (chaînes et entiers compris), champ manquant, ou en trop dans le manifeste ou un morceau, découpage
                # qui n'est pas un objet, vecteurs en nombre ou de dimension inattendus. Les types des autres valeurs
                # ne sont pas tous vérifiés.
                data = parse(decode_utf8(raw), strings=True)
                manifest = IndexManifest(**data["manifest"])
                if not isinstance(manifest.splitter, dict):   # status et l'API le lisent comme un dictionnaire
                    raise ValueError("« splitter » doit être un objet")
                chunks = [
                    Chunk(**{**c, "allowed_groups": frozenset(c["allowed_groups"])})
                    for c in data["chunks"]
                ]
                super().replace(manifest, chunks, data["vectors"])
            except (ValueError, KeyError, TypeError) as error:
                raise IndexUnreadableError(f"index illisible ({self._path}) : {error}") from error
            self._version = version

    @staticmethod
    def _file_version(path: Path) -> tuple[int, int, int]:
        # Chaque écriture crée un nouveau fichier (le temporaire) : un autre numéro que celui du fichier qu'il
        # remplace, mais sous Linux, parfois celui d'un index plus ancien (voir la limite dans __init__).
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def _shared(self, action: Callable[[], _T]) -> _T:
        for delay in _SHARING_DELAYS:
            try:
                return action()
            except PermissionError:
                self._sleep(delay)
        return action()

    def manifest(self) -> IndexManifest | None:
        self._load()
        return super().manifest()

    def search(self, vector, top_k, predicate, index_id=None):
        if self._version is None:   # jamais chargé ; sinon, pas de relecture : index_id est vérifié sur l'état chargé
            self._load()
        return super().search(vector, top_k, predicate, index_id)

    def replace(self, manifest, chunks, vectors) -> None:
        # En mémoire, exactement ce qui est écrit : mêmes scores avant et après un redémarrage.
        stored = [[round(x, 6) for x in _normalize(v)] for v in vectors]
        _check(manifest, chunks, stored)
        payload = {
            "manifest": asdict(manifest),
            "chunks": [
                {**asdict(c), "allowed_groups": sorted(c.allowed_groups)} for c in chunks
            ],
            "vectors": stored,
        }
        with self._lock:
            # Le fichier d'abord, la mémoire ensuite : si l'écriture échoue, l'index en service reste
            # le précédent, en mémoire comme sur le disque. Un fichier temporaire par écriture : deux
            # réindexations simultanées n'écrivent jamais dans le même.
            tmp = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex}.tmp")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                # Version lue avant le renommage, qui garde date, taille et numéro : celle de NOTRE
                # fichier, même si un autre processus le remplace aussitôt (il sera alors relu).
                version = self._file_version(tmp)
                self._shared(lambda: self._move_into_place(tmp))
            except OSError as error:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
                raise IndexWriteError(str(self._path), str(error)) from error
            super().replace(manifest, chunks, stored)
            self._version = version
            self._delete_abandoned_temporaries()

    def _move_into_place(self, tmp: Path) -> None:
        """Met le fichier écrit à la place de l'index, d'un coup : un lecteur voit l'ancien ou le nouveau."""
        os.replace(tmp, self._path)

    def _delete_abandoned_temporaries(self) -> None:
        """Supprime, après une écriture réussie, les fichiers temporaires abandonnés depuis plus d'une heure par des
        rédacteurs tués. Celui d'un rédacteur actif est récent : il n'est jamais touché. Seuls les
        noms qu'écrit `replace` sont examinés : `index.json.<id>.tmp`, <id> de 32 chiffres hexadécimaux (uuid4().hex).
        L'ancien index que met de côté le File.Replace de la version C# (….old.tmp) reste à la charge de celle-ci."""
        temporary = re.compile(rf"{re.escape(self._path.name)}\.[0-9a-f]{{32}}\.tmp")
        try:
            files = [file for file in self._path.parent.iterdir() if temporary.fullmatch(file.name)]
        except OSError:
            return   # dossier illisible un instant : la prochaine écriture s'en chargera
        for file in files:
            with contextlib.suppress(OSError):   # disparu entre-temps, ou qui ne se supprime pas : il ne gêne personne
                if time.time() - file.stat().st_mtime > _DELETED_AFTER:
                    file.unlink()
