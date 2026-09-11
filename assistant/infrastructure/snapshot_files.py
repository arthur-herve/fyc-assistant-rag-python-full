"""Adaptateur : instantanés conservés en fichiers JSON lisibles, un par nom."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from assistant.application.ports import Snapshot, SnapshotEntry
from assistant.application.snapshots import SNAPSHOT_NAME, InvalidSnapshotNameError

_ENTRY_FIELDS = {f.name for f in fields(SnapshotEntry)}


class SnapshotNotFoundError(LookupError):
    pass


class JsonSnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def _path(self, name: str) -> Path:
        if not SNAPSHOT_NAME.match(name):
            raise InvalidSnapshotNameError(name)
        return self._directory / f"{name}.json"

    def save(self, snapshot: Snapshot) -> None:
        path = self._path(snapshot.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(snapshot)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, name: str) -> Snapshot:
        path = self._path(name)
        if not path.exists():
            raise SnapshotNotFoundError(f"instantané introuvable : {name} (connus : {self.names()})")
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(
            # Les champs inconnus (instantané écrit par une version plus récente) sont ignorés.
            SnapshotEntry(**{**{k: v for k, v in e.items() if k in _ENTRY_FIELDS},
                             "cited_documents": tuple(e["cited_documents"])})
            for e in data["entries"]
        )
        return Snapshot(data["name"], data["created_at"], data.get("configuration", {}), entries)

    def names(self) -> list[str]:
        if not self._directory.is_dir():
            return []
        return sorted(p.stem for p in self._directory.glob("*.json"))
