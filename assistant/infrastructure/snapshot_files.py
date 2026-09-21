"""Adaptateur : instantanés conservés en fichiers JSON lisibles, un par nom."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from assistant.application.errors import SnapshotNotFoundError
from assistant.application.ports import Snapshot, SnapshotEntry
from assistant.application.snapshots import SNAPSHOT_NAME, InvalidSnapshotNameError

__all__ = ["JsonSnapshotStore", "SnapshotNotFoundError"]

_TEXT_FIELDS = ("question_id", "user_id", "question", "status", "text")


def _text(values: dict, key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise ValueError(f"champ « {key} » manquant ou non textuel")
    return value


def _entry(values: object) -> SnapshotEntry:
    """Une réponse relue, types vérifiés comme en C# : « abc » n'est pas une liste de documents.
    Les champs inconnus (instantané écrit par une version plus récente) sont ignorés."""
    if not isinstance(values, dict):
        raise ValueError("chaque réponse doit être un objet JSON")
    cited = values.get("cited_documents")
    if not isinstance(cited, list) or not all(isinstance(d, str) for d in cited):
        raise ValueError("« cited_documents » doit être une liste de textes")
    attempts = values.get("attempts", 0)
    if not isinstance(attempts, int) or isinstance(attempts, bool):
        raise ValueError("« attempts » doit être un entier")
    question_id, user_id, question, status, text = (_text(values, key) for key in _TEXT_FIELDS)
    return SnapshotEntry(question_id, user_id, question, status, tuple(cited), text, attempts)


class JsonSnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def _path(self, name: str) -> Path:
        if not SNAPSHOT_NAME.fullmatch(name):
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
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("un objet JSON est attendu")
            configuration = data.get("configuration", {})
            if not isinstance(configuration, dict):
                raise ValueError("« configuration » doit être un objet")
            entries = data.get("entries")
            if not isinstance(entries, list):
                raise ValueError("champ « entries » manquant ou qui n'est pas une liste")
            return Snapshot(_text(data, "name"), _text(data, "created_at"), configuration,
                            tuple(_entry(e) for e in entries))
        except ValueError as error:   # JSONDecodeError en est une
            raise ValueError(f"instantané illisible ({path}) : {error}") from error

    def names(self) -> list[str]:
        if not self._directory.is_dir():
            return []
        return sorted(p.stem for p in self._directory.glob("*.json"))
