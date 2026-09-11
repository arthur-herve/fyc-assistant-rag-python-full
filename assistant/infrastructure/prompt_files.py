"""Adaptateur : prompts stockés dans des fichiers TOML versionnés avec le code.

La version tracée combine la version déclarée et une empreinte du contenu :
modifier un prompt sans changer sa version reste détectable dans les traces.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from assistant.application.ports import PromptTemplate


class FilePromptRepository:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def get(self, name: str) -> PromptTemplate:
        path = self._directory / f"{name}.toml"
        # Normalise les fins de ligne : l'empreinte est la même sous Windows et Linux.
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        data = tomllib.loads(text)
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return PromptTemplate(
            name=name,
            version=f"{data['version']}+{fingerprint}",
            system=data["system"].strip(),
            user=data["user"].strip(),
        )
