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
        version, system, user = data["version"], data["system"].strip(), data["user"].strip()
        return PromptTemplate(
            name=name,
            version=f"{version}+{prompt_fingerprint(version, system, user)}",
            system=system,
            user=user,
        )


def prompt_fingerprint(version: str, system: str, user: str) -> str:
    """Empreinte canonique d'un prompt : SHA-256 de « version, system, user » séparés
    par des sauts de ligne, 8 premiers caractères hexadécimaux.

    Elle porte sur le contenu, pas sur le fichier : le même prompt en TOML ici et en
    JSON dans la version C# porte la même version, et les instantanés des deux
    versions se comparent sans écart. Même formule que ``Files.cs`` côté C#.
    """
    canonical = f"{version}\n{system}\n{user}".replace("\r\n", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
