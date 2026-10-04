"""Adaptateur : prompts stockés dans des fichiers TOML versionnés avec le code.

La version tracée combine la version déclarée et une empreinte du contenu :
modifier un prompt sans changer sa version reste détectable dans les traces.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from assistant.application.errors import PromptNotFoundError
from assistant.application.ports import PromptTemplate

from .text_files import read_utf8


class FilePromptRepository:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def get(self, name: str) -> PromptTemplate:
        path = self._directory / f"{name}.toml"
        try:
            # read_utf8 normalise déjà les fins de ligne (mode texte) : l'empreinte est la même sous Windows et Linux.
            data = tomllib.loads(read_utf8(path))
        except FileNotFoundError:
            # Fichier ou dossier absent : le nom, le dossier et les prompts connus, plutôt que
            # « [Errno 2] No such file or directory ».
            raise PromptNotFoundError(name, str(self._directory), self._names()) from None
        except (ValueError, RecursionError) as error:
            # TOMLDecodeError est une ValueError, comme un fichier qui n'est pas en UTF-8 ; RecursionError :
            # des tableaux imbriqués sur des milliers de niveaux.
            raise ValueError(f"prompt illisible ({path}) : {error}") from error
        values = [data.get(key) for key in ("version", "system", "user")]
        if not all(isinstance(value, str) for value in values):
            # Les prompts sont édités à la main (docs/artefacts.md) : dire quoi corriger, et où.
            raise ValueError(f"prompt illisible ({path}) : il faut trois textes, version, system et user")
        version, system, user = values[0], values[1].strip(), values[2].strip()
        return PromptTemplate(
            name=name,
            version=f"{version}+{prompt_fingerprint(version, system, user)}",
            system=system,
            user=user,
        )

    def _names(self) -> list[str]:
        """Les noms des prompts du dossier, triés comme ceux des instantanés (JsonSnapshotStore.names) : des
        fichiers seulement ; un dossier « x.toml » n'en est pas un."""
        return sorted(p.stem for p in self._directory.glob("*.toml") if p.is_file())


def prompt_fingerprint(version: str, system: str, user: str) -> str:
    """Empreinte canonique d'un prompt : SHA-256 de « version, system, user » séparés
    par des sauts de ligne, 8 premiers caractères hexadécimaux.

    Elle porte sur le contenu, pas sur le fichier : le même prompt en TOML ici et en
    JSON dans la version C# porte la même version, et les instantanés des deux
    versions se comparent sans écart. Même formule que ``Files.cs`` côté C#.
    """
    canonical = f"{version}\n{system}\n{user}".replace("\r\n", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
