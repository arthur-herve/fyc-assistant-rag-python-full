"""Adaptateur : lit un dossier de fichiers Markdown avec un en-tête simple.

Format attendu :

    ---
    id: teletravail
    titre: Politique de télétravail
    groupes: tous
    ---
    Texte du document…

`groupes` est obligatoire (« tous » pour un document public) : un droit d'accès
oublié ou mal écrit est une erreur, jamais un document rendu public en silence.
"""

from __future__ import annotations

import re
from pathlib import Path

from assistant.domain.model import Document

_GROUP = re.compile(r"[a-z0-9][a-z0-9_-]*")


class CorpusFormatError(ValueError):
    pass


def parse_markdown_document(content: str, origin: str = "<texte>") -> Document:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise CorpusFormatError(f"{origin} : en-tête '---' manquant")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise CorpusFormatError(f"{origin} : en-tête non refermé") from None

    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise CorpusFormatError(f"{origin} : ligne d'en-tête invalide « {line} »")
        meta[key.strip().lower()] = value.strip()

    if not meta.get("id"):
        raise CorpusFormatError(f"{origin} : champ 'id' obligatoire")
    groups = frozenset(g.strip() for g in meta.get("groupes", "").split(",") if g.strip())
    if not groups:
        raise CorpusFormatError(f"{origin} : champ 'groupes' obligatoire (« tous » pour un document public)")
    invalid = sorted(g for g in groups if not _GROUP.fullmatch(g))
    if invalid:
        raise CorpusFormatError(
            f"{origin} : groupe(s) invalide(s) {invalid} : des noms en minuscules séparés par des "
            "virgules, sans commentaire"
        )
    return Document(
        id=meta["id"],
        title=meta.get("titre", meta["id"]),
        text="\n".join(lines[end + 1 :]).strip(),
        allowed_groups=groups,
    )


class MarkdownCorpus:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def load(self) -> list[Document]:
        if not self._directory.is_dir():
            raise CorpusFormatError(f"Dossier de corpus introuvable : {self._directory}")
        documents = [
            parse_markdown_document(path.read_text(encoding="utf-8-sig"), str(path))
            for path in sorted(self._directory.glob("*.md"))
        ]
        ids = [d.id for d in documents]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise CorpusFormatError(f"Identifiants de documents en double : {sorted(duplicates)}")
        return documents
