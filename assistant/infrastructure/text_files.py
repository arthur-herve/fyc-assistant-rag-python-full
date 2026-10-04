"""Adaptateur : textes lus par l'application (corpus, prompts, instantanés, index, réponses du service
IA ; configuration et jeux de questions, par la racine de composition).

Tous sont en UTF-8. La marque d'ordre des octets (BOM) qu'ajoutent certains éditeurs sous Windows
est acceptée ; un fichier enregistré dans un autre encodage (latin-1…) est une erreur qui dit où,
jamais un texte lu de travers.
"""

from __future__ import annotations

from pathlib import Path


def _not_utf8(error: UnicodeDecodeError) -> str:
    # L'octet fautif et sa position, comptée après la marque d'ordre des octets.
    return f"pas en UTF-8 (octet 0x{error.object[error.start]:02x} à la position {error.start})"


def decode_utf8(data: bytes) -> str:
    """Les octets en texte ; ValueError s'ils ne sont pas en UTF-8."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(_not_utf8(error)) from error


def read_utf8(path: Path) -> str:
    """Le texte du fichier (fins de ligne normalisées par le mode texte) ; ValueError s'il n'est
    pas en UTF-8. Le message ne nomme pas le fichier : l'appelant le préfixe."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(_not_utf8(error)) from error
