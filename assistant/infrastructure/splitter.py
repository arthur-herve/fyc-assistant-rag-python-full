"""Adaptateur : découpage des documents en morceaux.

Les paramètres de découpage changent les réponses (principe CACE, séquence
3.2) : ils sont donc enregistrés dans le manifeste de l'index.

`max_chars` borne le texte du morceau, en caractères ; avec `include_title`,
le titre du document s'y ajoute (il aide la recherche) : un morceau peut donc
dépasser `max_chars` de la longueur du titre.
"""

from __future__ import annotations

import re

from assistant.domain.model import Chunk, Document


class ParagraphSplitter:
    def __init__(self, max_chars: int = 800, overlap_chars: int = 120,
                 include_title: bool = True) -> None:
        if max_chars < 100:
            raise ValueError("max_chars doit valoir au moins 100")
        if not 0 <= overlap_chars < max_chars // 2:
            raise ValueError("overlap_chars doit être compris entre 0 et max_chars / 2")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.include_title = include_title

    def describe(self) -> dict:
        return {
            "type": "paragraph",
            "max_chars": self.max_chars,
            "overlap_chars": self.overlap_chars,
            "include_title": self.include_title,
        }

    def split(self, document: Document) -> list[Chunk]:
        pieces = self._pieces(document.text)
        chunks: list[Chunk] = []
        current = ""
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > self.max_chars:
                chunks.append(self._chunk(document, current, len(chunks)))
                current = self._tail(current)
            current = f"{current}\n\n{piece}" if current else piece
        if current.strip():
            chunks.append(self._chunk(document, current, len(chunks)))
        return chunks

    def _pieces(self, text: str) -> list[str]:
        """Paragraphes, eux-mêmes redécoupés s'ils dépassent la taille maximale."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        budget = self.max_chars - self.overlap_chars - 2
        pieces: list[str] = []
        for paragraph in paragraphs:
            while len(paragraph) > budget:
                cut = paragraph.rfind(" ", 0, budget)
                cut = cut if cut > budget // 2 else budget
                pieces.append(paragraph[:cut].strip())
                paragraph = paragraph[cut:].strip()
            if paragraph:
                pieces.append(paragraph)
        return pieces

    def _tail(self, text: str) -> str:
        if self.overlap_chars == 0:
            return ""
        tail = text[-self.overlap_chars :]
        space = tail.find(" ")
        return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail

    def _chunk(self, document: Document, text: str, position: int) -> Chunk:
        body = text.strip()
        if self.include_title:
            body = f"{document.title}\n{body}"
        return Chunk(
            id=f"{document.id}#{position}",
            document_id=document.id,
            document_title=document.title,
            text=body,
            position=position,
            allowed_groups=document.allowed_groups,
        )
