"""Règle métier : toute réponse doit citer ses sources.

Le modèle est probabiliste, la vérification ne l'est pas : on contrôle que
la réponse contient au moins une citation et que chaque citation renvoie à
un passage réellement fourni.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Accepte [1], [2, 3] et [2,3]
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[int, ...]
    invalid: tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return bool(self.cited) and not self.invalid


def check_citations(text: str, passage_count: int) -> CitationCheck:
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    valid = tuple(n for n in numbers if 1 <= n <= passage_count)
    invalid = tuple(n for n in numbers if not 1 <= n <= passage_count)
    return CitationCheck(cited=valid, invalid=invalid)
