"""Règle métier : à quoi doit ressembler une réponse avant d'être montrée.

Le modèle est probabiliste et bavard ; cette vérification ne l'est pas. Elle
complète `citations.py` (la forme des citations) par le fond : une réponse
vide, trop longue, dans la mauvaise langue ou qui déverse un raisonnement
n'est pas une réponse, même si elle contient « [1] ».

Leçon du 11/09/2026 : le raisonnement en anglais de qwen3 (« Okay, let's see…
passage [1]… ») avait passé la vérification des citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FRENCH_MARKERS = frozenset("le la les des une un du de et est pas pour vous votre dans par sur au aux".split())
ENGLISH_MARKERS = frozenset("the is are and user let's okay this that with for question passage answer should".split())
REASONING_MARKERS = ("<think>", "okay, let", "okay, so", "let me ", "the user is asking",
                     "first, i need", "wait, ")


@dataclass(frozen=True)
class OutputCheck:
    problems: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.problems


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())


def check_output(text: str, max_chars: int = 1500) -> OutputCheck:
    problems: list[str] = []
    stripped = text.strip()
    if not stripped:
        return OutputCheck(("réponse vide",))
    if len(stripped) > max_chars:
        problems.append(f"réponse trop longue ({len(stripped)} caractères, {max_chars} au plus)")
    lowered = stripped.lower()
    found = [m for m in REASONING_MARKERS if m in lowered]
    if found:
        problems.append(f"raisonnement du modèle déversé dans la réponse (« {found[0].strip()} »)")
    words = _words(stripped)
    if len(words) >= 8:
        french = sum(w in FRENCH_MARKERS for w in words)
        english = sum(w in ENGLISH_MARKERS for w in words)
        if english > french and english >= 3:
            problems.append("réponse dans une autre langue que le français")
    return OutputCheck(tuple(problems))
