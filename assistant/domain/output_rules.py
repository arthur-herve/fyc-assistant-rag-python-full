"""Règle métier : à quoi doit ressembler une réponse avant d'être montrée.

Le modèle est probabiliste et bavard ; cette vérification ne l'est pas. Elle
complète `citations.py` (la forme des citations) par le fond : une réponse
vide, trop longue, dans la mauvaise langue ou qui déverse un raisonnement
n'est pas une réponse, même si elle contient « [1] ».

Ces règles ne nomment aucun modèle, mais leurs marqueurs de raisonnement
viennent des modèles rencontrés (qwen3…) : c'est une connaissance du modèle,
assumée, datée et testée sur les cas réels (ADR 0008), à revoir quand on en
change. Ce qui se règle ou se retire mécaniquement est neutralisé côté service IA :
balises <think> (retirées quel que soit le moteur), budget de réflexion (Ollama). Leçon du 11/09/2026 : un raisonnement en anglais contenant « [1] » avait
passé la vérification des citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FRENCH_MARKERS = frozenset("le la les des une un du de et est pas pour vous votre dans par sur au aux".split())
ENGLISH_MARKERS = frozenset("the is are and user let's okay this that with for answer should".split())
# Marqueurs génériques d'un raisonnement déversé : balise de réflexion, monologue en anglais.
# Bornés par des mots : « le billet me semble clair » ne contient pas « let me ».
_REASONING = re.compile(
    r"</?think>|\b(?:okay|ok),\s+(?:let|so)\b|\blet me\b|\bthe user is asking\b"
    r"|\bfirst,\s+i need\b|\bwait,\s",
    re.IGNORECASE,
)


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
    found = _REASONING.search(stripped)
    if found:
        problems.append(f"raisonnement du modèle déversé dans la réponse (« {found.group(0).strip()} »)")
    words = _words(stripped)
    if len(words) >= 5:
        french = sum(w in FRENCH_MARKERS for w in words)
        english = sum(w in ENGLISH_MARKERS for w in words)
        if english > french and english >= 3:
            problems.append("réponse dans une autre langue que le français")
    return OutputCheck(tuple(problems))
