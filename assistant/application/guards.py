"""Décorateurs de ports côté application : la couche anticorruption (séquence 4.1).

Un décorateur implémente le même port que l'objet qu'il enveloppe ; le cas
d'usage ne sait pas s'il parle au modèle ou à un garde-fou. On ajoute ainsi
un comportement sans toucher ni au cœur, ni à l'adaptateur HTTP. L'assemblage
se fait uniquement dans `composition.py`.

Ceux qui portent une règle métier vivent ici, dans l'application ; ceux qui
sont purement techniques (cache, journal, nouvelles tentatives) vivent dans
`infrastructure/decorators.py`.
"""

from __future__ import annotations

from assistant.domain.output_rules import check_output

from .errors import ModelOutputRejectedError
from .ports import Generation, GenerationRequest, Generator


class OutputValidatingGenerator:
    """Refuse une sortie qui n'a pas la forme d'une réponse (vide, trop longue,
    autre langue, raisonnement déversé). Le cas d'usage compte le rejet comme
    une tentative ratée et peut réessayer."""

    def __init__(self, inner: Generator, max_chars: int = 1500) -> None:
        self._inner = inner
        self._max_chars = max_chars

    def generate(self, request: GenerationRequest) -> Generation:
        generation = self._inner.generate(request)
        check = check_output(generation.text, self._max_chars)
        if not check.is_valid:
            raise ModelOutputRejectedError(generation.model, generation.text, check.problems)
        return generation
