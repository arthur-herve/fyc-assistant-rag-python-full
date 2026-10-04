"""Règle métier n° 2 : toute réponse doit citer ses sources. À ÉCRIRE.

Le modèle est probabiliste, la vérification ne l'est pas : on contrôle que
la réponse contient au moins une citation et que chaque citation renvoie à
un passage réellement fourni.

Format des citations : [1], [2, 3] ou [2,3]. Comportement attendu (voir
tests/test_domain.py) :
- `cited` : les numéros valides (entre 1 et passage_count), dans l'ordre
  d'apparition, sans doublon ;
- `invalid` : les numéros hors de cet intervalle, nombres trop grands compris (des
  milliers de chiffres, que int() refuse de convertir) : une citation invalide, pas une
  exception ;
- `is_valid` : au moins une citation valide ET aucune citation invalide.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[int, ...]
    invalid: tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return bool(self.cited) and not self.invalid


def check_citations(text: str, passage_count: int) -> CitationCheck:
    raise NotImplementedError("à écrire : séquence 2.2")
