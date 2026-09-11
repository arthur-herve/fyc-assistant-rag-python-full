"""Règle métier n° 1 : qui peut lire quoi. À ÉCRIRE.

Le contrôle d'accès est appliqué de façon déterministe AVANT l'appel au
modèle. On ne demande jamais au modèle de « ne pas révéler » un document :
un passage interdit n'entre tout simplement pas dans le prompt.

Règle attendue (voir tests/test_domain.py) :
- un morceau dont les groupes autorisés contiennent « tous » est lisible par
  tout le monde, même un utilisateur sans groupe ;
- sinon, il faut au moins un groupe en commun entre l'utilisateur et le morceau.
"""

from __future__ import annotations

from coeur.model import Chunk, User

PUBLIC_GROUP = "tous"


class AccessPolicy:
    def can_read(self, user: User, chunk: Chunk) -> bool:
        raise NotImplementedError("à écrire : séquence 2.2")
