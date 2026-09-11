"""Règle métier : qui peut lire quoi.

Le contrôle d'accès est appliqué de façon déterministe AVANT l'appel au
modèle. On ne demande jamais au modèle de « ne pas révéler » un document :
un passage interdit n'entre tout simplement pas dans le prompt.
"""

from __future__ import annotations

from .model import Chunk, User

PUBLIC_GROUP = "tous"


class AccessPolicy:
    def can_read(self, user: User, chunk: Chunk) -> bool:
        if PUBLIC_GROUP in chunk.allowed_groups:
            return True
        return bool(user.groups & chunk.allowed_groups)
