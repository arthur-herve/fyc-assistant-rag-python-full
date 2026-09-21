"""Cas d'usage : retrouver les passages qu'un utilisateur a le droit de lire.

C'est la moitié déterministe de la chaîne (à modèle d'embeddings fixé) : elle sert
à `AskQuestion`, mais aussi au banc d'essai, qui mesure la recherche seule pour
calibrer le seuil de pertinence sans appeler le générateur.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.domain.access import AccessPolicy
from assistant.domain.model import Passage, User

from .errors import IndexModelMismatchError, IndexNotBuiltError, IndexReplacedError
from .ports import Embedder, IndexManifest, VectorIndex


@dataclass(frozen=True)
class Retrieval:
    """Résultat d'une recherche : les passages accessibles et l'index qui les a fournis."""

    manifest: IndexManifest
    passages: list[Passage]


class SearchPassages:
    def __init__(self, embedder: Embedder, index: VectorIndex,
                 access_policy: AccessPolicy | None = None) -> None:
        self._embedder = embedder
        self._index = index
        self._access = access_policy or AccessPolicy()

    def execute(self, user: User, question: str, top_k: int) -> Retrieval:
        """Vérifie que l'index existe et qu'il a été construit avec le modèle qui a produit
        les vecteurs de la question (sinon `IndexModelMismatchError` : réindexer, jamais
        contourner), puis cherche les `top_k` passages les plus proches parmi ceux que
        l'utilisateur peut lire. Les droits sont filtrés ici, avant tout prompt.

        L'index peut être reconstruit par un autre processus entre la vérification et la
        recherche (`serve` relit le fichier quand il change) : on vérifie après coup que
        l'index interrogé est bien celui qu'on a contrôlé, sinon on recommence une fois
        (y compris quand la recherche échoue parce que le nouvel index a une autre dimension)."""
        for _ in range(2):
            manifest = self._index.manifest()
            if manifest is None:
                raise IndexNotBuiltError()

            query = self._embedder.embed_query(question)
            if (query.model, query.dimension) != (manifest.embedding_model, manifest.dimension):
                raise IndexModelMismatchError(
                    manifest.embedding_model, manifest.dimension, query.model, query.dimension
                )

            try:
                passages = self._index.search(
                    query.vectors[0], top_k, predicate=lambda chunk: self._access.can_read(user, chunk)
                )
            except ValueError:
                if self._unchanged(manifest):
                    raise   # même index : une vraie erreur
                continue
            if self._unchanged(manifest):
                return Retrieval(manifest, passages)
        raise IndexReplacedError()

    def _unchanged(self, manifest: IndexManifest) -> bool:
        current = self._index.manifest()
        return current is not None and current.index_id == manifest.index_id
