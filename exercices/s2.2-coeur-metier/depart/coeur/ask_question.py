"""Cas d'usage : répondre à une question à partir des documents accessibles. À ÉCRIRE.

Il ne connaît que les ports (coeur/ports.py) et le domaine : aucune IA, aucun
réseau, aucun fichier. C'est ce qui permet de le tester entièrement avec les
doubles de fakes.py (tests/test_ask_question.py).

Déroulé attendu, du plus déterministe au moins déterministe :

1. la question vide est refusée (EmptyQuestionError) ;
2. sans index construit : IndexNotBuiltError ;
3. la question est vectorisée ; si le modèle ou la dimension renvoyés ne
   correspondent pas au manifeste de l'index : IndexModelMismatchError
   (c'est le second verdict de la problématique : on ne contourne pas) ;
4. recherche des `top_k` passages, filtrée par les droits d'accès
   (AccessPolicy.can_read) — le filtre est passé à l'index, un passage
   interdit n'entre jamais dans le prompt ;
5. seuil de pertinence : sans passage au-dessus de `min_score`, on renvoie
   NO_RELEVANT_SOURCE sans appeler le modèle (trace : generation_model=None,
   prompt_version=None, attempts=0) ;
6. prompt : template.render(question=..., passages=format_passages(relevant)),
   passages numérotés [1], [2]… dans l'ordre des scores ;
7. génération, `max_attempts` fois au plus, avec seed, seed+1… si une graine
   est fournie ; après chaque tentative, check_citations(text, len(relevant)) ;
   à la première sortie valide : ANSWERED, sources = passages cités ;
8. sinon : UNSOURCED, texte UNSOURCED_MESSAGE (la sortie non sourcée n'est
   PAS montrée), sources = tous les passages pertinents, attempts = max.

Chaque réponse porte une AnswerTrace complète (index_id, modèles, version du
prompt, passages retrouvés avec leurs scores, seuil, tentatives, sorties brutes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from coeur.access import AccessPolicy
from coeur.citations import check_citations
from coeur.errors import EmptyQuestionError, IndexModelMismatchError, IndexNotBuiltError
from coeur.model import (
    NO_RELEVANT_SOURCE_MESSAGE,
    UNSOURCED_MESSAGE,
    Answer,
    AnswerStatus,
    AnswerTrace,
    Passage,
    Source,
    User,
)
from coeur.ports import Embedder, GenerationRequest, Generator, PromptRepository, VectorIndex


@dataclass(frozen=True)
class AskSettings:
    top_k: int = 4
    # Attention : ce seuil n'a de sens que pour UN modèle d'embeddings donné.
    min_score: float = 0.35
    max_attempts: int = 2
    temperature: float = 0.2
    max_tokens: int = 400
    seed: int | None = None
    prompt_name: str = "answer"


def format_passages(passages: Sequence[Passage]) -> str:
    """« [1] Titre\ntexte » pour chaque passage, séparés par une ligne vide."""
    raise NotImplementedError("à écrire : séquence 2.2")


class AskQuestion:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        generator: Generator,
        prompts: PromptRepository,
        settings: AskSettings = AskSettings(),
        access_policy: AccessPolicy | None = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._generator = generator
        self._prompts = prompts
        self._settings = settings
        self._access = access_policy or AccessPolicy()

    def execute(self, user: User, question: str) -> Answer:
        raise NotImplementedError("à écrire : séquence 2.2")
