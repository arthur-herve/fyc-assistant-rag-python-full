"""Entités du domaine de l'assistant documentaire.

Ce module ne dépend d'aucune technologie d'IA, ni de HTTP, ni du stockage : il
décrit ce que manipule le métier (documents, utilisateurs, réponses sourcées).
Il sait seulement qu'une réponse vient de modèles identifiés (`AnswerTrace`) :
c'est voulu, c'est ce qui rend une réponse explicable (S4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Document:
    """Document interne, avec les groupes autorisés à le lire."""

    id: str
    title: str
    text: str
    allowed_groups: frozenset[str]


@dataclass(frozen=True)
class Chunk:
    """Morceau de document indexable. Il hérite des droits de son document."""

    id: str
    document_id: str
    document_title: str
    text: str
    position: int
    allowed_groups: frozenset[str]


@dataclass(frozen=True)
class User:
    id: str
    groups: frozenset[str]


@dataclass(frozen=True)
class Passage:
    """Morceau retrouvé pour une question, avec son score de similarité."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Source:
    """Source citée dans une réponse : le numéro renvoie au passage fourni au modèle."""

    number: int
    document_id: str
    document_title: str
    chunk_id: str


class AnswerStatus(Enum):
    ANSWERED = "answered"
    NO_RELEVANT_SOURCE = "no_relevant_source"
    UNSOURCED = "unsourced"


@dataclass(frozen=True)
class AnswerTrace:
    """Tout ce qu'il faut pour expliquer et reproduire une réponse (séquence 4.2)."""

    index_id: str
    embedding_model: str
    generation_model: str | None
    prompt_version: str | None
    retrieved: tuple[tuple[str, float], ...]
    min_score: float
    attempts: int
    raw_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Answer:
    question: str
    status: AnswerStatus
    text: str
    sources: tuple[Source, ...]
    trace: AnswerTrace


NO_RELEVANT_SOURCE_MESSAGE = (
    "Je n'ai trouvé aucun document accessible qui réponde à cette question."
)

UNSOURCED_MESSAGE = (
    "Je n'ai pas pu produire de réponse correctement sourcée. "
    "Consultez directement les passages ci-dessous."
)
