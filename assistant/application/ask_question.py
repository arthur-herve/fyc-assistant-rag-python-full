"""Cas d'usage : répondre à une question à partir des documents accessibles.

Déroulé, du plus déterministe au moins déterministe :
1. règles métier sur la question (déterministe) ;
2. recherche des passages, filtrée par les droits d'accès (déterministe à
   modèle d'embeddings fixé) ;
3. seuil de pertinence : sans passage pertinent, on n'appelle pas le modèle ;
4. génération (probabiliste), encadrée par deux vérifications déterministes :
   la forme de la sortie (décorateur de validation, `guards.py`) et les citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from assistant.domain.access import AccessPolicy
from assistant.domain.citations import check_citations
from assistant.domain.errors import EmptyQuestionError
from assistant.domain.model import (
    NO_RELEVANT_SOURCE_MESSAGE,
    UNSOURCED_MESSAGE,
    Answer,
    AnswerStatus,
    AnswerTrace,
    Passage,
    Source,
    User,
)

from .errors import IndexModelMismatchError, IndexNotBuiltError, ModelOutputRejectedError
from .ports import Embedder, GenerationRequest, Generator, PromptRepository, VectorIndex


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
    blocks = []
    for number, passage in enumerate(passages, start=1):
        blocks.append(
            f"[{number}] {passage.chunk.document_title}\n{passage.chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


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
        question = question.strip()
        if not question:
            raise EmptyQuestionError()

        manifest = self._index.manifest()
        if manifest is None:
            raise IndexNotBuiltError()

        query = self._embedder.embed_query(question)
        if (query.model, query.dimension) != (manifest.embedding_model, manifest.dimension):
            raise IndexModelMismatchError(
                manifest.embedding_model, manifest.dimension, query.model, query.dimension
            )

        settings = self._settings
        passages = self._index.search(
            query.vectors[0],
            settings.top_k,
            predicate=lambda chunk: self._access.can_read(user, chunk),
        )
        relevant = [p for p in passages if p.score >= settings.min_score]
        retrieved = tuple((p.chunk.id, round(p.score, 4)) for p in passages)

        if not relevant:
            return Answer(
                question=question,
                status=AnswerStatus.NO_RELEVANT_SOURCE,
                text=NO_RELEVANT_SOURCE_MESSAGE,
                sources=(),
                trace=AnswerTrace(
                    index_id=manifest.index_id,
                    embedding_model=manifest.embedding_model,
                    generation_model=None,
                    prompt_version=None,
                    retrieved=retrieved,
                    min_score=settings.min_score,
                    attempts=0,
                ),
            )

        template = self._prompts.get(settings.prompt_name)
        prompt = template.render(question=question, passages=format_passages(relevant))

        raw_outputs: list[str] = []
        generation_model = None
        for attempt in range(1, settings.max_attempts + 1):
            seed = None if settings.seed is None else settings.seed + attempt - 1
            try:
                generation = self._generator.generate(
                    GenerationRequest(
                        system=template.system,
                        prompt=prompt,
                        temperature=settings.temperature,
                        max_tokens=settings.max_tokens,
                        seed=seed,
                    )
                )
            except ModelOutputRejectedError as rejected:
                # Garde-fou de forme (décorateur de validation) : tentative ratée, tracée.
                generation_model = rejected.model
                raw_outputs.append(f"<rejetée : {' ; '.join(rejected.problems)}> {rejected.text}")
                continue
            generation_model = generation.model
            raw_outputs.append(generation.text)
            check = check_citations(generation.text, len(relevant))
            if check.is_valid:
                sources = tuple(
                    Source(
                        number=n,
                        document_id=relevant[n - 1].chunk.document_id,
                        document_title=relevant[n - 1].chunk.document_title,
                        chunk_id=relevant[n - 1].chunk.id,
                    )
                    for n in check.cited
                )
                return Answer(
                    question=question,
                    status=AnswerStatus.ANSWERED,
                    text=generation.text.strip(),
                    sources=sources,
                    trace=self._trace(
                        manifest.index_id, manifest.embedding_model, generation_model,
                        template.version, retrieved, attempt, raw_outputs,
                    ),
                )

        # Règle métier : pas de réponse non sourcée. On renvoie les passages.
        sources = tuple(
            Source(
                number=n,
                document_id=p.chunk.document_id,
                document_title=p.chunk.document_title,
                chunk_id=p.chunk.id,
            )
            for n, p in enumerate(relevant, start=1)
        )
        return Answer(
            question=question,
            status=AnswerStatus.UNSOURCED,
            text=UNSOURCED_MESSAGE,
            sources=sources,
            trace=self._trace(
                manifest.index_id, manifest.embedding_model, generation_model,
                template.version, retrieved, settings.max_attempts, raw_outputs,
            ),
        )

    def _trace(self, index_id, embedding_model, generation_model, prompt_version,
               retrieved, attempts, raw_outputs) -> AnswerTrace:
        return AnswerTrace(
            index_id=index_id,
            embedding_model=embedding_model,
            generation_model=generation_model,
            prompt_version=prompt_version,
            retrieved=retrieved,
            min_score=self._settings.min_score,
            attempts=attempts,
            raw_outputs=tuple(raw_outputs),
        )
