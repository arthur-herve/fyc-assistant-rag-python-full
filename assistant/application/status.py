"""Cas d'usage : l'index est-il encore valable ? (séquence 4.2)

Un index dépend de trois choses qui vivent ailleurs que dans le code : le
corpus, le découpage et le modèle d'embeddings. Si l'une a changé depuis
l'indexation, les réponses ne sont plus explicables. Ce cas d'usage compare
l'état courant au manifeste de l'index et nomme ce qui a bougé.

La réindexation est au RAG ce que le réentraînement est à un modèle appris.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import AIServiceError
from .index_corpus import corpus_fingerprint
from .ports import (
    DocumentSource, Embedder, IndexManifest, PromptRepository, TextSplitter, VectorIndex,
)
from .snapshots import config_value_to_text


def describe_splitter(splitter: dict[str, Any]) -> str:
    """Le découpage, clés triées : {include_title: true, max_chars: 800, …}. Le même texte dans le
    message « découpage modifié » et dans l'affichage de status."""
    return "{" + ", ".join(f"{k}: {config_value_to_text(v)}" for k, v in sorted(splitter.items())) + "}"


@dataclass(frozen=True)
class StatusReport:
    index: IndexManifest | None
    corpus_documents: int
    corpus_fingerprint: str
    splitter: dict[str, Any]
    embedding_model: str | None
    embedding_dimension: int | None
    ai_service_error: str | None
    prompt_version: str
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def up_to_date(self) -> bool:
        return not self.issues

    @property
    def unverified(self) -> bool:
        """Rien à refaire de connu, mais le modèle servi n'a pas pu être vérifié."""
        return self.ai_service_error is not None and len(self.issues) == 1


class CheckStatus:
    def __init__(
        self,
        source: DocumentSource,
        splitter: TextSplitter,
        embedder: Embedder,
        index: VectorIndex,
        prompts: PromptRepository,
        prompt_name: str = "answer",
    ) -> None:
        self._source = source
        self._splitter = splitter
        self._embedder = embedder
        self._index = index
        self._prompts = prompts
        self._prompt_name = prompt_name

    def execute(self) -> StatusReport:
        manifest = self._index.manifest()
        documents = self._source.load()
        fingerprint = corpus_fingerprint(documents)
        splitter = self._splitter.describe()
        prompt_version = self._prompts.get(self._prompt_name).version

        model: str | None = None
        dimension: int | None = None
        service_error: str | None = None
        try:
            probe = self._embedder.embed_query("vérification du modèle servi")
            model, dimension = probe.model, probe.dimension
        except AIServiceError as error:
            service_error = str(error)

        issues: list[str] = []
        if manifest is None:
            issues.append("aucun index : lancer l'indexation")
        else:
            if fingerprint != manifest.corpus_fingerprint:
                issues.append(
                    f"corpus modifié depuis l'indexation ({manifest.document_count} documents "
                    f"indexés, {len(documents)} aujourd'hui) : réindexer"
                )
            if splitter != manifest.splitter:
                issues.append(
                    f"découpage modifié ({describe_splitter(manifest.splitter)} → {describe_splitter(splitter)}) "
                    f": réindexer"
                )
            if model is not None and (model, dimension) != (manifest.embedding_model, manifest.dimension):
                issues.append(
                    f"le service IA sert « {model} » ({dimension} dim.) mais l'index a été construit "
                    f"avec « {manifest.embedding_model} » ({manifest.dimension} dim.) : réindexer"
                )
        if service_error is not None:
            issues.append(f"modèle servi non vérifié, le service IA a échoué : {service_error}")

        return StatusReport(
            index=manifest,
            corpus_documents=len(documents),
            corpus_fingerprint=fingerprint,
            splitter=splitter,
            embedding_model=model,
            embedding_dimension=dimension,
            ai_service_error=service_error,
            prompt_version=prompt_version,
            issues=tuple(issues),
        )
