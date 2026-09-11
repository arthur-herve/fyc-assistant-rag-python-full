"""Cas d'usage : figer le comportement du système, puis mesurer ce qui a bougé.

Un système dont un maillon est probabiliste ne se valide pas par assertion
exacte (séquence 3.1). On enregistre ses réponses à un jeu de questions fixe
avec la configuration qui les a produites (instantané), on change UNE chose
(modèle, découpage, prompt, seuil…), on enregistre à nouveau, et on compare :
le taux de dérive n'a de sens que lu à côté des différences de configuration
(séquence 3.2, principe CACE).

Idée reprise des explorations d'AssistantQR, réécrite derrière un port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from assistant.domain.model import User

from .ask_question import AskQuestion
from .errors import ApplicationError
from .ports import Clock, Snapshot, SnapshotEntry, SnapshotStore


@dataclass(frozen=True)
class SnapshotQuestion:
    id: str
    user: User
    question: str


class RecordSnapshot:
    def __init__(self, ask_question: AskQuestion, store: SnapshotStore, clock: Clock,
                 configuration: dict[str, Any]) -> None:
        self._ask = ask_question
        self._store = store
        self._clock = clock
        self._configuration = dict(configuration)

    def execute(self, name: str, questions: Sequence[SnapshotQuestion]) -> Snapshot:
        entries: list[SnapshotEntry] = []
        configuration = dict(self._configuration)
        for q in questions:
            try:
                answer = self._ask.execute(q.user, q.question)
            except ApplicationError as error:
                entries.append(SnapshotEntry(q.id, q.user.id, q.question, "error", (), str(error)))
                continue
            trace = answer.trace
            # Les identifiants concrets ne sont connus qu'après un appel : on les relève.
            configuration.setdefault("index_id", trace.index_id)
            configuration.setdefault("embedding_model_id", trace.embedding_model)
            if trace.generation_model:
                configuration.setdefault("generation_model_id", trace.generation_model)
            if trace.prompt_version:
                configuration.setdefault("prompt_version", trace.prompt_version)
            entries.append(SnapshotEntry(
                question_id=q.id,
                user_id=q.user.id,
                question=q.question,
                status=answer.status.value,
                cited_documents=tuple(sorted({s.document_id for s in answer.sources})),
                text=answer.text,
                attempts=trace.attempts,
            ))
        snapshot = Snapshot(
            name=name,
            created_at=self._clock.now().isoformat(timespec="seconds"),
            configuration=configuration,
            entries=tuple(entries),
        )
        self._store.save(snapshot)
        return snapshot


# Natures d'écart, de la plus bénigne à la plus grave.
IDENTICAL = "identique"
TEXT_CHANGED = "texte modifié"            # mêmes sources, même statut : reformulation
SOURCES_CHANGED = "sources modifiées"     # la réponse ne s'appuie plus sur le même matériau
STATUS_CHANGED = "statut modifié"         # un refus devient une réponse, ou l'inverse
MISSING = "absente d'un des deux"


@dataclass(frozen=True)
class EntryDifference:
    question_id: str
    kind: str
    before: SnapshotEntry | None
    after: SnapshotEntry | None


@dataclass(frozen=True)
class SnapshotComparison:
    baseline: str
    candidate: str
    configuration_differences: tuple[tuple[str, Any, Any], ...]
    differences: tuple[EntryDifference, ...]

    @property
    def compared(self) -> int:
        return sum(1 for d in self.differences if d.kind != MISSING)

    @property
    def changed(self) -> int:
        return sum(1 for d in self.differences if d.kind not in (IDENTICAL, MISSING))

    @property
    def drift_rate(self) -> float | None:
        return None if not self.compared else round(self.changed / self.compared, 3)

    def count(self, kind: str) -> int:
        return sum(1 for d in self.differences if d.kind == kind)


def compare_snapshots(baseline: Snapshot, candidate: Snapshot) -> SnapshotComparison:
    """Fonction pure : aucune dépendance, testable exhaustivement."""
    keys = sorted(set(baseline.configuration) | set(candidate.configuration))
    config_diff = tuple(
        (k, baseline.configuration.get(k), candidate.configuration.get(k))
        for k in keys if baseline.configuration.get(k) != candidate.configuration.get(k)
    )
    before = {e.question_id: e for e in baseline.entries}
    after = {e.question_id: e for e in candidate.entries}
    differences: list[EntryDifference] = []
    for question_id in list(before) + [q for q in after if q not in before]:
        a, b = before.get(question_id), after.get(question_id)
        if a is None or b is None:
            kind = MISSING
        elif a.status != b.status:
            kind = STATUS_CHANGED
        elif a.cited_documents != b.cited_documents:
            kind = SOURCES_CHANGED
        elif a.text.strip() != b.text.strip():
            kind = TEXT_CHANGED
        else:
            kind = IDENTICAL
        differences.append(EntryDifference(question_id, kind, a, b))
    return SnapshotComparison(baseline.name, candidate.name, config_diff, tuple(differences))
