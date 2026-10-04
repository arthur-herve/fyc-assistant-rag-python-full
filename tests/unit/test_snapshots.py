"""Instantanés : figer le comportement, changer une chose, mesurer la dérive (S3.1, S3.2, S4.2)."""

import unittest

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.index_corpus import IndexCorpus
from assistant.application.ports import Snapshot, SnapshotEntry
from assistant.application.errors import IndexNotBuiltError
from assistant.application.snapshots import (
    IDENTICAL, MISSING, SOURCES_CHANGED, STATUS_CHANGED, TEXT_CHANGED,
    InvalidSnapshotNameError, RecordSnapshot, SnapshotQuestion, compare_snapshots,
)
from assistant.domain.model import User
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, MemorySnapshotStore, ScriptedGenerator,
    StaticPrompts, WholeDocumentSplitter, make_document,
)

ALICE = User("alice", frozenset({"tous"}))
DOCS = [make_document("teletravail", "Deux jours de télétravail par semaine."),
        make_document("frais", "Les frais de repas sont plafonnés.")]
QUESTIONS = [
    SnapshotQuestion("tt", ALICE, "Combien de jours de télétravail ?"),
    SnapshotQuestion("frais", ALICE, "Quel plafond pour les frais de repas ?"),
    SnapshotQuestion("hors", ALICE, "Quelle est la capitale de l'Australie ?"),
]


def record(name, generator, store=None, min_score=0.5):
    index = FakeIndex()
    IndexCorpus(ListSource(DOCS), WholeDocumentSplitter(), KeywordEmbedder(), index,
                FixedClock()).execute()
    ask = AskQuestion(KeywordEmbedder(), index, generator, StaticPrompts(),
                      AskSettings(min_score=min_score, max_attempts=1))
    store = store or MemorySnapshotStore()
    use_case = RecordSnapshot(ask, store, FixedClock(), {"generation_model": generator.model})
    return use_case.execute(name, QUESTIONS), store


class RecordSnapshotTest(unittest.TestCase):
    def test_records_answers_refusals_and_configuration(self):
        snapshot, store = record("ref", ScriptedGenerator("Deux jours [1].", model="llm-a"))
        self.assertEqual(snapshot.created_at, "2026-09-11T12:00:00+00:00")
        self.assertEqual([e.status for e in snapshot.entries],
                         ["answered", "answered", "no_relevant_source"])
        self.assertEqual(snapshot.entries[0].cited_documents, ("teletravail",))
        self.assertEqual(snapshot.configuration["generation_model"], "llm-a")
        self.assertEqual(snapshot.configuration["generation_model_id"], "llm-a")
        self.assertEqual(snapshot.configuration["embedding_model_id"], "fake-keywords")
        self.assertIn("index_id", snapshot.configuration)
        self.assertIs(store.load("ref"), snapshot)


    def test_invalid_name_is_refused_before_any_question_is_asked(self):
        # « ref\n » et 65 caractères commencent bien : seul un motif vérifié en entier (fullmatch) les refuse.
        for name in ("../evil", "ref\n", "a" * 65):
            with self.subTest(name=name):
                generator = ScriptedGenerator("Deux jours [1].")
                ask = AskQuestion(KeywordEmbedder(), FakeIndex(), generator, StaticPrompts())
                with self.assertRaises(InvalidSnapshotNameError):
                    RecordSnapshot(ask, MemorySnapshotStore(), FixedClock(), {}).execute(name, QUESTIONS)
                self.assertEqual(generator.requests, [])

    def test_an_invalid_name_is_quoted_between_guillemets(self):
        # Entre « », guillemets à la française, plutôt que sous sa forme Python ('a b') : on voit où le nom
        # commence et finit, espace comprise.
        self.assertEqual(str(InvalidSnapshotNameError("a b")),
                         "nom d'instantané invalide : « a b » (lettres, chiffres, . _ - ; 64 caractères au plus)")

    def test_a_systematic_error_is_raised_not_recorded(self):
        ask = AskQuestion(KeywordEmbedder(), FakeIndex(), ScriptedGenerator("x"), StaticPrompts())
        store = MemorySnapshotStore()
        with self.assertRaises(IndexNotBuiltError):
            RecordSnapshot(ask, store, FixedClock(), {}).execute("ref", QUESTIONS)
        self.assertEqual(store.names(), [])


class CompareSnapshotsTest(unittest.TestCase):
    def test_same_generator_same_index_gives_zero_drift(self):
        a, _ = record("a", ScriptedGenerator("Deux jours [1]."))
        b, _ = record("b", ScriptedGenerator("Deux jours [1]."))
        comparison = compare_snapshots(a, b)
        self.assertEqual(comparison.configuration_differences, ())
        self.assertEqual(comparison.drift_rate, 0.0)
        self.assertEqual(comparison.count(IDENTICAL), 3)

    def test_changing_the_generator_is_visible_in_configuration_and_text(self):
        a, _ = record("a", ScriptedGenerator("Deux jours [1].", model="llm-a"))
        b, _ = record("b", ScriptedGenerator("Vous avez droit à deux jours [1].", model="llm-b"))
        comparison = compare_snapshots(a, b)
        keys = [k for k, _, _ in comparison.configuration_differences]
        self.assertEqual(keys, ["generation_model", "generation_model_id"])
        self.assertEqual(comparison.count(TEXT_CHANGED), 2)   # les deux réponses reformulées
        self.assertEqual(comparison.count(IDENTICAL), 1)      # le refus, inchangé
        self.assertEqual(comparison.drift_rate, round(2 / 3, 3))

    def test_a_refusal_that_becomes_an_answer_is_a_status_change(self):
        a, _ = record("a", ScriptedGenerator("Deux jours [1]."), min_score=0.5)
        b, _ = record("b", ScriptedGenerator("Deux jours [1]."), min_score=0.0)
        comparison = compare_snapshots(a, b)
        self.assertEqual(comparison.count(STATUS_CHANGED), 1)
        change = next(d for d in comparison.differences if d.kind == STATUS_CHANGED)
        self.assertEqual((change.before.status, change.after.status),
                         ("no_relevant_source", "answered"))

    def test_kinds_are_ranked_status_then_sources_then_text(self):
        def entry(status="answered", docs=("a",), text="x"):
            return SnapshotEntry("q", "alice", "?", status, tuple(docs), text)
        base = Snapshot("a", "t", {}, (entry(),))
        self.assertEqual(compare_snapshots(base, Snapshot("b", "t", {}, (entry(status="unsourced"),))).differences[0].kind, STATUS_CHANGED)
        self.assertEqual(compare_snapshots(base, Snapshot("b", "t", {}, (entry(docs=("b",)),))).differences[0].kind, SOURCES_CHANGED)
        self.assertEqual(compare_snapshots(base, Snapshot("b", "t", {}, (entry(text="y"),))).differences[0].kind, TEXT_CHANGED)
        self.assertEqual(compare_snapshots(base, Snapshot("b", "t", {}, (entry(text=" x "),))).differences[0].kind, IDENTICAL)

    def test_missing_questions_are_reported_but_not_counted_as_drift(self):
        a = Snapshot("a", "t", {}, (SnapshotEntry("q1", "alice", "?", "answered", ("a",), "x"),))
        b = Snapshot("b", "t", {}, (SnapshotEntry("q2", "alice", "?", "answered", ("a",), "x"),))
        comparison = compare_snapshots(a, b)
        self.assertEqual(comparison.count(MISSING), 2)
        self.assertEqual(comparison.compared, 0)
        self.assertIsNone(comparison.drift_rate)


if __name__ == "__main__":
    unittest.main()
