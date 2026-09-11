"""Tests déterministes du cas d'usage : aucun modèle, aucun réseau."""

import unittest

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.errors import IndexModelMismatchError, IndexNotBuiltError
from assistant.application.index_corpus import IndexCorpus
from assistant.domain.errors import EmptyQuestionError
from assistant.domain.model import AnswerStatus, User
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, ScriptedGenerator, StaticPrompts,
    WholeDocumentSplitter, count_markers, make_document,
)

ALICE = User("alice", frozenset({"tous"}))
BRUNO = User("bruno", frozenset({"tous", "rh"}))


def indexed(embedder=None):
    index = FakeIndex()
    documents = [
        make_document("teletravail", "Deux jours de télétravail par semaine."),
        make_document("frais", "Le repas est plafonné à 25 euros, frais remboursés."),
        make_document("grille", "Salaire senior : 56 000 à 68 000 euros.", groups=["rh"]),
    ]
    IndexCorpus(ListSource(documents), WholeDocumentSplitter(), embedder or KeywordEmbedder(),
                index, clock=FixedClock()).execute()
    return index


def use_case(generator, index=None, embedder=None, **settings):
    return AskQuestion(embedder or KeywordEmbedder(), index or indexed(), generator,
                       StaticPrompts(), AskSettings(**{"min_score": 0.5, **settings}))


class AskQuestionTest(unittest.TestCase):
    def test_answers_with_cited_sources(self):
        generator = ScriptedGenerator("Deux jours par semaine [1].")
        answer = use_case(generator).execute(ALICE, "Combien de jours de télétravail ?")

        self.assertEqual(answer.status, AnswerStatus.ANSWERED)
        self.assertEqual([s.document_id for s in answer.sources], ["teletravail"])
        self.assertEqual(answer.trace.generation_model, "fake-llm")
        self.assertEqual(answer.trace.prompt_version, "test-v1")

    def test_does_not_call_the_model_without_relevant_passage(self):
        generator = ScriptedGenerator("ne devrait pas être appelé [1]")
        answer = use_case(generator).execute(ALICE, "Quelle est la capitale de l'Australie ?")

        self.assertEqual(answer.status, AnswerStatus.NO_RELEVANT_SOURCE)
        self.assertEqual(generator.requests, [])

    def test_restricted_passages_never_reach_the_prompt(self):
        generator = ScriptedGenerator("[1]")
        use_case(generator, min_score=0.0).execute(ALICE, "Quel salaire pour un senior ?")
        for request in generator.requests:
            self.assertNotIn("56 000", request.prompt)

    def test_authorized_user_gets_restricted_passages(self):
        generator = ScriptedGenerator("Entre 56 000 et 68 000 euros [1].")
        answer = use_case(generator).execute(BRUNO, "Quel salaire pour un senior ?")
        self.assertEqual([s.document_id for s in answer.sources], ["grille"])

    def test_retries_when_the_model_forgets_to_cite(self):
        generator = ScriptedGenerator("Deux jours.", "Deux jours [1].")
        answer = use_case(generator).execute(ALICE, "Combien de jours de télétravail ?")

        self.assertEqual(answer.status, AnswerStatus.ANSWERED)
        self.assertEqual(answer.trace.attempts, 2)
        self.assertEqual(len(answer.trace.raw_outputs), 2)

    def test_refuses_an_unsourced_answer_after_all_attempts(self):
        generator = ScriptedGenerator("Deux jours.", "Deux jours [9].")
        answer = use_case(generator).execute(ALICE, "Combien de jours de télétravail ?")

        self.assertEqual(answer.status, AnswerStatus.UNSOURCED)
        self.assertNotIn("Deux jours", answer.text)  # la sortie non sourcée n'est pas montrée
        self.assertTrue(answer.sources)             # mais les passages le sont

    def test_passages_are_numbered_in_the_prompt(self):
        generator = ScriptedGenerator("[1]")
        use_case(generator, min_score=0.0).execute(ALICE, "télétravail et frais de repas")
        self.assertEqual(count_markers(generator.requests[0].prompt), 2)

    def test_changing_the_embedding_model_without_reindexing_is_refused(self):
        index = indexed(KeywordEmbedder(model="modele-a"))
        ask = use_case(ScriptedGenerator("[1]"), index=index,
                       embedder=KeywordEmbedder(model="modele-b"))
        with self.assertRaises(IndexModelMismatchError):
            ask.execute(ALICE, "télétravail")

    def test_changing_the_generation_model_needs_no_reindexing(self):
        index = indexed()
        for model in ("llm-a", "llm-b"):
            answer = use_case(ScriptedGenerator("Deux jours [1].", model=model),
                              index=index).execute(ALICE, "jours de télétravail")
            self.assertEqual(answer.trace.generation_model, model)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(EmptyQuestionError):
            use_case(ScriptedGenerator("[1]")).execute(ALICE, "   ")

    def test_asking_before_indexing_is_explicit(self):
        ask = use_case(ScriptedGenerator("[1]"), index=FakeIndex())
        with self.assertRaises(IndexNotBuiltError):
            ask.execute(ALICE, "télétravail")

    def test_seed_changes_between_attempts(self):
        generator = ScriptedGenerator("sans citation", "avec [1]")
        use_case(generator, seed=42).execute(ALICE, "jours de télétravail")
        self.assertEqual([r.seed for r in generator.requests], [42, 43])


if __name__ == "__main__":
    unittest.main()
