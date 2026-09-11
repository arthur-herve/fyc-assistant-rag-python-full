"""`status` : l'index est-il encore cohérent avec le corpus, le découpage et le modèle ? (S4.2)"""

import unittest

from assistant.application.errors import AIServiceError
from assistant.application.index_corpus import IndexCorpus
from assistant.application.status import CheckStatus
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, StaticPrompts, WholeDocumentSplitter,
    make_document,
)

DOCS = [make_document("a", "télétravail deux jours"), make_document("b", "frais de repas")]


class OtherSplitter(WholeDocumentSplitter):
    def describe(self):
        return {"type": "whole", "max_chars": 300}


class BrokenEmbedder(KeywordEmbedder):
    def embed_query(self, text):
        raise AIServiceError("Service IA injoignable")


def build_index(documents=DOCS, embedder=None):
    index = FakeIndex()
    IndexCorpus(ListSource(documents), WholeDocumentSplitter(), embedder or KeywordEmbedder(),
                index, FixedClock()).execute()
    return index


def status(index, documents=DOCS, splitter=None, embedder=None):
    return CheckStatus(ListSource(documents), splitter or WholeDocumentSplitter(),
                       embedder or KeywordEmbedder(), index, StaticPrompts()).execute()


class CheckStatusTest(unittest.TestCase):
    def test_up_to_date_when_nothing_changed(self):
        report = status(build_index())
        self.assertTrue(report.up_to_date)
        self.assertEqual(report.corpus_documents, 2)
        self.assertEqual(report.embedding_model, "fake-keywords")
        self.assertEqual(report.prompt_version, "test-v1")

    def test_no_index_yet(self):
        report = status(FakeIndex())
        self.assertFalse(report.up_to_date)
        self.assertIn("aucun index", report.issues[0])

    def test_detects_a_modified_corpus(self):
        report = status(build_index(), documents=DOCS + [make_document("c", "congés")])
        self.assertEqual(len(report.issues), 1)
        self.assertIn("corpus modifié", report.issues[0])
        self.assertIn("réindexer", report.issues[0])

    def test_detects_a_changed_access_right_as_a_corpus_change(self):
        restricted = [make_document("a", "télétravail deux jours", groups=("rh",)), DOCS[1]]
        report = status(build_index(), documents=restricted)
        self.assertIn("corpus modifié", report.issues[0])

    def test_detects_a_changed_splitter(self):
        report = status(build_index(), splitter=OtherSplitter())
        self.assertIn("découpage modifié", report.issues[0])

    def test_detects_that_the_ai_service_now_serves_another_model(self):
        report = status(build_index(), embedder=KeywordEmbedder(model="fake-keywords-v2"))
        self.assertIn("fake-keywords-v2", report.issues[0])
        self.assertIn("réindexer", report.issues[0])

    def test_unreachable_ai_service_is_reported_not_fatal(self):
        report = status(build_index(), embedder=BrokenEmbedder())
        self.assertIsNone(report.embedding_model)
        self.assertIn("injoignable", report.issues[0])
        self.assertIsNotNone(report.index)


if __name__ == "__main__":
    unittest.main()
