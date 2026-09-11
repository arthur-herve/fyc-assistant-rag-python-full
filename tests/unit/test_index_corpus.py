import unittest

from assistant.application.errors import EmptyCorpusError, InconsistentEmbeddingsError
from assistant.application.index_corpus import IndexCorpus, corpus_fingerprint
from assistant.application.ports import EmbeddingBatch
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, WholeDocumentSplitter, make_document,
)


class SwitchingEmbedder(KeywordEmbedder):
    """Simule un service IA dont le modèle change au milieu d'une indexation."""

    def embed_documents(self, texts):
        batch = super().embed_documents(texts)
        model = "modele-a" if len(self.calls) == 1 else "modele-b"
        return EmbeddingBatch(model, batch.dimension, batch.vectors)


class DescribedSplitter(WholeDocumentSplitter):
    def __init__(self, size):
        self.size = size

    def describe(self):
        return {"type": "whole", "size": self.size}


DOCS = [make_document("a", "télétravail"), make_document("b", "frais de repas"),
        make_document("c", "congés")]


class IndexCorpusTest(unittest.TestCase):
    def test_manifest_records_what_is_needed_to_explain_the_index(self):
        index = FakeIndex()
        manifest = IndexCorpus(ListSource(DOCS), WholeDocumentSplitter(), KeywordEmbedder(),
                               index, FixedClock()).execute()
        self.assertEqual(manifest.created_at, "2026-09-11T12:00:00+00:00")
        self.assertEqual(manifest.embedding_model, "fake-keywords")
        self.assertEqual(manifest.chunk_count, 3)
        self.assertEqual(manifest.splitter, {"type": "whole"})
        self.assertIs(index.manifest(), manifest)

    def test_index_id_changes_when_the_splitter_changes(self):
        ids = {
            IndexCorpus(ListSource(DOCS), DescribedSplitter(size), KeywordEmbedder(),
                        FakeIndex(), FixedClock()).execute().index_id
            for size in (300, 800)
        }
        self.assertEqual(len(ids), 2)

    def test_refuses_a_model_change_during_indexing(self):
        use_case = IndexCorpus(ListSource(DOCS), WholeDocumentSplitter(), SwitchingEmbedder(),
                               FakeIndex(), FixedClock(), batch_size=2)
        with self.assertRaises(InconsistentEmbeddingsError):
            use_case.execute()

    def test_empty_corpus(self):
        with self.assertRaises(EmptyCorpusError):
            IndexCorpus(ListSource([]), WholeDocumentSplitter(), KeywordEmbedder(),
                        FakeIndex(), FixedClock()).execute()

    def test_fingerprint_changes_with_access_rights(self):
        public = [make_document("a", "texte")]
        restricted = [make_document("a", "texte", groups=["rh"])]
        self.assertNotEqual(corpus_fingerprint(public), corpus_fingerprint(restricted))


if __name__ == "__main__":
    unittest.main()
