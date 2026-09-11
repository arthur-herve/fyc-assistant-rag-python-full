import tempfile
import unittest
from pathlib import Path

from assistant.application.ports import IndexManifest, Snapshot, SnapshotEntry
from assistant.composition import PROMPTS_DIR
from assistant.domain.model import Chunk
from assistant.infrastructure.markdown_corpus import CorpusFormatError, MarkdownCorpus, parse_markdown_document
from assistant.infrastructure.clock import SystemClock
from assistant.infrastructure.prompt_files import FilePromptRepository
from assistant.infrastructure.snapshot_files import JsonSnapshotStore, SnapshotNotFoundError
from assistant.infrastructure.splitter import ParagraphSplitter
from assistant.infrastructure.vector_index import JsonVectorIndex
from tests.fakes import make_document


class MarkdownCorpusTest(unittest.TestCase):
    def test_parses_header(self):
        doc = parse_markdown_document("---\nid: rh\ntitre: Guide RH\ngroupes: rh, direction\n---\nTexte.")
        self.assertEqual((doc.id, doc.title, doc.text), ("rh", "Guide RH", "Texte."))
        self.assertEqual(doc.allowed_groups, frozenset({"rh", "direction"}))

    def test_documents_are_public_by_default(self):
        doc = parse_markdown_document("---\nid: a\n---\nTexte.")
        self.assertEqual(doc.allowed_groups, frozenset({"tous"}))

    def test_id_is_mandatory(self):
        with self.assertRaises(CorpusFormatError):
            parse_markdown_document("---\ntitre: Sans id\n---\nTexte.")

    def test_solveo_corpus_loads(self):
        documents = MarkdownCorpus(Path(__file__).parents[2] / "corpus" / "solveo").load()
        self.assertGreaterEqual(len(documents), 5)

    def test_service_public_corpus_loads_with_simulated_access_groups(self):
        documents = MarkdownCorpus(Path(__file__).parents[2] / "corpus" / "service-public").load()
        self.assertGreaterEqual(len(documents), 300)
        groups = {g for d in documents for g in d.allowed_groups}
        self.assertEqual(groups, {"tous", "rh", "direction"})


class SplitterTest(unittest.TestCase):
    def test_short_document_gives_one_chunk_with_title(self):
        chunks = ParagraphSplitter(max_chars=200, overlap_chars=20).split(make_document("a", "Court.", title="Titre"))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("Titre\n"))

    def test_chunks_respect_the_size_limit(self):
        text = "\n\n".join(f"Paragraphe numéro {i} " + "mot " * 40 for i in range(10))
        splitter = ParagraphSplitter(max_chars=300, overlap_chars=50, include_title=False)
        chunks = splitter.split(make_document("a", text))
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c.text) <= 300 for c in chunks))

    def test_chunks_inherit_access_rights(self):
        doc = make_document("a", "x " * 500, groups=["rh"])
        chunks = ParagraphSplitter(max_chars=200, overlap_chars=20).split(doc)
        self.assertTrue(all(c.allowed_groups == frozenset({"rh"}) for c in chunks))

    def test_a_giant_paragraph_is_cut(self):
        doc = make_document("a", "mot " * 1000)
        chunks = ParagraphSplitter(max_chars=400, overlap_chars=40, include_title=False).split(doc)
        self.assertTrue(all(len(c.text) <= 400 for c in chunks))


class JsonVectorIndexTest(unittest.TestCase):
    def manifest(self, dimension=2):
        return IndexManifest("id", "m", dimension, "fp", {}, 1, 2, "2026-09-11T00:00:00+00:00")

    def chunks(self):
        return [Chunk("a#0", "a", "A", "texte a", 0, frozenset({"tous"})),
                Chunk("b#0", "b", "B", "texte b", 0, frozenset({"rh"}))]

    def test_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "index.json"
            JsonVectorIndex(path).replace(self.manifest(), self.chunks(), [[1, 0], [0, 1]])
            reloaded = JsonVectorIndex(path)
            self.assertEqual(reloaded.manifest(), self.manifest())
            results = reloaded.search([0.9, 0.1], 2, predicate=lambda c: True)
            self.assertEqual([p.chunk.id for p in results], ["a#0", "b#0"])
            self.assertEqual(results[1].chunk.allowed_groups, frozenset({"rh"}))

    def test_search_applies_the_predicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = JsonVectorIndex(Path(tmp) / "index.json")
            index.replace(self.manifest(), self.chunks(), [[1, 0], [0, 1]])
            results = index.search([0, 1], 5, predicate=lambda c: "rh" not in c.allowed_groups)
            self.assertEqual([p.chunk.id for p in results], ["a#0"])

    def test_rejects_vectors_of_the_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                JsonVectorIndex(Path(tmp) / "i.json").replace(
                    self.manifest(3), self.chunks(), [[1, 0], [0, 1]])

    def test_missing_file_means_no_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(JsonVectorIndex(Path(tmp) / "absent.json").manifest())


class PromptRepositoryTest(unittest.TestCase):
    def test_version_includes_a_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer.toml"
            path.write_text('version = "v1"\nsystem = "a"\nuser = "{question}"\n', encoding="utf-8")
            first = FilePromptRepository(tmp).get("answer").version
            path.write_text('version = "v1"\nsystem = "b"\nuser = "{question}"\n', encoding="utf-8")
            second = FilePromptRepository(tmp).get("answer").version
        self.assertTrue(first.startswith("v1+"))
        self.assertNotEqual(first, second)  # modifié sans changer la version : détecté

    def test_project_prompt_renders(self):
        prompt = FilePromptRepository(PROMPTS_DIR).get("answer")
        rendered = prompt.render(question="Q ?", passages="[1] Titre\ntexte")
        self.assertIn("Q ?", rendered)
        self.assertIn("[1] Titre", rendered)


class JsonSnapshotStoreTest(unittest.TestCase):
    def test_round_trip_and_listing(self):
        snapshot = Snapshot("ref", "2026-09-11T12:00:00+00:00", {"generation_model": "extractive"},
                            (SnapshotEntry("q1", "alice", "Q ?", "answered", ("a", "b"), "Texte [1]", 1),))
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSnapshotStore(Path(tmp) / "instantanes")
            self.assertEqual(store.names(), [])
            store.save(snapshot)
            self.assertEqual(store.load("ref"), snapshot)
            self.assertEqual(store.names(), ["ref"])
            with self.assertRaises(SnapshotNotFoundError):
                store.load("absent")

    def test_rejects_names_that_could_escape_the_directory(self):
        with self.assertRaises(ValueError):
            JsonSnapshotStore("x").load("../autre")


class SystemClockTest(unittest.TestCase):
    def test_is_timezone_aware(self):
        self.assertIsNotNone(SystemClock().now().tzinfo)


if __name__ == "__main__":
    unittest.main()
