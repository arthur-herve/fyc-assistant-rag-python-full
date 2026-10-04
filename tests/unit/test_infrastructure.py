import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from assistant.application.errors import PromptNotFoundError
from assistant.application.ports import IndexManifest, Snapshot, SnapshotEntry
from assistant.composition import PROMPTS_DIR
from assistant.domain.model import Chunk
from assistant.infrastructure.markdown_corpus import CorpusFormatError, MarkdownCorpus, parse_markdown_document
from assistant.infrastructure.clock import SystemClock
from assistant.infrastructure.prompt_files import FilePromptRepository
from assistant.infrastructure.snapshot_files import JsonSnapshotStore, SnapshotNotFoundError
from assistant.infrastructure.splitter import ParagraphSplitter
from assistant.infrastructure.vector_index import IndexUnreadableError, JsonVectorIndex
from tests.fakes import make_document


class MarkdownCorpusTest(unittest.TestCase):
    def test_parses_header(self):
        doc = parse_markdown_document("---\nid: rh\ntitre: Guide RH\ngroupes: rh, direction\n---\nTexte.")
        self.assertEqual((doc.id, doc.title, doc.text), ("rh", "Guide RH", "Texte."))
        self.assertEqual(doc.allowed_groups, frozenset({"rh", "direction"}))

    def test_access_groups_are_mandatory(self):
        """Un droit oublié ou mal écrit ne rend jamais un document public en silence."""
        for header in ("id: a", "id: a\ngroupes:", "id: a\ngroupe: rh"):
            with self.subTest(header=header), self.assertRaises(CorpusFormatError):
                parse_markdown_document(f"---\n{header}\n---\nTexte.")

    def test_a_comment_in_the_groups_is_rejected(self):
        """L'exemple commenté d'une ancienne documentation donnait des droits faux."""
        with self.assertRaises(CorpusFormatError):
            parse_markdown_document("---\nid: a\ngroupes: rh  # ou : rh, direction\n---\nTexte.")

    def test_a_file_with_a_byte_order_mark_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.md").write_text("---\nid: a\ngroupes: tous\n---\nTexte.", encoding="utf-8-sig")
            self.assertEqual(MarkdownCorpus(tmp).load()[0].id, "a")

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

    def test_forbidden_passages_take_no_place_in_the_top_k(self):
        """Pré-filtrage (ADR 0006) : un passage interdit mieux classé ne masque pas le passage autorisé."""
        with tempfile.TemporaryDirectory() as tmp:
            index = JsonVectorIndex(Path(tmp) / "index.json")
            index.replace(self.manifest(), self.chunks(), [[1, 0], [0, 1]])
            results = index.search([0, 1], 1, predicate=lambda c: "rh" not in c.allowed_groups)
            self.assertEqual([p.chunk.id for p in results], ["a#0"])

    def test_rejects_vectors_of_the_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                JsonVectorIndex(Path(tmp) / "i.json").replace(
                    self.manifest(3), self.chunks(), [[1, 0], [0, 1]])

    def test_missing_file_means_no_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(JsonVectorIndex(Path(tmp) / "absent.json").manifest())

    def test_an_unreadable_file_is_not_an_absent_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            path.write_text("{ pas du json", encoding="utf-8")
            index = JsonVectorIndex(path)
            for _ in range(2):   # la deuxième lecture ne dit pas « aucun index » non plus
                with self.assertRaises(IndexUnreadableError):
                    index.manifest()

    def test_a_file_that_is_not_utf8_is_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            path.write_bytes(b'{"manifest": "\xff\xfe"}')
            with self.assertRaises(IndexUnreadableError) as caught:
                JsonVectorIndex(path).manifest()
            self.assertIn(f"index illisible ({path})", str(caught.exception))

    def test_an_index_rebuilt_by_another_process_is_seen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            JsonVectorIndex(path).replace(self.manifest(), self.chunks(), [[1, 0], [0, 1]])
            server = JsonVectorIndex(path)                       # le processus `serve`
            self.assertEqual(server.manifest().index_id, "id")
            rebuilt = IndexManifest("id-2", "m", 2, "fp", {}, 1, 1, "2026-09-21T00:00:00+00:00")
            JsonVectorIndex(path).replace(rebuilt, self.chunks()[:1], [[1, 0]])   # `index` en ligne de commande
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))   # une seconde plus tard
            self.assertEqual(server.manifest().index_id, "id-2")
            self.assertEqual(len(server.search([1, 0], 5, predicate=lambda c: True)), 1)

    def test_an_index_rebuilt_with_the_same_size_and_date_is_seen(self):
        """Même taille, même date : seule la reconstruction (nouveau fichier) les distingue."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            first = IndexManifest("id", "m", 2, "fp", {}, 1, 2, "t1")
            JsonVectorIndex(path).replace(first, self.chunks(), [[1, 0], [0, 1]])
            server = JsonVectorIndex(path)
            self.assertEqual(server.manifest().created_at, "t1")
            stat = path.stat()
            JsonVectorIndex(path).replace(IndexManifest("id", "m", 2, "fp", {}, 1, 2, "t2"), self.chunks(), [[1, 0], [0, 1]])
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(path.stat().st_size, stat.st_size)
            self.assertEqual(server.manifest().created_at, "t2")

    def test_rejects_a_query_of_the_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = JsonVectorIndex(Path(tmp) / "index.json")
            index.replace(self.manifest(), self.chunks(), [[1, 0], [0, 1]])
            with self.assertRaises(ValueError):
                index.search([1, 0, 0], 2, predicate=lambda c: True)


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

    def test_an_incomplete_prompt_names_the_file(self):
        """Édité à la main : un champ oublié donne une erreur qui dit quoi corriger, pas une KeyError."""
        for content in ('system = "a"\nuser = "{question}"\n',               # sans version
                        'version = 2\nsystem = "a"\nuser = "{question}"\n',  # version qui n'est pas un texte
                        'version = "v1\n'):                                    # TOML invalide
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "casse.toml").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "casse.toml"):
                    FilePromptRepository(tmp).get("casse")

    def test_project_prompt_renders(self):
        prompt = FilePromptRepository(PROMPTS_DIR).get("answer")
        rendered = prompt.render(question="Q ?", passages="[1] Titre\ntexte")
        self.assertIn("Q ?", rendered)
        self.assertIn("[1] Titre", rendered)

    def test_the_line_endings_of_the_file_change_nothing(self):
        """LF, CRLF (Windows, ou git qui convertit à l'extraction) ou CR seul : read_utf8, en mode texte, les ramène
        à LF. Le prompt livré garde sa version, empreinte comprise : celle de la version C#."""
        text = (PROMPTS_DIR / "answer.toml").read_text(encoding="utf-8")
        for newline in ("\n", "\r\n", "\r"):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "answer.toml").write_bytes(text.replace("\n", newline).encode("utf-8"))
                self.assertEqual(FilePromptRepository(tmp).get("answer").version, "v1+085b70e7")

    def test_an_unknown_prompt_names_its_folder_and_the_known_ones(self):
        """Un nom mal tapé, ou un dossier absent : le nom, le dossier et les prompts connus (« aucun »), pas
        « [Errno 2] No such file or directory »."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("b", "a-v2", "a"):
                Path(tmp, f"{name}.toml").write_text('version = "v1"\nsystem = "s"\nuser = "{question}"\n',
                                                     encoding="utf-8")
            Path(tmp, "notes.txt").write_text("pas un prompt", encoding="utf-8")
            Path(tmp, "dossier.toml").mkdir()   # un dossier non plus : des fichiers seulement
            absent = Path(tmp) / "absent"
            for directory, known in ((tmp, "a, a-v2, b"), (absent, "aucun")):
                with self.subTest(directory=directory), self.assertRaises(PromptNotFoundError) as caught:
                    FilePromptRepository(directory).get("answr")
                self.assertEqual(str(caught.exception),
                                 f"prompt introuvable : answr dans {directory} (connus : {known})")


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

    def test_an_unknown_snapshot_names_the_known_ones_without_brackets(self):
        # Un message qui se lit : ni « ['a', 'b'] », ni « [] » pour un dossier vide.
        snapshot = Snapshot("ref", "2026-09-11T12:00:00+00:00", {}, ())
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSnapshotStore(Path(tmp) / "instantanes")
            for saved, known in (((), "aucun"), (("ref", "b"), "b, ref")):
                for name in saved:
                    store.save(replace(snapshot, name=name))
                with self.subTest(known=known), self.assertRaises(SnapshotNotFoundError) as caught:
                    store.load("absent")
                self.assertEqual(str(caught.exception), f"instantané introuvable : absent (connus : {known})")

    def test_a_folder_named_like_a_snapshot_is_not_one(self):
        """Seuls les fichiers comptent : un dossier « dossier.json » n'est pas listé, et il est introuvable (sa
        lecture finissait en « fichier ou dossier inaccessible »)."""
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSnapshotStore(tmp)
            store.save(Snapshot("ref", "2026-09-11T12:00:00+00:00", {}, ()))
            (Path(tmp) / "dossier.json").mkdir()
            self.assertEqual(store.names(), ["ref"])
            with self.assertRaises(SnapshotNotFoundError) as caught:
                store.load("dossier")
            self.assertEqual(str(caught.exception), "instantané introuvable : dossier (connus : ref)")

    def test_rejects_names_that_could_escape_the_directory(self):
        for name in ("../autre", "ref\n"):   # « $ » laisserait passer un saut de ligne final
            with self.subTest(name=name), self.assertRaises(ValueError):
                JsonSnapshotStore("x").load(name)

    def test_an_incomplete_snapshot_names_the_file(self):
        entry = {"question_id": "q", "user_id": "a", "question": "?", "status": "answered",
                 "cited_documents": ["d"], "text": "x", "attempts": 1}
        for payload in ({"name": "a", "created_at": "t"},                           # sans entries
                        {"name": "a", "created_at": "t", "entries": [{"status": "answered"}]},
                        {"name": "a", "created_at": "t", "entries": [{**entry, "text": None}]},
                        {"name": "a", "created_at": "t", "entries": [{**entry, "cited_documents": "abc"}]},
                        {"name": "a", "created_at": "t", "configuration": [], "entries": [entry]},
                        "pas du json"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                text = payload if isinstance(payload, str) else json.dumps(payload)
                (Path(tmp) / "a.json").write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "a.json"):
                    JsonSnapshotStore(tmp).load("a")

    def test_ignores_unknown_fields_written_by_a_newer_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "futur.json"
            path.write_text(json.dumps({"name": "futur", "created_at": "t", "configuration": {},
                                        "entries": [{"question_id": "q", "user_id": "a", "question": "?",
                                                     "status": "answered", "cited_documents": ["d"],
                                                     "text": "x", "attempts": 1, "nouveau_champ": 42}]}),
                            encoding="utf-8")
            self.assertEqual(JsonSnapshotStore(tmp).load("futur").entries[0].cited_documents, ("d",))


class SystemClockTest(unittest.TestCase):
    def test_is_timezone_aware(self):
        self.assertIsNotNone(SystemClock().now().tzinfo)


class PromptRenderingTest(unittest.TestCase):
    def test_only_the_two_variables_are_replaced_in_one_pass(self):
        """Un exemple JSON dans le prompt ne plante pas ; un passage qui contient « {question} »
        n'est pas substitué une seconde fois."""
        from assistant.application.ports import PromptTemplate
        template = PromptTemplate("n", "v", "s", 'Réponds en JSON {"a": 1}\n{passages}\nQuestion : {question}')
        rendered = template.render(question="Q ?", passages="[1] parle de {question}")
        self.assertEqual(rendered, 'Réponds en JSON {"a": 1}\n[1] parle de {question}\nQuestion : Q ?')


if __name__ == "__main__":
    unittest.main()
