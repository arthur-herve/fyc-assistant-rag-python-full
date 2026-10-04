"""Adaptateurs de fichiers : encodage du corpus et des prompts, instantanés et index (lus strictement,
écrits par json.dumps) ; blancs de la règle de sortie de l'exercice S4.1."""

import json
import tempfile
import unittest
from pathlib import Path

from assistant.application.ports import IndexManifest, Snapshot, SnapshotEntry
from assistant.domain.blanks import WHITESPACE
from assistant.domain.model import Chunk
from assistant.infrastructure.markdown_corpus import CorpusFormatError, MarkdownCorpus, parse_markdown_document
from assistant.infrastructure.prompt_files import FilePromptRepository
from assistant.infrastructure.snapshot_files import JsonSnapshotStore
from assistant.infrastructure.vector_index import IndexUnreadableError, JsonVectorIndex

ROOT = Path(__file__).resolve().parents[2]
BOM = b"\xef\xbb\xbf"
PROMPT = 'version = "v1"\nsystem = "Réponds en français."\nuser = "{question}"\n'
DEEP = "[" * 100_000 + "]" * 100_000   # au-delà de ce que json et tomllib lisent sans RecursionError
SPLITTER = {"type": "paragraph", "max_chars": 800, "overlap_chars": 120, "include_title": True}
# Ce qu'un encodeur JSON peut écrire tel quel ou échapper (espace insécable, emoji : tels quels avec
# ensure_ascii=False) et ce que json.dumps échappe toujours (guillemet, saut de ligne).
SPECIAL = 'Deux jours\xa0: « oui » "x"\nfin \U0001f600'


def nested(levels: int) -> str:
    """Des listes imbriquées sur `levels` niveaux."""
    return "[" * levels + "]" * levels


ENTRY = {"question_id": "q", "user_id": "a", "question": "?", "status": "answered",
         "cited_documents": ["d"], "text": "x", "attempts": 1}


def _snapshot(**entry):
    return {"name": "a", "created_at": "t", "entries": [{**ENTRY, **entry}]}


def _without(key):
    return {"name": "a", "created_at": "t", "entries": [{k: v for k, v in ENTRY.items() if k != key}]}


# (contenu de a.json, ce qui ne va pas) : chaque champ est vérifié et nommé.
MALFORMED_SNAPSHOTS = [
    (["a"], "un objet JSON est attendu"),
    ({"name": "a", "created_at": "t"}, "champ « entries » manquant ou qui n'est pas une liste"),
    ({"name": "a", "created_at": "t", "entries": "x"}, "champ « entries » manquant ou qui n'est pas une liste"),
    ({"name": "a", "created_at": "t", "configuration": None, "entries": []}, "« configuration » doit être un objet"),
    ({"name": "a", "created_at": "t", "configuration": [], "entries": []}, "« configuration » doit être un objet"),
    ({"created_at": "t", "entries": []}, "champ « name » manquant ou non textuel"),
    ({"name": "a", "created_at": 5, "entries": []}, "champ « created_at » manquant ou non textuel"),
    ({"name": "a", "created_at": "t", "entries": ["x"]}, "chaque réponse doit être un objet JSON"),
    (_snapshot(cited_documents="abc"), "« cited_documents » doit être une liste de textes"),
    (_snapshot(cited_documents=[1]), "« cited_documents » doit être une liste de textes"),
    (_without("cited_documents"), "« cited_documents » doit être une liste de textes"),
    (_snapshot(attempts=None), "« attempts » doit être un entier"),
    (_snapshot(attempts=True), "« attempts » doit être un entier"),
    (_snapshot(attempts=1.5), "« attempts » doit être un entier"),
    (_snapshot(attempts="1"), "« attempts » doit être un entier"),
    (_without("question_id"), "champ « question_id » manquant ou non textuel"),
    (_snapshot(user_id=None), "champ « user_id » manquant ou non textuel"),
    (_snapshot(text=5), "champ « text » manquant ou non textuel"),
]

# JSON qui n'est pas strict : clé en double (json.loads garderait la dernière), plus de 900 niveaux, NaN (json.loads le
# lirait), chaîne qui n'est pas du texte (json.loads la garderait), entier de plus de 4300 chiffres (int() le refuserait
# en anglais).
NOT_TEXT = "chaîne qui n'est pas du texte : surrogate UTF-16 isolé (\\ud800 à \\udfff sans sa paire)"
TOO_LONG = "nombre entier de plus de 4300 chiffres"
NOT_STRICT_SNAPSHOTS = [
    ('{"name": "a", "name": "b", "created_at": "t", "entries": []}', "clé « name » en double"),
    ('{"name": "a", "created_at": "t", "configuration": {"x": 1, "x": 2}, "entries": []}', "clé « x » en double"),
    (json.dumps(_snapshot()).replace('"q"', '"q", "question_id": "r"'), "clé « question_id » en double"),
    ('{"name": "a", "created_at": "t", "configuration": {"x": ' + nested(899) + '}, "entries": []}',
     "JSON trop imbriqué : plus de 900 niveaux"),
    ('{"name": "a", "created_at": "t", "configuration": {"x": NaN}, "entries": []}', "« NaN » n'est pas du JSON"),
    (json.dumps(_snapshot()).replace('"x"', '"x\\ud800"'), NOT_TEXT),
    ('{"name": "a", "created_at": "t", "configuration": {"\\udc00": 1}, "entries": []}', NOT_TEXT),
    (json.dumps(_snapshot()).replace('"attempts": 1', '"attempts": 1' + "0" * 4300), TOO_LONG),
]

# Ce qu'écrit JsonSnapshotStore.save (json.dumps, indent=2, accents tels quels).
SNAPSHOT_FILE = (
    '{\n'
    '  "name": "ref",\n'
    '  "created_at": "2026-09-11T12:00:00+00:00",\n'
    '  "configuration": {\n'
    '    "corpus": "solveo",\n'
    '    "embedding_model": "hashing",\n'
    '    "splitter": {\n'
    '      "type": "paragraph",\n'
    '      "max_chars": 800,\n'
    '      "overlap_chars": 120,\n'
    '      "include_title": true\n'
    '    },\n'
    '    "top_k": 4,\n'
    '    "min_score": 0.4,\n'
    '    "temperature": 1.0,\n'
    '    "seed": null,\n'
    '    "index_id": "id"\n'
    '  },\n'
    '  "entries": [\n'
    '    {\n'
    '      "question_id": "q1",\n'
    '      "user_id": "alice",\n'
    '      "question": "Combien ?",\n'
    '      "status": "answered",\n'
    '      "cited_documents": [\n'
    '        "a",\n'
    '        "b"\n'
    '      ],\n'
    '      "text": "Deux jours\U000000a0: « oui » \\"x\\"\\nfin \U0001f600",\n'
    '      "attempts": 1\n'
    '    },\n'
    '    {\n'
    '      "question_id": "q2",\n'
    '      "user_id": "bruno",\n'
    '      "question": "Et ?",\n'
    '      "status": "no_relevant_source",\n'
    '      "cited_documents": [],\n'
    '      "text": "Rien.",\n'
    '      "attempts": 0\n'
    '    }\n'
    '  ]\n'
    '}'
)
INDEX_FILE = (
    '{"manifest": {"index_id": "id", "embedding_model": "m", "dimension": 2, '
    '"corpus_fingerprint": "fp", "splitter": {"type": "paragraph", "max_chars": 800, '
    '"overlap_chars": 120, "include_title": true}, "document_count": 1, "chunk_count": 2, '
    '"created_at": "2026-09-11T00:00:00+00:00"}, "chunks": [{"id": "a#0", "document_id": "a", '
    '"document_title": "Congés", "text": "Deux jours\U000000a0: « oui » \\"x\\"\\nfin \U0001f600", "position": 0, '
    '"allowed_groups": ["rh", "tous"]}, {"id": "a#1", "document_id": "a", '
    '"document_title": "Congés", "text": "suite", "position": 1, "allowed_groups": ["tous"]}], '
    '"vectors": [[0.6, 0.8], [1.2e-05, 1.0]]}'
)


class SnapshotFileTest(unittest.TestCase):
    def test_a_malformed_snapshot_names_the_file_and_the_field(self):
        for payload, problem in MALFORMED_SNAPSHOTS:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "a.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    JsonSnapshotStore(tmp).load("a")
                self.assertEqual(str(caught.exception), f"instantané illisible ({path}) : {problem}")

    def test_optional_fields_take_their_default(self):
        """Sans « configuration » ni « attempts » (instantané d'une version plus ancienne) : accepté."""
        entry = {k: v for k, v in ENTRY.items() if k != "attempts"}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(json.dumps({"name": "a", "created_at": "t", "entries": [entry]}),
                                           encoding="utf-8")
            snapshot = JsonSnapshotStore(tmp).load("a")
        self.assertEqual((snapshot.configuration, snapshot.entries[0].attempts), ({}, 0))

    def test_attempts_take_any_integer(self):
        for attempts in (2**31, -2**31 - 1, 10**100):   # tout entier, sans borne de 32 bits
            with self.subTest(attempts=attempts), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "a.json").write_text(json.dumps(_snapshot(attempts=attempts)), encoding="utf-8")
                self.assertEqual(JsonSnapshotStore(tmp).load("a").entries[0].attempts, attempts)

    def test_a_byte_order_mark_is_accepted_and_latin_1_is_refused(self):
        content = json.dumps({"name": "Congés", "created_at": "t", "entries": []}, ensure_ascii=False)
        position = content.encode("latin-1").index(b"\xe9")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_bytes(BOM + content.encode("utf-8"))
            self.assertEqual(JsonSnapshotStore(tmp).load("a").name, "Congés")
            # Sans puis avec marque d'ordre des octets : la position se compte après elle.
            for prefix in (b"", BOM):
                with self.subTest(prefix=prefix):
                    path.write_bytes(prefix + content.encode("latin-1"))
                    with self.assertRaises(ValueError) as caught:
                        JsonSnapshotStore(tmp).load("a")
                    self.assertEqual(str(caught.exception), f"instantané illisible ({path}) : "
                                                            f"pas en UTF-8 (octet 0xe9 à la position {position})")

    def test_a_deeply_nested_snapshot_is_unreadable_not_a_crash(self):
        """json.loads lève RecursionError (pas une ValueError) : c'était une trace d'erreur."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text('{"name": "a", "created_at": "t", "configuration": {"x": ' + DEEP + '}, "entries": []}',
                            encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                JsonSnapshotStore(tmp).load("a")
            self.assertTrue(str(caught.exception).startswith(f"instantané illisible ({path}) : "))

    def test_a_snapshot_that_is_not_strict_json_is_unreadable(self):
        for content, problem in NOT_STRICT_SNAPSHOTS:
            with self.subTest(content=content[:80]), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "a.json"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    JsonSnapshotStore(tmp).load("a")
                self.assertEqual(str(caught.exception), f"instantané illisible ({path}) : {problem}")

    def test_nine_hundred_levels_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:   # 900 niveaux
            Path(tmp, "a.json").write_text('{"name": "a", "created_at": "t", "configuration": {"x": ' + nested(898)
                                           + '}, "entries": []}', encoding="utf-8")
            self.assertEqual(JsonSnapshotStore(tmp).load("a").name, "a")

    def test_a_snapshot_is_written_as_frozen(self):
        """Configuration dans l'ordre de la composition, réels avec « .0 », espace insécable et emoji tels quels."""
        snapshot = Snapshot("ref", "2026-09-11T12:00:00+00:00",
                            {"corpus": "solveo", "embedding_model": "hashing", "splitter": SPLITTER, "top_k": 4,
                             "min_score": 0.4, "temperature": 1.0, "seed": None, "index_id": "id"},
                            (SnapshotEntry("q1", "alice", "Combien ?", "answered", ("a", "b"), SPECIAL, 1),
                             SnapshotEntry("q2", "bruno", "Et ?", "no_relevant_source", (), "Rien.", 0)))
        with tempfile.TemporaryDirectory() as tmp:
            JsonSnapshotStore(tmp).save(snapshot)
            # Les octets, pas read_text, qui ramènerait à « \n » les « \r\n » qu'écrirait le mode texte sous Windows.
            self.assertEqual(Path(tmp, "ref.json").read_bytes().decode("utf-8"), SNAPSHOT_FILE)


class IndexFileTest(unittest.TestCase):
    """L'index se lit strictement (json_text) et s'écrit par json.dumps."""

    INDEX = ('{"manifest": {"index_id": "id", "embedding_model": "m", "dimension": 2, "corpus_fingerprint": "fp",'
             ' "splitter": {"x": 0}, "document_count": 1, "chunk_count": 1, "created_at": "t"}, "chunks": [{"id": "a#0",'
             ' "document_id": "a", "document_title": "Congés", "text": "x", "position": 0, "allowed_groups": ["tous"]}],'
             ' "vectors": [[0.6, 0.8]]}')

    def read(self, tmp: str, content: bytes):
        path = Path(tmp) / "index.json"
        path.write_bytes(content)
        return JsonVectorIndex(path).manifest()

    def test_a_byte_order_mark_is_accepted_and_latin_1_is_refused(self):
        """La position se compte après la marque d'ordre des octets."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.read(tmp, BOM + self.INDEX.encode("utf-8")).index_id, "id")
            for prefix in (b"", BOM):
                with self.subTest(prefix=prefix), self.assertRaises(IndexUnreadableError) as caught:
                    self.read(tmp, prefix + self.INDEX.encode("latin-1"))
                self.assertEqual(str(caught.exception), f"index illisible ({Path(tmp) / 'index.json'}) : "
                                                        f"pas en UTF-8 (octet 0xe9 à la position {self.INDEX.index('é')})")

    def test_an_index_that_is_not_strict_json_is_unreadable(self):
        for content, problem in (
            (self.INDEX.replace('"index_id": "id"', '"index_id": "id", "index_id": "autre"'), "clé « index_id » en double"),
            (self.INDEX.replace('{"x": 0}', '{"x": ' + nested(898) + "}"), "JSON trop imbriqué : plus de 900 niveaux"),
            (self.INDEX.replace("0.6", "NaN"), "« NaN » n'est pas du JSON"),
            (self.INDEX.replace('"text": "x"', '"text": "x\\ud800"'), NOT_TEXT),
            (self.INDEX.replace('{"x": 0}', '{"\\ud800": 0}'), NOT_TEXT),
            (self.INDEX.replace('"dimension": 2', '"dimension": 2' + "0" * 4300), TOO_LONG),
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(IndexUnreadableError) as caught:
                    self.read(tmp, content.encode("utf-8"))
                self.assertEqual(str(caught.exception), f"index illisible ({Path(tmp) / 'index.json'}) : {problem}")

    def test_a_splitter_that_is_not_an_object_makes_the_index_unreadable(self):
        """status et l'API lisent le découpage comme un dictionnaire : autre chose qu'un objet rend l'index illisible,
        au lieu d'une trace d'erreur."""
        for splitter in ("[1]", '"x"', "3", "null"):
            with self.subTest(splitter=splitter), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(IndexUnreadableError) as caught:
                    self.read(tmp, self.INDEX.replace('{"x": 0}', splitter).encode("utf-8"))
                self.assertEqual(str(caught.exception),
                                 f"index illisible ({Path(tmp) / 'index.json'}) : « splitter » doit être un objet")

    def test_nine_hundred_levels_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:   # 900 niveaux
            content = self.INDEX.replace('{"x": 0}', '{"x": ' + nested(897) + "}")
            self.assertEqual(self.read(tmp, content.encode("utf-8")).index_id, "id")

    def test_an_index_is_written_as_frozen(self):
        """Découpage dans l'ordre où ParagraphSplitter le décrit, réels avec « .0 » ou en notation « 1.2e-05 »."""
        manifest = IndexManifest("id", "m", 2, "fp", SPLITTER, 1, 2, "2026-09-11T00:00:00+00:00")
        chunks = [Chunk("a#0", "a", "Congés", SPECIAL, 0, frozenset({"tous", "rh"})),
                  Chunk("a#1", "a", "Congés", "suite", 1, frozenset({"tous"}))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            JsonVectorIndex(path).replace(manifest, chunks, [[3, 4], [0.000012, 1]])
            self.assertEqual(path.read_bytes().decode("utf-8"), INDEX_FILE)   # une ligne : mêmes octets partout


class PromptFileTest(unittest.TestCase):
    def test_a_byte_order_mark_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer.toml"
            path.write_bytes(PROMPT.encode("utf-8"))
            plain = FilePromptRepository(tmp).get("answer")
            path.write_bytes(BOM + PROMPT.encode("utf-8"))
            self.assertEqual(FilePromptRepository(tmp).get("answer"), plain)

    def test_a_prompt_saved_in_latin_1_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer.toml"
            path.write_bytes(PROMPT.encode("latin-1"))
            with self.assertRaises(ValueError) as caught:
                FilePromptRepository(tmp).get("answer")
            position = PROMPT.encode("latin-1").index(b"\xe9")
            self.assertEqual(str(caught.exception),
                             f"prompt illisible ({path}) : pas en UTF-8 (octet 0xe9 à la position {position})")

    def test_a_missing_field_says_what_to_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "casse.toml"
            path.write_text('version = "v1"\nsystem = "a"\n', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                FilePromptRepository(tmp).get("casse")
            self.assertEqual(str(caught.exception),
                             f"prompt illisible ({path}) : il faut trois textes, version, system et user")

    def test_a_deeply_nested_prompt_is_unreadable_not_a_crash(self):
        """tomllib lève RecursionError (pas une ValueError) : c'était une trace d'erreur."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer.toml"
            path.write_text(PROMPT + "x = " + DEEP + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                FilePromptRepository(tmp).get("answer")
            self.assertTrue(str(caught.exception).startswith(f"prompt illisible ({path}) : "))


class CorpusFileTest(unittest.TestCase):
    def test_a_document_saved_in_latin_1_is_refused_with_its_name(self):
        """Lu en silence, « Congés » deviendrait « Cong�s » dans l'index. La position se compte après
        la marque d'ordre des octets."""
        content = "---\nid: conges\ngroupes: tous\n---\nCongés payés."
        position = content.encode("latin-1").index(b"\xe9")
        for prefix in (b"", BOM):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "conges.md"
                path.write_bytes(prefix + content.encode("latin-1"))
                with self.assertRaises(CorpusFormatError) as caught:
                    MarkdownCorpus(tmp).load()
                self.assertEqual(str(caught.exception), f"{path} : pas en UTF-8 (octet 0xe9 à la position {position})")

    def test_invalid_groups_and_duplicate_ids_are_listed_in_code_point_order(self):
        """Listes écrites [a, b] et rangées par points de code (« B » avant « b »)."""
        with self.assertRaises(CorpusFormatError) as caught:
            parse_markdown_document("---\nid: a\ngroupes: Tous, rh #x\n---\nTexte.")
        self.assertEqual(str(caught.exception), "<texte> : groupe(s) invalide(s) [Tous, rh #x] : des noms en "
                                                "minuscules séparés par des virgules, sans commentaire")
        with tempfile.TemporaryDirectory() as tmp:
            for n, doc_id in enumerate(["b", "B", "b", "B", "c"]):
                Path(tmp, f"{n}.md").write_text(f"---\nid: {doc_id}\ngroupes: tous\n---\nTexte.", encoding="utf-8")
            with self.assertRaises(CorpusFormatError) as caught:
                MarkdownCorpus(tmp).load()
        self.assertEqual(str(caught.exception), "Identifiants de documents en double : [B, b]")


class BlankTest(unittest.TestCase):
    """Les blancs de la règle de sortie de l'exercice S4.1 (output_rules.py), ceux d'Unicode (blanks.py)."""

    def test_the_blanks_are_those_of_isspace_without_x1c_to_x1f(self):
        # Ceux de str.isspace(), moins les séparateurs \x1c à \x1f, qu'Unicode ne compte pas parmi les blancs.
        spaces = {chr(c) for c in range(0x110000) if chr(c).isspace()}
        self.assertEqual(set(WHITESPACE), spaces - set("\x1c\x1d\x1e\x1f"))

    def test_the_blanks_are_defined_once_in_the_domain(self):
        """Dans assistant/domain/blanks.py, qu'importe output_rules.py : une copie à tenir identique à la main
        finirait par s'en écarter."""
        marks = ("[^\\S\\x1c-\\x1f]", "\\u205f")   # la classe d'expression régulière, la chaîne des blancs
        defining = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "assistant").rglob("*.py")
                          if any(mark in path.read_text(encoding="utf-8") for mark in marks))
        self.assertEqual(defining, ["assistant/domain/blanks.py"])


if __name__ == "__main__":
    unittest.main()
