"""Bout en bout, en HTTP réel : application ↔ service IA, en mode hors-ligne."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from ai_service.registry import ModelRegistry
from ai_service.server import create_server as create_ai_server
from assistant.application.errors import IndexModelMismatchError
from assistant.composition import AppConfig, build
from assistant.domain.model import AnswerStatus
from assistant.interface.http_api import create_server as create_app_server

ROOT = Path(__file__).resolve().parents[2]
AI_CONFIG = {
    "embedding": {"hashing": {"backend": "hashing", "dimension": 256},
                  "hashing-512": {"backend": "hashing", "dimension": 512}},
    "generation": {"extractive": {"backend": "extractive"},
                   "extractive-bruite": {"backend": "extractive", "citation_failure_rate": 0.3,
                                         "randomize": True, "model_id": "extractive-bruite"}},
}


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ai_server = create_ai_server(ModelRegistry.from_dict(AI_CONFIG), "127.0.0.1", 0, quiet=True)
        ai_url = serve(cls.ai_server)
        cls.tmp = tempfile.TemporaryDirectory()
        config_path = Path(cls.tmp.name) / "app.toml"
        config_path.write_text(f"""
[ai_service]
base_url = "{ai_url}"
embedding_model = "hashing"
generation_model = "extractive"
timeout_seconds = 10
[corpus]
directory = "{(ROOT / 'corpus' / 'solveo').as_posix()}"
[index]
path = "{(Path(cls.tmp.name) / 'index.json').as_posix()}"
[snapshots]
directory = "{(Path(cls.tmp.name) / 'instantanes').as_posix()}"
[splitter]
max_chars = 800
overlap_chars = 120
[retrieval]
top_k = 4
[retrieval.min_score]
default = 0.15
[users.alice]
groups = ["tous"]
[users.bruno]
groups = ["tous", "rh"]
""", encoding="utf-8")
        cls.config = AppConfig.load(config_path)
        cls.container = build(cls.config)
        cls.manifest = cls.container.index_corpus.execute()

    @classmethod
    def tearDownClass(cls):
        cls.ai_server.shutdown()
        cls.ai_server.server_close()
        cls.tmp.cleanup()

    def ask(self, user, question, **overrides):
        container = build(self.config, **overrides) if overrides else self.container
        return container.ask_question.execute(self.config.user(user), question)

    def test_index_manifest_comes_from_the_ai_service(self):
        self.assertEqual(self.manifest.embedding_model, "hashing-256-stem6")
        self.assertEqual(self.manifest.document_count, 9)

    def test_answer_is_sourced(self):
        answer = self.ask("alice", "Combien de jours de télétravail par semaine ?")
        self.assertEqual(answer.status, AnswerStatus.ANSWERED)
        self.assertIn("deux jours", answer.text)
        self.assertEqual({s.document_id for s in answer.sources}, {"teletravail"})

    def test_confidential_document_is_never_cited_for_an_unauthorized_user(self):
        answer = self.ask("alice", "Quelle est la fourchette de salaire d'un consultant senior ?")
        self.assertNotIn("grille-salaires", {s.document_id for s in answer.sources})
        self.assertNotIn("56 000", answer.text)

    def test_authorized_user_reads_the_confidential_document(self):
        answer = self.ask("bruno", "Quelle est la fourchette de salaire d'un consultant senior ?")
        self.assertIn("grille-salaires", {s.document_id for s in answer.sources})

    def test_out_of_scope_question(self):
        answer = self.ask("alice", "Quelle est la capitale de l'Australie ?")
        self.assertEqual(answer.status, AnswerStatus.NO_RELEVANT_SOURCE)

    def test_verdict_1_changing_the_generation_model_works_with_the_same_index(self):
        answer = self.ask("alice", "Combien de jours de congés payés par an ?",
                          generation_model="extractive-bruite")
        self.assertEqual(answer.trace.index_id, self.manifest.index_id)
        self.assertIn(answer.trace.generation_model, ("extractive-bruite", None))

    def test_verdict_2_changing_the_embedding_model_requires_reindexing(self):
        with self.assertRaises(IndexModelMismatchError):
            self.ask("alice", "Combien de jours de congés ?", embedding_model="hashing-512")

    def test_status_is_up_to_date_after_indexing(self):
        report = self.container.check_status.execute()
        self.assertTrue(report.up_to_date, report.issues)
        self.assertEqual(report.embedding_model, "hashing-256-stem6")

    def test_status_detects_another_served_model(self):
        report = build(self.config, embedding_model="hashing-512").check_status.execute()
        self.assertFalse(report.up_to_date)
        self.assertIn("hashing-512", report.issues[0])

    def test_snapshots_record_and_compare_through_the_real_stack(self):
        from assistant.application.snapshots import SnapshotQuestion, compare_snapshots
        questions = [SnapshotQuestion("tt", self.config.user("alice"), "Combien de jours de télétravail ?"),
                     SnapshotQuestion("hors", self.config.user("alice"), "Quelle est la capitale de l'Australie ?")]
        self.container.record_snapshot.execute("ref", questions)
        build(self.config, generation_model="extractive-bruite").record_snapshot.execute("bruit", questions)
        store = self.container.snapshots
        self.assertEqual(store.names(), ["bruit", "ref"])
        comparison = compare_snapshots(store.load("ref"), store.load("bruit"))
        self.assertIn(("generation_model", "extractive", "extractive-bruite"),
                      comparison.configuration_differences)
        self.assertEqual(comparison.compared, 2)

    def test_http_api_of_the_application(self):
        app_server = create_app_server(self.container, "127.0.0.1", 0, quiet=True)
        url = serve(app_server)
        try:
            request = urllib.request.Request(
                url + "/v1/ask",
                data=json.dumps({"user": "alice", "question": "Quel est le plafond d'un repas d'affaires le midi ?"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read())
            self.assertEqual(body["status"], "answered")
            self.assertIn("25 euros", body["text"])
            self.assertEqual(body["trace"]["embedding_model"], "hashing-256-stem6")

            request = urllib.request.Request(
                url + "/v1/ask", data=json.dumps({"user": "mallory", "question": "x"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 403)
        finally:
            app_server.shutdown()
            app_server.server_close()


if __name__ == "__main__":
    unittest.main()
