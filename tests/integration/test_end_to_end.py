"""Bout en bout, en HTTP réel : application ↔ service IA, en mode hors-ligne."""

import csv
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from ai_service.registry import ModelRegistry
from ai_service.server import create_server as create_ai_server
from assistant.application.errors import IndexModelMismatchError
from assistant.composition import AppConfig, build
from assistant.domain.model import AnswerStatus
from assistant.interface.benchmark import load_questions, run_benchmark
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
        self.assertEqual(answer.trace.generation_model, "extractive-bruite")   # le générateur a bien servi

    def test_verdict_2_changing_the_embedding_model_requires_reindexing(self):
        with self.assertRaises(IndexModelMismatchError):
            self.ask("alice", "Combien de jours de congés ?", embedding_model="hashing-512")

    def test_the_benchmark_measures_through_the_real_stack(self):
        questions = [q for q in load_questions(ROOT / "eval" / "questions.json")
                     if q.user in ("alice", "bruno")][:6]
        out = Path(self.tmp.name) / "banc"
        run_benchmark(self.config, ["hashing"], ["extractive"], questions, runs=1, out_dir=out,
                      log=lambda *_: None)
        with open(out / "resultats.csv", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["question_id"] for r in rows], [q.id for q in questions])
        self.assertEqual({r["generation_model_id"] for r in rows if r["status"] == "answered"}, {"extractive"})
        self.assertTrue((out / "rapport.md").is_file())
        # Une option fausse est refusée avant tout travail : pas de dossier de résultats à moitié rempli.
        with self.assertRaises(ValueError):
            run_benchmark(self.config, ["hashing"], ["extractive"], questions, runs=1,
                          out_dir=Path(self.tmp.name) / "jamais", min_score_mode="abc", log=lambda *_: None)
        self.assertFalse((Path(self.tmp.name) / "jamais").exists())

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
        # Les réglages qui changent le statut des réponses font partie de l'empreinte (S4.2).
        configuration = store.load("ref").configuration
        self.assertEqual((configuration["max_attempts"], configuration["validate_output"],
                          configuration["max_output_chars"]), (2, True, 1500))

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


class SwappableRegistry:
    """Registre du service IA remplaçable à chaud : imite l'équipe qui change le
    modèle derrière un alias pendant que l'application tourne."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.current = registry

    def embedding(self, alias):
        return self.current.embedding(alias)

    def generation(self, alias):
        return self.current.generation(alias)

    def describe(self):
        return self.current.describe()


def hashing_registry(stem: int = 6) -> ModelRegistry:
    return ModelRegistry.from_dict({
        "embedding": {"hashing": {"backend": "hashing", "dimension": 256, "stem": stem}},
        "generation": {"extractive": {"backend": "extractive"}},
    })


class WithoutFingerprint:
    """Un moteur qui ne fournit pas d'empreinte des poids (serveur compatible OpenAI…) :
    le même nom de modèle, quels que soient les poids servis derrière."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def embedding(self, alias):
        model = self._registry.embedding(alias)
        return SimpleNamespace(embed=lambda texts, input_type: replace(
            model.embed(texts, input_type), model_id="sans-empreinte"))

    def __getattr__(self, name):
        return getattr(self._registry, name)


class ModelChangedBehindTheAliasTest(unittest.TestCase):
    """Processus long (`serve`), configuration livrée (cache d'embeddings compris) :
    ce qui est servi derrière l'alias change, puis on réindexe (ADR 0003 et 0009, S4.2)."""

    QUESTION = "Combien de jours de télétravail par semaine ?"

    def setUp(self):
        self.registry = SwappableRegistry(hashing_registry(stem=6))
        self.ai_server = create_ai_server(self.registry, "127.0.0.1", 0, quiet=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.config = replace(AppConfig.load(ROOT / "config/app.toml"), ai_base_url=serve(self.ai_server),
                              index_path=Path(self.tmp.name) / "index.json")
        self.assertTrue(self.config.decorators.get("cache_embeddings"))
        self.alice = self.config.user("alice")

    def tearDown(self):
        self.ai_server.shutdown()
        self.ai_server.server_close()
        self.tmp.cleanup()

    def test_reindexing_follows_the_served_model_and_the_cache_does_not_hide_it(self):
        container = build(self.config)
        self.assertEqual(container.index_corpus.execute().embedding_model, "hashing-256-stem6")
        container.ask_question.execute(self.alice, self.QUESTION)             # mémorisée par le cache
        self.assertTrue(container.check_status.execute().up_to_date)

        # Nouveau modèle derrière le même alias, MÊME dimension : sans contrôle, la panne serait silencieuse.
        self.registry.current = hashing_registry(stem=4)
        # Question déjà posée : vecteurs du même modèle que l'index, la réponse reste cohérente.
        self.assertEqual(container.ask_question.execute(self.alice, self.QUESTION).trace.embedding_model,
                         "hashing-256-stem6")
        with self.assertRaises(IndexModelMismatchError):                     # question nouvelle : détecté
            container.ask_question.execute(self.alice, "Quel est le plafond d'un repas d'affaires ?")
        self.assertFalse(container.check_status.execute().up_to_date)       # status n'utilise pas le cache

        # La réindexation suit le modèle servi, et la question déjà posée repart au service.
        self.assertEqual(container.index_corpus.execute().embedding_model, "hashing-256-stem4")
        self.assertEqual(container.ask_question.execute(self.alice, self.QUESTION).trace.embedding_model,
                         "hashing-256-stem4")
        self.assertTrue(container.check_status.execute().up_to_date)

    def test_new_vectors_under_the_same_model_name_are_picked_up_by_reindexing(self):
        """Moteur sans empreinte : on change les poids, pas l'identifiant du modèle."""
        self.registry.current = WithoutFingerprint(hashing_registry(stem=6))
        container = build(self.config)
        before = container.index_corpus.execute()
        container.ask_question.execute(self.alice, self.QUESTION)             # mémorisée par le cache

        self.registry.current = WithoutFingerprint(hashing_registry(stem=4))
        after = container.index_corpus.execute()
        self.assertEqual(after.index_id, before.index_id)   # rien, dans l'identifiant, ne trahit le changement
        again = container.ask_question.execute(self.alice, self.QUESTION)
        fresh = build(self.config).ask_question.execute(self.alice, self.QUESTION)   # un processus neuf
        self.assertEqual(again.trace.retrieved, fresh.trace.retrieved)


if __name__ == "__main__":
    unittest.main()
