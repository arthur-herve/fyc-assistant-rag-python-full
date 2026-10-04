"""Bout en bout, en HTTP réel : application ↔ service IA, en mode hors-ligne."""

import contextlib
import csv
import io
import json
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_service.registry import ModelRegistry
from ai_service.server import create_server as create_ai_server
from assistant.application.errors import IndexModelMismatchError
from assistant.composition import AppConfig, build
from assistant.domain.model import AnswerStatus
from assistant.interface.benchmark import EvalQuestion, load_questions, run_benchmark
from assistant.interface.http_api import create_server as create_app_server

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "experiences"))

import _commun  # noqa: E402
import cace_decoupage  # noqa: E402
import changement_embeddings  # noqa: E402
import changement_generateur  # noqa: E402
import stabilite  # noqa: E402
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
        cls.config_path = config_path
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
        written = []

        def log(message):
            if "passage 1/1 terminé" in message:
                # Chaque ligne est sur le disque dès qu'elle est connue, pas en fin de campagne.
                with open(out / "resultats.csv", encoding="utf-8", newline="") as handle:
                    written.append(len(list(csv.DictReader(handle))))

        summary = run_benchmark(self.config, ["hashing"], ["extractive"], questions, runs=1, out_dir=out,
                                validation=questions[:2], log=log)
        self.assertEqual(written, [len(questions)])
        with open(out / "resultats.csv", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["question_id"] for r in rows], [q.id for q in questions])
        self.assertEqual({r["generation_model_id"] for r in rows if r["status"] == "answered"}, {"extractive"})
        self.assertTrue((out / "rapport.md").is_file())
        self.assertEqual(json.loads((out / "synthese.json").read_text(encoding="utf-8")), summary.to_json())
        self.assertEqual(summary.retrieval[0].validation.questions, 2)
        # Une option fausse est refusée avant tout travail : pas de dossier de résultats à moitié rempli.
        with self.assertRaises(ValueError):
            run_benchmark(self.config, ["hashing"], ["extractive"], questions, runs=1,
                          out_dir=Path(self.tmp.name) / "jamais", min_score_mode="abc", log=lambda *_: None)
        self.assertFalse((Path(self.tmp.name) / "jamais").exists())

    def test_the_benchmark_searches_with_the_rights_of_each_question(self):
        """La recherche du banc passe par SearchPassages (droits, contrôle du modèle), pas par l'index directement."""
        question = "Quelle est la fourchette de salaire d'un consultant senior ?"
        # Même question, même attente : seul bruno a le droit de lire la grille des salaires.
        questions = [EvalQuestion("rh", question, "bruno", True, ("grille-salaires",), (), ()),
                     EvalQuestion("sans-droit", question, "alice", True, ("grille-salaires",), (), ())]
        summary = run_benchmark(self.config, ["hashing"], ["extractive"], questions, runs=1,
                                out_dir=Path(self.tmp.name) / "banc-droits", log=lambda *_: None)
        self.assertEqual((summary.retrieval[0].hit_at_1, summary.retrieval[0].hit_at_k), (0.5, 0.5))
        self.assertEqual(summary.generation[0].forbidden_leaks, 0)

    def test_the_benchmark_says_when_the_default_threshold_applies(self):
        questions = load_questions(ROOT / "eval" / "questions.json")[:2]
        out, lines = Path(self.tmp.name) / "banc-default", []
        summary = run_benchmark(self.config, ["hashing"], ["extractive"], questions, runs=1, out_dir=out,
                                log=lines.append)
        self.assertTrue(summary.retrieval[0].configured_threshold_is_default)   # [retrieval.min_score] : `default` seul
        self.assertIn("seuil configuré=0.15 (default : aucun seuil pour cet alias)", "\n".join(lines))
        self.assertIn("| 0.15 (default) |", (out / "rapport.md").read_text(encoding="utf-8"))

        out = Path(self.tmp.name) / "banc-calibre"
        summary = run_benchmark(replace(self.config, min_scores={"hashing": 0.15}), ["hashing"], ["extractive"],
                                questions, runs=1, out_dir=out, log=lambda *_: None)
        self.assertFalse(summary.retrieval[0].configured_threshold_is_default)
        self.assertNotIn("(default)", (out / "rapport.md").read_text(encoding="utf-8"))

    def test_the_benchmark_logs_its_thresholds_and_report_path_and_writes_lf_only(self):
        """Fins de ligne \\n (Windows compris), chemin du rapport avec des « / » ; le journal donne le seuil
        configuré (0) et le seuil imposé (1), y compris pour la validation."""
        questions = load_questions(ROOT / "eval" / "questions.json")[:2]
        out, lines = Path(self.tmp.name) / "banc-octets", []
        run_benchmark(replace(self.config, min_scores={"hashing": 0.0}), ["hashing"], ["extractive"], questions,
                      runs=1, out_dir=out, min_score_mode="1", validation=questions[:1], log=lines.append)
        log = "\n".join(lines)
        self.assertEqual(float(re.search(r" · seuil configuré=(\S+) · ", log).group(1)), 0)
        self.assertEqual(float(re.search(r" · seuil utilisé=(\S+)\n", log).group(1)), 1)
        self.assertEqual(float(re.search(r"\(1 questions jamais vues, seuil (\S+)\) :", log).group(1)), 1)
        self.assertEqual(lines[-1], f"\nRapport : {(out / 'rapport.md').as_posix()}")
        for name in ("rapport.md", "synthese.json"):
            with self.subTest(name):
                self.assertNotIn(b"\r", (out / name).read_bytes())

    def test_two_benchmarks_of_the_same_second_keep_their_own_folder(self):
        """Sans --out, le dossier est daté à la seconde : un second banc lancé dans la même seconde écrivait dans le
        même dossier, et son rapport remplaçait celui du premier, sans rien dire. Il reçoit « -2 » (new_out_dir), et
        tous ses fichiers y vont. --out, lui, reste le dossier donné, même s'il existe déjà (la commande :
        test_cli.py)."""
        questions = load_questions(ROOT / "eval" / "questions.json")
        base = Path(self.tmp.name) / "resultats" / "20261001-164152"
        lines = []
        for count in (1, 2):
            run_benchmark(self.config, ["hashing"], ["extractive"], questions[:count], runs=1, out_dir=base,
                          new_out_dir=True, log=lines.append)
        second = base.with_name("20261001-164152-2")
        self.assertEqual([line for line in lines if line.startswith("\nRapport : ")],
                         [f"\nRapport : {(folder / 'rapport.md').as_posix()}" for folder in (base, second)])
        for count, folder in ((1, base), (2, second)):
            with self.subTest(folder=folder.name):
                self.assertIn(f"\n{count} questions · ", (folder / "rapport.md").read_text(encoding="utf-8"))
                self.assertEqual(sorted(p.name for p in folder.iterdir()),
                                 ["index-hashing.json", "rapport.md", "resultats.csv", "synthese.json"])
        run_benchmark(self.config, ["hashing"], ["extractive"], questions[:3], runs=1, out_dir=base,
                      log=lambda *_: None)
        self.assertIn("\n3 questions · ", (base / "rapport.md").read_text(encoding="utf-8"))
        self.assertFalse(base.with_name("20261001-164152-3").exists())

    def test_the_benchmark_uses_the_prompt_it_is_given(self):
        """--prompt (run_benchmark(prompt_name=…)) : le rapport nomme sa version."""
        out = Path(self.tmp.name) / "banc-prompt"
        run_benchmark(self.config, ["hashing"], ["extractive"], load_questions(ROOT / "eval" / "questions.json")[:1],
                      runs=1, out_dir=out, prompt_name="answer-v2", log=lambda *_: None)
        prompts = self.container.prompts
        self.assertNotEqual(prompts.get("answer-v2").version, prompts.get("answer").version)
        self.assertIn(f"prompt `{prompts.get('answer-v2').version}`", (out / "rapport.md").read_text(encoding="utf-8"))

    def experiment(self, script, *options: str, config: Path | None = None) -> tuple[str | None, str, str]:
        """Une expérience lancée comme dans un terminal : message de sortie (None si elle aboutit), sortie, erreurs.
        Sans `config`, la configuration de la classe."""
        argv = [f"{script.__name__}.py", "--config", str(config or self.config_path), "--questions",
                str(ROOT / "eval" / "questions.json"), *options]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr), \
                contextlib.redirect_stdout(stdout):
            try:
                _commun.run(script.main)
                stopped = None
            except SystemExit as stop:
                stopped = str(stop.code)
        return stopped, stdout.getvalue(), stderr.getvalue()

    def test_an_experiment_says_when_the_default_threshold_applies(self):
        """Sans seuil configuré, le modèle principal et `--other` reçoivent `default` : l'expérience le dit
        (console et rapport, ADR 0004). Rapport en \\n et chemin avec des « / », sous Windows aussi."""
        out = Path(self.tmp.name) / "exp-embeddings"
        stopped, stdout, stderr = self.experiment(changement_embeddings, "--limit", "2", "--other", "hashing-512",
                                                  "--out", str(out))
        self.assertIsNone(stopped)
        warnings = [f"Attention : aucun seuil de pertinence configuré pour « {alias} » : valeur `default` 0.15 "
                    "(ADR 0004 : lancer le banc d'essai)" for alias in ("hashing", "hashing-512")]
        self.assertEqual(stderr.splitlines(), warnings)
        report = (out / "rapport.md").read_text(encoding="utf-8")
        for warning in warnings:
            self.assertIn(f"> {warning}", report)
        self.assertIn("| seuil | 0.15 | 0.15 |", report)
        self.assertNotIn(b"\r", (out / "rapport.md").read_bytes())
        self.assertTrue(stdout.endswith(f"\nRapport : {(out / 'rapport.md').as_posix()}\n"), stdout)

    def test_the_other_alias_of_an_experiment_is_checked_before_any_index(self):
        """--other doit être servi, et du bon type (GET /v1/models) : sinon, ni index ni instantané, et pas
        d'avertissement avant l'erreur."""
        out = Path(self.tmp.name) / "exp-autre"
        cases = [
            (changement_embeddings, "extractive", "modèle d'embeddings (servis : hashing, hashing-512)"),
            (changement_embeddings, "nexiste", "modèle d'embeddings (servis : hashing, hashing-512)"),
            (changement_generateur, "hashing", "modèle de génération (servis : extractive, extractive-bruite)"),
        ]
        for script, other, served in cases:
            with self.subTest(script=script.__name__, other=other):
                stopped, stdout, stderr = self.experiment(script, "--limit", "1", "--other", other, "--out", str(out))
                self.assertEqual(stopped, f"Erreur : le service IA ne sert pas « {other} » comme {served}")
                self.assertEqual((stdout, stderr), ("", ""))
                self.assertFalse(out.exists())

    def test_an_experiment_prints_its_steps_and_its_report_names_the_files_as_given(self):
        """La console : « Avant : 800 / 120 » (taille / recouvrement, comme les colonnes du rapport), une ligne par
        index et par instantané, « Rapport : » et son chemin. L'en-tête du rapport : les chemins tels qu'ils ont été
        donnés, avec des « / ». Le corpus Solvéo donne 15 morceaux en 800 / 120, et 42 en 300 / 50."""
        out = Path(self.tmp.name) / "exp-cace"
        stopped, stdout, _ = self.experiment(cace_decoupage, "--limit", "1", "--out", str(out))
        self.assertIsNone(stopped)
        for line in ("Avant : 800 / 120\n", "Après : 300 / 50\n"):
            self.assertIn(line, stdout)
        for name, chunks in (("avant", 15), ("apres", 42)):
            self.assertRegex(stdout, rf"  index « {name} » : {chunks} morceaux, hashing-256-stem6, 256 dim\., ")
            self.assertRegex(stdout, rf"  instantané « {name} » : 1 réponses en ")
        self.assertTrue(stdout.endswith(f"\nRapport : {(out / 'rapport.md').as_posix()}\n"), stdout)
        self.assertIn(f"\nConfiguration `{self.config_path.as_posix()}` · 1 questions de "
                      f"`{(ROOT / 'eval' / 'questions.json').as_posix()}` · corpus `solveo`.\n",
                      (out / "rapport.md").read_text(encoding="utf-8"))

    def test_stability_writes_its_mean_drift(self):
        out = Path(self.tmp.name) / "exp-stabilite"
        stopped, _, _ = self.experiment(stabilite, "--limit", "2", "--runs", "2", "--out", str(out))
        self.assertIsNone(stopped)
        # Générateur déterministe : aucune dérive.
        found = re.search(r"\*\*Dérive moyenne à configuration constante : (\S+)\*\* \(0 changement\(s\) de statut "
                          r"sur 1 comparaison\(s\)\)\.", (out / "rapport.md").read_text(encoding="utf-8"))
        self.assertIsNotNone(found)
        self.assertEqual(float(found.group(1)), 0)

    def test_two_experiments_of_the_same_second_keep_their_own_folder(self):
        """Sans --out, comme le banc : le second dossier reçoit « -2 », et ses index et instantanés le suivent (ils
        étaient écrits dans le dossier du premier). --out, lui, reste le dossier donné, même s'il existe déjà."""
        base = Path(self.tmp.name) / "resultats-exp"
        with mock.patch.object(_commun, "results_dir", lambda prefix, now: base / f"{prefix}20261001-164152"):
            outputs = [self.experiment(stabilite, "--limit", "1", "--runs", "2") for _ in range(2)]
        folders = [base / "exp-stabilite-20261001-164152", base / "exp-stabilite-20261001-164152-2"]
        outputs.append(self.experiment(stabilite, "--limit", "1", "--runs", "2", "--out", str(folders[0])))
        for number, ((stopped, stdout, _), folder) in enumerate(zip(outputs, [*folders, folders[0]]), 1):
            with self.subTest(run=number, folder=folder.name):
                self.assertIsNone(stopped)
                self.assertTrue(stdout.endswith(f"\nRapport : {(folder / 'rapport.md').as_posix()}\n"), stdout)
                self.assertEqual(sorted(p.name for p in folder.iterdir()),
                                 ["index-partage.json", "instantanes", "rapport.md"])
                self.assertEqual(len(list((folder / "instantanes").iterdir())), 2)
        self.assertFalse((base / "exp-stabilite-20261001-164152-3").exists())

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
