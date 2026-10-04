"""La ligne de commande (python -m assistant), appelée comme on la tape : une saisie fausse est
refusée avant tout travail (code 1 ; 2 veut dire « à refaire » pour status), et les affichages
se lisent (null, true, [a, b], « 33 % »)."""

import contextlib
import io
import json
import logging
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from ai_service.registry import ModelRegistry
from ai_service.server import create_server as create_ai_server
from assistant.__main__ import main
from assistant.application.ports import IndexManifest, Snapshot, SnapshotEntry
from assistant.application.snapshots import compare_snapshots, config_value_to_text
from assistant.application.status import StatusReport
from assistant.composition import PROMPTS_DIR, AppConfig, UnknownUserError, build
from assistant.domain.model import Answer, AnswerStatus, AnswerTrace, Source
from assistant.interface import benchmark, http_api
from assistant.interface.presenter import (
    answer_to_text, comparison_to_text, manifest_to_dict, status_to_dict, status_to_text,
)

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Combien de jours de télétravail par semaine ?"
WARNING = "Attention : aucun seuil de pertinence configuré pour « hashing » : valeur `default` 0.15"
SPLITTER = {"type": "paragraph", "max_chars": 800, "overlap_chars": 120, "include_title": True}
# Ce qu'affiche ask --json pour la réponse de test_the_json_of_an_answer_is_written_as_frozen.
ANSWER_JSON = (
    '{\n'
    '  "question": "Combien ?",\n'
    '  "status": "answered",\n'
    '  "text": "Deux jours\U000000a0: « oui » \\"x\\"\\nfin \U0001f600",\n'
    '  "sources": [\n'
    '    {\n'
    '      "number": 1,\n'
    '      "document_id": "a",\n'
    '      "document_title": "Congés",\n'
    '      "chunk_id": "a#0"\n'
    '    }\n'
    '  ],\n'
    '  "trace": {\n'
    '    "index_id": "id",\n'
    '    "embedding_model": "hashing",\n'
    '    "generation_model": "extractive",\n'
    '    "prompt_version": "v1+abc",\n'
    '    "retrieved": [\n'
    '      {\n'
    '        "chunk_id": "a#0",\n'
    '        "score": 1.0\n'
    '      },\n'
    '      {\n'
    '        "chunk_id": "a#1",\n'
    '        "score": 0.25\n'
    '      }\n'
    '    ],\n'
    '    "min_score": 0.4,\n'
    '    "attempts": 1\n'
    '  }\n'
    '}'
)


def run(*argv: str) -> tuple[int, str, str]:
    """Code de retour, sortie et erreurs, comme dans un terminal (argparse sort par SystemExit)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as stop:
            code = stop.code
    return code, out.getvalue(), err.getvalue()


class CommandLineTest(unittest.TestCase):
    """Un vrai service IA hors-ligne : index, status et ask passent par toute la pile."""

    @classmethod
    def setUpClass(cls):
        registry = ModelRegistry.from_dict({
            "embedding": {"hashing": {"backend": "hashing", "dimension": 256},
                          "hashing-512": {"backend": "hashing", "dimension": 512}},
            "generation": {"extractive": {"backend": "extractive"}},
        })
        cls.ai_server = create_ai_server(registry, "127.0.0.1", 0, quiet=True)
        threading.Thread(target=cls.ai_server.serve_forever, daemon=True).start()
        cls.ai_url = f"http://127.0.0.1:{cls.ai_server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.ai_server.shutdown()
        cls.ai_server.server_close()

    def setUp(self):
        # main() configure le journal (logging.basicConfig) : on rend l'état d'avant après chaque test.
        root = logging.getLogger()
        self.addCleanup(setattr, root, "handlers", root.handlers[:])
        self.addCleanup(root.setLevel, root.level)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.config = self.write_config("app.toml", self.ai_url)
        self.missing = str(self.dir / "absente.toml")   # lue seulement si la saisie est juste

    def write_config(self, name: str, ai_url: str) -> str:
        """Pas de seuil pour « hashing » (repli sur `default`), un seuil pour « hashing-512 »."""
        path = self.dir / name
        path.write_text(f"""
[ai_service]
base_url = "{ai_url}"
embedding_model = "hashing"
generation_model = "extractive"
timeout_seconds = 10
[corpus]
directory = "{(ROOT / 'corpus' / 'solveo').as_posix()}"
[index]
path = "{(self.dir / 'index.json').as_posix()}"
[snapshots]
directory = "{(self.dir / 'instantanes').as_posix()}"
[retrieval.min_score]
default = 0.15
hashing-512 = 0.15
[users.alice]
groups = ["tous"]
""", encoding="utf-8")
        return str(path)

    def assert_refused(self, argv, message):
        """Code 1, rien sur la sortie, `message` sur la sortie d'erreur ; la configuration (absente) n'est pas lue."""
        code, out, err = run(*argv, "--config", self.missing)
        self.assertEqual((code, out), (1, ""), err)
        self.assertIn(message, err)
        self.assertNotIn("absente.toml", err)

    def test_a_mistyped_abbreviated_or_foreign_option_is_refused_before_any_work(self):
        # argparse refuse ce que la commande ne lit pas, sans abréviation (« --us » n'est pas « --user ») ; le
        # message est en français et renvoie à l'aide de la commande.
        cases = {
            ("ask", QUESTION, "--us", "bruno"): "--us bruno",
            ("status", "--jsn"): "--jsn",
            ("status", "-json"): "-json",
            ("serve", "--json"): "--json",   # une option d'une autre commande
            ("benchmark", "-v"): "-v",
            ("--json", "status"): "--json",   # placée avant la commande : assistant lui-même ne la lit pas
        }
        for argv, unknown in cases.items():
            with self.subTest(argv):
                self.assert_refused(argv, f"assistant: error: option(s) ou argument(s) non reconnu(s) : {unknown} "
                                          "(voir assistant <commande> -h)\n")

    def test_a_double_dash_ends_the_options(self):
        # Après « -- », tout est un argument, même ce qui commence par un tiret ; les options se placent donc avant.
        code, out, err = run("ask", "--config", self.missing, "--", "-vingt degrés ?")   # l'exemple de l'aide
        self.assertEqual((code, out), (1, ""))
        self.assertIn("absente.toml", err)   # la saisie est acceptée : c'est le fichier qui manque
        code, out, err = run("status", "--config", self.missing, "--", "--json")
        self.assertEqual((code, out), (1, ""))
        self.assertRegex(err, r"non reconnu\(s\) : (-- )?--json \(")   # un argument, en trop pour status
        self.assertNotIn("absente.toml", err)

    def test_a_question_that_starts_with_a_dash_follows_a_double_dash(self):
        self.assertEqual(run("index", "--config", self.config)[0], 0)
        code, out, err = run("ask", "--json", "--config", self.config, "--", "-télétravail")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["question"], "-télétravail")

    def test_a_question_made_of_blanks_is_empty(self):
        # U+001C (séparateur) est un blanc pour str.strip(), comme l'espace et la tabulation : question vide, code 1.
        self.assertEqual(run("index", "--config", self.config)[0], 0)
        for question in (" \t", "\x1c"):
            with self.subTest(question=question):
                code, out, err = run("ask", "--config", self.config, "--", question)
                self.assertEqual((code, out), (1, ""))
                self.assertIn("Erreur : La question est vide.", err)

    def test_an_extra_argument_is_refused(self):
        cases = {
            ("status", "foo"): "foo",
            ("status", "-1"): "-1",   # un nombre négatif est une valeur, pas une option
            ("snapshot", "compare", "a", "b", "c"): "c",
            ("ask", "Combien", "de", "jours", "?"): "de jours ?",   # la question, entre guillemets
        }
        for argv, extra in cases.items():
            with self.subTest(argv):
                self.assert_refused(argv, f"option(s) ou argument(s) non reconnu(s) : {extra} "
                                          "(voir assistant <commande> -h)\n")

    def test_the_options_of_each_command_reach_the_configuration(self):
        for argv in (["index", "--if-stale", "-v"], ["ask", QUESTION, "--user", "alice", "--json", "--verbose"],
                     ["status", "--json"], ["snapshot", "record", "x", "--questions", "q.json", "--limit", "2"],
                     ["snapshot", "compare", "a", "b", "--prompt", "answer-v2"], ["snapshot", "list", "-v"],
                     ["serve", "--host", "127.0.0.1", "--port", "0", "--quiet"],
                     ["benchmark", "--embedding-model", "hashing-512", "--generation-model", "extractive",
                      "--prompt", "answer-v2"],
                     # Une question qui commence par un tiret (elle contient une espace), une valeur avec espace.
                     ["ask", "-5 jours ?"], ["ask", "- Combien de jours ?", "--user=bruno martin"],
                     ["status", "-vv"],   # -v deux fois, pour argparse
                     # Les bornes : le port 65535, et --limit 1.
                     ["serve", "--port", "65535"], ["snapshot", "record", "x", "--limit", "1"],
                     ["benchmark", "--limit", "1"]):
            with self.subTest(argv):
                code, _, err = run(*argv, "--config", self.missing)
                self.assertEqual(code, 1)
                self.assertIn("absente.toml", err)   # la saisie est acceptée : c'est le fichier qui manque
                self.assertNotIn("non reconnu", err)

    def test_benchmark_embedding_model_and_generation_model_name_one_model_each(self):
        received = []
        with mock.patch.object(benchmark, "main_benchmark", received.append):
            run("benchmark", "--embedding-model", "hashing-512", "--generation-model", "extractive")
            run("benchmark", "--embedding", "hashing", "nomic", "--embedding-model", "hashing-512")   # --embedding prime
            run("benchmark")
        self.assertEqual([(args.embedding, args.generation) for args in received],
                         [(["hashing-512"], ["extractive"]), (["hashing", "nomic"], None), (None, None)])

    def test_without_out_the_benchmark_folder_is_dated_under_the_project_root(self):
        # Sans --out : eval/resultats/AAAAMMJJ-HHMMSS sous la racine du projet, en chemin complet.
        with mock.patch.object(benchmark, "run_benchmark") as run_benchmark:
            code, _, err = run("benchmark", "--limit", "1", "--config", self.config)
        self.assertEqual(code, 0, err)
        out_dir = run_benchmark.call_args.kwargs["out_dir"]
        self.assertEqual(out_dir.parent, ROOT / "eval" / "resultats")
        self.assertRegex(out_dir.name, r"\A\d{8}-\d{6}\Z")

    def test_an_empty_out_is_the_dated_folder(self):
        # --out "" vaut l'option absente : le dossier daté, jamais un dossier qui existe déjà.
        with mock.patch.object(benchmark, "run_benchmark") as run_benchmark:
            code, _, err = run("benchmark", "--limit", "1", "--config", self.config, "--out", "")
        self.assertEqual(code, 0, err)
        out_dir = run_benchmark.call_args.kwargs["out_dir"]
        self.assertEqual(out_dir.parent, ROOT / "eval" / "resultats")
        self.assertRegex(out_dir.name, r"\A\d{8}-\d{6}\Z")
        self.assertTrue(run_benchmark.call_args.kwargs["new_out_dir"])

    def test_with_out_the_benchmark_writes_in_the_folder_given_even_if_it_exists(self):
        # Seul le dossier par défaut, daté à la seconde, n'est jamais un dossier qui existe déjà (« -2 », « -3 »…) :
        # --out reste le dossier donné, même s'il existe.
        questions = self.dir / "questions.json"
        questions.write_text(json.dumps({"questions": [{"id": "a", "user": "alice", "question": "Combien ?"}]}),
                             encoding="utf-8")
        out_dir = self.dir / "banc"
        out_dir.mkdir()
        code, out, err = run("benchmark", "--runs", "1", "--questions", str(questions), "--out", str(out_dir),
                             "--config", self.config)
        self.assertEqual(code, 0, err)
        self.assertTrue(out.endswith(f"\nRapport : {(out_dir / 'rapport.md').as_posix()}\n"), out)
        self.assertEqual(sorted(p.name for p in out_dir.iterdir()),
                         ["index-hashing.json", "rapport.md", "resultats.csv", "synthese.json"])
        self.assertFalse((self.dir / "banc-2").exists())

    def test_record_takes_the_limit_before_checking_the_users(self):
        questions = self.dir / "questions.json"
        questions.write_text(json.dumps({"questions": [
            {"id": "a", "user": "alice", "question": "Combien de jours ?"},
            {"id": "c", "user": "claire", "question": "Quel budget ?"}]}), encoding="utf-8")
        code, _, err = run("snapshot", "record", "x", "--questions", str(questions), "--limit", "1", "--config", self.config)
        self.assertEqual(code, 1)
        self.assertIn("Aucun index", err)   # et non « utilisateur inconnu : claire »

    def test_benchmark_refuses_a_limit_of_zero_like_the_experiments(self):
        # Le banc, comme les expériences et snapshot record, refuse --limit 0 avant tout
        # travail : ni service IA interrogé, ni dossier de résultats.
        config = self.write_config("coupe.toml", "http://127.0.0.1:1")
        out_dir = self.dir / "banc"
        code, out, err = run("benchmark", "--limit", "0", "--out", str(out_dir), "--config", config)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("Erreur : --limit doit valoir au moins 1, pas 0", err)
        self.assertFalse(out_dir.exists())

    def test_benchmark_checks_every_option_before_the_ai_service(self):
        # Service IA injoignable : une option fausse est dite quand même, avant de l'interroger, et sans
        # dossier de résultats. --prompt compris. Le message entier, jusqu'à la fin de ligne : rien n'y est ajouté.
        config = self.write_config("coupe.toml", "http://127.0.0.1:1")
        questions = self.dir / "questions.json"
        questions.write_text(json.dumps({"questions": [{"id": "a", "user": "alice", "question": "Combien ?"}]}),
                             encoding="utf-8")
        out_dir = self.dir / "banc"
        cases = {
            ("--runs", "0"): "Erreur : --runs doit valoir au moins 1, pas 0\n",
            ("--min-score", "abc"): "Erreur : --min-score attend config, auto ou un nombre de [-1, 1] (ex. 0.6), "
                                    "pas « abc »\n",
            ("--max-chars", "50"): "Erreur : max_chars doit valoir au moins 100\n",
            ("--overlap-chars", "900"): "Erreur : overlap_chars doit être compris entre 0 et max_chars / 2\n",
            ("--prompt", "nexiste"): "nexiste",
        }
        for options, message in cases.items():
            with self.subTest(options):
                code, out, err = run("benchmark", *options, "--questions", str(questions), "--out", str(out_dir),
                                     "--config", config)
                self.assertEqual((code, out), (1, ""), err)
                self.assertIn(message, err)
                self.assertNotIn("injoignable", err)
                self.assertFalse(out_dir.exists())

    def test_an_unreachable_ai_service_is_an_error_before_any_folder(self):
        # « Erreur : », le message du service injoignable, code 1, avant tout dossier ; c'était un SystemExit
        # sans préfixe, avec le message d'urllib. Les expériences : test_banc.py. Un vrai refus de connexion (2 s sous
        # Windows), le seul des tests de l'application : ailleurs, il est simulé (test_banc.connection_refused).
        config = self.write_config("coupe.toml", "http://127.0.0.1:1")
        questions = self.dir / "questions.json"
        questions.write_text(json.dumps({"questions": [{"id": "a", "user": "alice", "question": "Combien ?"}]}),
                             encoding="utf-8")
        out_dir = self.dir / "banc"
        self.assertEqual(run("benchmark", "--questions", str(questions), "--out", str(out_dir), "--config", config),
                         (1, "", "Erreur : service IA injoignable (http://127.0.0.1:1). Lancez-le d'abord : "
                                 "python -m ai_service\n"))
        self.assertFalse(out_dir.exists())

    def test_the_help_of_min_score_names_its_bounds(self):
        code, out, _ = run("benchmark", "--help")
        self.assertEqual(code, 0)
        self.assertIn("--min-score config|auto|<n de [-1, 1]>", out)

    def test_the_help_says_how_to_end_the_options_and_bounds_limit(self):
        # argparse coupe les lignes à sa guise, d'où les blancs ramenés à un. L'exemple a besoin de « -- » : sans lui,
        # argparse lit -v dans « -vingt degrés ? » ; « -5 jours ? » passe sans.
        self.assertIn("« -- » termine les options : ce qui suit est un argument (ask -- \"-vingt degrés ?\" : sans « -- », "
                      "cette question qui commence par -v serait prise pour une option).",
                      " ".join(run("--help")[1].split()))
        for command in (["benchmark"], ["snapshot", "record"]):
            with self.subTest(command):
                self.assertIn("--limit LIMIT ne garder que les N premières questions (au moins 1)",
                              " ".join(run(*command, "--help")[1].split()))

    def test_the_help_of_config_names_assistant_config_for_every_command(self):
        # Sans --config, ASSISTANT_CONFIG (test_an_empty_value_is_no_value), comme le dit
        # l'aide des expériences (tools/experiences/_commun.py) ; le banc n'avait pas d'aide pour --config.
        for command in (["index"], ["ask"], ["serve"], ["status"], ["snapshot", "record"], ["snapshot", "compare"],
                        ["snapshot", "list"], ["benchmark"]):
            with self.subTest(command):
                self.assertIn("--config CONFIG fichier de configuration (défaut : ASSISTANT_CONFIG, sinon config/app.toml)",
                              " ".join(run(*command, "--help")[1].split()))

    def test_a_usage_error_exits_with_1_and_help_with_0(self):
        # Les messages d'argparse (en anglais) pour ces cas-là ; le code est 1, comme toute erreur de saisie.
        for argv in ([], ["ask"], ["status", "--json=1"], ["index", "--config"], ["serve", "--port", "abc"],
                     ["inconnue"], ["snapshot"], ["snapshot list"]):
            with self.subTest(argv):
                self.assertEqual(run(*argv)[0], 1)   # jamais 2 : status l'emploie pour « à refaire »
        # « --us » n'est pas lu comme --user (qui réclamerait sa valeur).
        for argv in (["--help"], ["ask", "-h"], ["--json", "status", "--help"], ["status", "--jsn", "-h"],
                     ["ask", "--us", "-h"]):
            with self.subTest(argv):
                self.assertEqual(run(*argv)[0], 0)

    def test_a_port_out_of_range_or_a_limit_below_one_is_refused_before_any_work(self):
        # --limit : au moins 1 partout, même message que le banc et les expériences. Un nombre négatif est lu
        # comme une valeur (« --port -1 »), puis refusé hors bornes.
        cases = {
            ("serve", "--port", "99999"): "l'option --port attend un port entre 0 et 65535, pas « 99999 »",
            ("serve", "--port", "-1"): "l'option --port attend un port entre 0 et 65535, pas « -1 »",
            ("serve", "--port=65536"): "l'option --port attend un port entre 0 et 65535, pas « 65536 »",
            ("snapshot", "record", "x", "--limit", "-1"): "--limit doit valoir au moins 1, pas -1",
            ("snapshot", "record", "x", "--limit", "0"): "--limit doit valoir au moins 1, pas 0",
            ("benchmark", "--limit=-2"): "--limit doit valoir au moins 1, pas -2",
            ("benchmark", "--limit", "0"): "--limit doit valoir au moins 1, pas 0",
        }
        for argv, message in cases.items():
            with self.subTest(argv):
                self.assert_refused(argv, f"Erreur : {message}\n")

    def test_an_unknown_prompt_names_its_folder_and_the_known_ones(self):
        # « prompt introuvable : … », et non plus « fichier ou dossier inaccessible — [Errno 2] ».
        code, out, err = run("status", "--prompt", "nexiste", "--config", self.config)
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(err, f"Erreur : prompt introuvable : nexiste dans {PROMPTS_DIR} "
                              f"(connus : answer, answer-v2)\n")

    def test_an_empty_value_is_no_value(self):
        """--config, --embedding-model, --generation-model ou --prompt donnés vides valent la configuration, comme sans
        l'option (« valeur or défaut ») : une seule règle pour ces quatre options."""
        self.assertEqual(run("index", "--config", self.config)[0], 0)
        status, ask = run("status", "--config", self.config), run("ask", QUESTION, "--config", self.config)
        self.assertEqual((status[0], ask[0]), (0, 0))
        for option in ("--embedding-model", "--generation-model", "--prompt"):
            with self.subTest(option):
                self.assertEqual(run("status", option, "", "--config", self.config), status)
                self.assertEqual(run("ask", QUESTION, option, "", "--config", self.config), ask)
        with mock.patch.dict(os.environ, {"ASSISTANT_CONFIG": self.config}):
            self.assertEqual(run("status", "--config", ""), status)

    def test_assistant_log_shows_the_journal_at_info_level_or_below(self):
        """ASSISTANT_LOG : le nom d'un niveau de logging, sans tenir compte de la casse ; le journal des décorateurs
        (niveau INFO) se voit avec INFO ou DEBUG. Toute autre valeur vaut WARNING : passée telle quelle à logging,
        une valeur inconnue (ou vide) finissait en pile (« ValueError: Unknown level »). Avec -v, c'est INFO."""
        config = self.dir / "journal.toml"   # celle du test, avec le journal des décorateurs (comme config/app.toml)
        config.write_text(Path(self.config).read_text(encoding="utf-8") + "[decorators]\nlog = true\n", encoding="utf-8")
        self.assertEqual(run("index", "--config", str(config))[0], 0)
        for value, shown in (("INFO", True), ("info", True), ("Info", True), ("debug", True), ("notset", True),
                             ("WARNING", False), ("warn", False), ("error", False), ("CRITICAL", False),
                             ("fatal", False), ("verbeux", False), ("", False)):
            logging.getLogger().handlers = []   # basicConfig n'agit qu'ainsi, comme au lancement (rendus par setUp)
            with self.subTest(value), mock.patch.dict(os.environ, {"ASSISTANT_LOG": value}):
                code, out, err = run("ask", QUESTION, "--config", str(config))
                self.assertEqual(code, 0, err)
                self.assertIn("deux jours", out)
                self.assertEqual("[assistant] embeddings requête" in err, shown, err)
        logging.getLogger().handlers = []
        with mock.patch.dict(os.environ, {"ASSISTANT_LOG": "ERROR"}):
            code, _, err = run("ask", QUESTION, "-v", "--config", str(config))
        self.assertEqual(code, 0, err)
        self.assertIn("[assistant] embeddings requête", err)

    def test_the_threshold_warning_is_for_the_commands_that_search_with_the_model_they_use(self):
        code, out, err = run("index", "--config", self.config)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out)["embedding_model"], "hashing-256-stem6")
        code, _, err = run("status", "--config", self.config)
        self.assertEqual((code, err), (0, ""))
        code, out, err = run("ask", QUESTION, "--config", self.config)
        self.assertEqual(code, 0)
        self.assertIn(WARNING, err)
        self.assertIn("deux jours", out)
        # Le modèle réellement utilisé a son seuil : pas d'avertissement (l'index, lui, est à refaire).
        code, _, err = run("ask", QUESTION, "--embedding-model", "hashing-512", "--config", self.config)
        self.assertEqual(code, 1)
        self.assertNotIn("Attention", err)
        code, out, err = run("snapshot", "record", "ref", "--limit", "2", "--config", self.config)
        self.assertEqual(code, 0)
        self.assertIn(WARNING, err)
        for argv in (["snapshot", "list"], ["snapshot", "compare", "ref", "ref"]):
            with self.subTest(argv):
                code, _, err = run(*argv, "--config", self.config)
                self.assertEqual((code, err), (0, ""))

    def test_the_snapshot_commands_print_their_folder_and_the_sorted_configuration(self):
        code, out, _ = run("snapshot", "list", "--config", self.config)
        self.assertEqual(out, f"Aucun instantané dans {(self.dir / 'instantanes').as_posix()}\n")
        run("index", "--config", self.config)
        code, out, _ = run("snapshot", "record", "ref", "--limit", "2", "--config", self.config)
        lines = out.splitlines()
        self.assertEqual(lines[0], f"Instantané « ref » : 2 réponses, enregistré dans {(self.dir / 'instantanes').as_posix()}")
        keys = [line.split(" = ")[0].strip() for line in lines[1:]]
        self.assertEqual(keys, sorted(keys))   # clés triées, faciles à retrouver
        self.assertIn("  seed = null", lines)
        self.assertIn("  validate_output = true", lines)
        self.assertIn("  splitter = {include_title=true, max_chars=800, overlap_chars=120, type=paragraph}", lines)

    def test_index_if_stale_rebuilds_only_what_status_says_is_stale(self):
        code, out, _ = run("index", "--if-stale", "--config", self.config)
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("Index à refaire :\n  - aucun index : lancer l'indexation\n"), out)
        self.assertEqual(json.loads(out.split("\n", 2)[2])["embedding_model"], "hashing-256-stem6")
        self.assertEqual(run("index", "--if-stale", "--config", self.config)[:2], (0, "Index à jour : rien à refaire.\n"))
        # Service IA en erreur : on ne réindexe pas à l'aveugle.
        broken = self.write_config("cassee.toml", f"{self.ai_url}/introuvable")
        code, out, err = run("index", "--if-stale", "--config", broken)
        self.assertEqual((code, out), (3, ""))
        self.assertIn("Index non vérifié : le service IA n'a pas pu être interrogé, impossible de réindexer.", err)

    def test_serve_announces_the_port_really_chosen(self):
        real_create_server, servers = http_api.create_server, []

        def create_then_stop(*args, **kwargs):
            """Le vrai serveur, arrêté aussitôt démarré, comme par un Ctrl+C."""
            server = real_create_server(*args, **kwargs)
            servers.append((server, kwargs))

            def interrupted():
                raise KeyboardInterrupt

            server.serve_forever = interrupted
            return server

        with mock.patch.object(http_api, "create_server", create_then_stop):
            code, out, err = run("serve", "--port", "0", "--quiet", "--config", self.config)
        self.assertEqual(code, 0, err)
        (server, kwargs), = servers
        port = server.server_address[1]
        self.assertNotEqual(port, 0)
        self.assertTrue(kwargs["quiet"])
        self.assertIn(f"Application sur http://127.0.0.1:{port} (service IA : {self.ai_url})", out)
        self.assertIn("Arrêt de l'application.", out)
        self.assertIn(WARNING, err)   # serve répond aux questions : il cherche des passages

    def test_serve_announces_an_address_to_connect_to(self):
        """Une adresse à laquelle on peut se connecter : « localhost » tel quel, pas l'adresse résolue ; pour
        0.0.0.0 (toutes les interfaces, mais pas une adresse où se connecter), 127.0.0.1, en le disant. Un serveur
        factice, déjà arrêté, porte l'adresse qu'aurait le vrai : on n'écoute pas sur toutes les interfaces."""
        class Stopped:
            def __init__(self, address):
                self.server_address = address

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                pass

        cases = (("localhost", "127.0.0.1", "http://localhost:54321"),
                 ("0.0.0.0", "0.0.0.0", "http://127.0.0.1:54321, à l'écoute sur toutes les interfaces"))
        for host, bound, announced in cases:
            with self.subTest(host), mock.patch.object(http_api, "create_server",
                                                       lambda *args, bound=bound, **kwargs: Stopped((bound, 54321))):
                code, out, err = run("serve", "--host", host, "--port", "0", "--config", self.config)
            self.assertEqual(code, 0, err)
            self.assertIn(f"Application sur {announced} (service IA : {self.ai_url})", out)

    def test_serve_on_a_taken_port_says_so(self):
        with socket.socket() as taken:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):   # Windows : sans lui, un second bind partagerait le port
                taken.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            code, out, err = run("serve", "--port", str(taken.getsockname()[1]), "--config", self.config)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("Erreur : impossible d'écouter (port déjà pris, adresse inconnue ou non autorisée) — ", err)
        self.assertNotIn("fichier", err)


class ReadableDisplayTest(unittest.TestCase):
    """Des affichages qui se lisent : null, true, [a, b], « 33 % », clés du découpage triées."""

    def test_an_unknown_user_names_the_known_ones_without_brackets(self):
        """Même texte en ligne de commande et dans le 403 de l'API HTTP."""
        config = AppConfig.load(ROOT / "config" / "app.toml")
        expected = "utilisateur inconnu : mallory (connus : alice, bruno, claire)"
        with self.assertRaises(UnknownUserError) as caught:
            config.user("mallory")
        self.assertEqual(str(caught.exception), expected)
        server = http_api.create_server(build(config), "127.0.0.1", 0, quiet=True)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/v1/ask",
                                         data=json.dumps({"user": "mallory", "question": "?"}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=5)
        self.addCleanup(refused.exception.close)
        self.assertEqual(refused.exception.code, 403)
        self.assertEqual(json.loads(refused.exception.read())["error"], {"code": "unknown_user", "message": expected})

    def test_configuration_values(self):
        # null, true et false plutôt que None, True et False ; listes et dictionnaires sans guillemets, clés triées.
        self.assertEqual([config_value_to_text(v) for v in (None, True, False, 4, "hashing")],
                         ["null", "true", "false", "4", "hashing"])
        self.assertEqual(config_value_to_text(SPLITTER),
                         "{include_title=true, max_chars=800, overlap_chars=120, type=paragraph}")
        self.assertEqual(config_value_to_text(["a", None, True, 0.5]), "[a, null, true, 0.5]")

    def comparison(self, changed: int, compared: int, configuration=None):
        def snapshot(name, configuration, text):
            entries = tuple(SnapshotEntry(f"q{i}", "alice", "?", "answered", ("teletravail", "conges"),
                                          text if i < changed else "même")
                            for i in range(compared))
            return Snapshot(name, "2026-09-11", configuration, entries)
        before, after = configuration or ({}, {})
        return compare_snapshots(snapshot("a", before, "avant"), snapshot("b", after, "après"))

    def test_the_drift_rate_lists_and_configuration_differences(self):
        text = comparison_to_text(self.comparison(1, 3, ({"seed": None, "validate_output": True, "splitter": SPLITTER},
                                                         {"seed": 42, "validate_output": False,
                                                          "splitter": {**SPLITTER, "max_chars": 300}})))
        self.assertIn("  taux de dérive      : 33 %", text)
        self.assertIn("  avant : answered [teletravail, conges] « avant »", text)
        self.assertIn("  - seed : null → 42", text)
        self.assertIn("  - validate_output : true → false", text)
        self.assertIn("  - splitter : {include_title=true, max_chars=800, overlap_chars=120, type=paragraph} → "
                      "{include_title=true, max_chars=300, overlap_chars=120, type=paragraph}", text)

    def test_the_whole_comparison_line_by_line(self):
        """Le texte entier, ligne à ligne : celui de snapshot compare et des rapports d'expérience."""
        self.assertEqual(comparison_to_text(self.comparison(1, 2, ({"seed": None}, {"seed": 42}))).split("\n"), [
            "Comparaison : a → b", "", "Différences de configuration", "  - seed : null → 42", "",
            "Dérive", "  questions comparées : 2", "  réponses modifiées  : 1", "  taux de dérive      : 50 %", "",
            "| Nature | Nombre | Lecture |", "|---|---|---|",
            "| statut modifié | 0 | changement de comportement : refus devenu réponse, ou l'inverse |",
            "| sources modifiées | 0 | même décision, autres documents cités |",
            "| texte modifié | 1 | mêmes sources, même statut, texte différent : à relire, le sens a pu changer "
            "(Oui devenu Non…) |",
            "| identique | 1 | rien n'a bougé |",
            "| absente d'un des deux | 0 | question présente d'un seul côté |",
            "", "q0 [texte modifié]", "  avant : answered [teletravail, conges] « avant »",
            "  après : answered [teletravail, conges] « après »",
        ])

    def test_answers_are_cut_at_90_characters(self):
        """« avant » et « après » : les 90 premiers caractères (points de code, text[:90]). Un emoji avant la coupure
        compte pour un et reste entier ; une moitié de paire isolée compte pour un et reste telle quelle."""
        emoji = "\U0001F600"
        cases = {"a" * 89 + emoji + "b" * 10: "a" * 89 + emoji,
                 emoji + "a" * 88 + "\ud800" + "b": emoji + "a" * 88 + "\ud800",
                 "court": "court"}

        def snapshot(name, text):
            return Snapshot(name, "2026-09-11", {}, (SnapshotEntry("q0", "alice", "?", "answered", ("d",), text, 1),))

        for text, head in cases.items():
            with self.subTest(text=text):
                comparison = compare_snapshots(snapshot("a", text), snapshot("b", text + "c"))
                self.assertIn(f"  avant : answered [d] « {head} »", comparison_to_text(comparison).split("\n"))

    def test_the_status_and_the_manifest_sort_the_splitter_keys(self):
        manifest = IndexManifest("d5276d0355c9", "hashing-256-stem6", 256, "0" * 64, SPLITTER, 9, 15, "2026-09-11")
        report = StatusReport(manifest, 9, "0" * 64, SPLITTER, "hashing-256-stem6", 256, None, "v1")
        described = "{include_title: true, max_chars: 800, overlap_chars: 120, type: paragraph}"
        self.assertIn(f"· découpage {described}", status_to_text(report))
        self.assertIn(f"Découpage  : {described}", status_to_text(report))
        sorted_keys = ["include_title", "max_chars", "overlap_chars", "type"]
        self.assertEqual(list(manifest_to_dict(manifest)["splitter"]), sorted_keys)
        self.assertEqual(list(status_to_dict(report)["splitter"]), sorted_keys)

    def test_the_verbose_answer(self):
        # Seuls les guillemets, l'antislash et les caractères de contrôle U+0000 à U+001F sont échappés (json.dumps) :
        # DEL, espaces insécables, emoji et séparateur de ligne U+2028 restent tels quels.
        raw = 'Deux\u00a0jours\u202f! \U0001F600\u2028\x1b\x7f "q" \\ \n\t'
        trace = AnswerTrace("d5276d0355c9", "hashing-256-stem6", "extractive", "v1",
                            (("teletravail#0", 0.527613), ("conges#0", 0.0)), 0.6, 1, ('Deux jours.\n[1]', raw))
        text = answer_to_text(Answer("?", AnswerStatus.ANSWERED, "Deux jours. [1]", (), trace), verbose=True)
        self.assertIn("seuil=0.6\n", text)
        self.assertIn("  retrouvé conges#0 score=0.0\n", text)
        self.assertIn("  retrouvé teletravail#0 score=0.5276\n", text)   # quatre décimales
        self.assertIn('  sortie brute 1 : "Deux jours.\\n[1]"\n', text)
        last_line = text.split("\n")[-1]   # pas splitlines(), qui couperait aussi en U+2028
        self.assertEqual(last_line, '  sortie brute 2 : "Deux\u00a0jours\u202f! \U0001F600\u2028\\u001b\x7f \\"q\\" \\\\ \\n\\t"')

    def test_the_json_of_an_answer_is_written_as_frozen(self):
        # Ce qu'affiche ask --json, la sortie de main : json.dumps(…, ensure_ascii=False, indent=2). Le cas d'usage
        # rend cette réponse (build remplacé) : ni service IA, ni index.
        root = logging.getLogger()   # main() configure le journal : on rend l'état d'avant après le test
        self.addCleanup(setattr, root, "handlers", root.handlers[:])
        self.addCleanup(root.setLevel, root.level)
        trace = AnswerTrace("id", "hashing", "extractive", "v1+abc", (("a#0", 1.0), ("a#1", 0.25)), 0.4, 1)
        answer = Answer("Combien ?", AnswerStatus.ANSWERED, 'Deux jours\xa0: « oui » "x"\nfin \U0001f600',
                        (Source(1, "a", "Congés", "a#0"),), trace)
        container = mock.Mock()
        container.ask_question.execute.return_value = answer
        with mock.patch("assistant.__main__.build", return_value=container):
            code, out, err = run("ask", "Combien ?", "--json", "--config", str(ROOT / "config" / "app.toml"))
        self.assertEqual((code, out), (0, ANSWER_JSON + "\n"), err)


if __name__ == "__main__":
    unittest.main()
