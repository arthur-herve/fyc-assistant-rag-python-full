"""Banc d'essai et expériences : ce qui est vérifié avant tout travail (sans service IA, sans indexation)."""

import ast
import contextlib
import http.server
import io
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
import urllib.request
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from assistant.application.errors import AIServiceError, PromptNotFoundError
from assistant.application.ports import Snapshot, SnapshotEntry
from assistant.composition import PROJECT_ROOT, PROMPTS_DIR, AppConfig, UnknownUserError
from assistant.interface import benchmark
from assistant.interface.benchmark import (
    EvalQuestion, check_ai_service, check_served_model, create_new_dir, fixed_min_score, limit_questions,
    load_questions, results_dir, run_benchmark,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "tools" / "experiences"
sys.path.insert(0, str(EXPERIMENTS))

import _commun  # noqa: E402
import cace_decoupage  # noqa: E402
import prompt_v2  # noqa: E402
import stabilite  # noqa: E402

KNOWN_KEYS = "answerable, expected_documents, expected_keywords, forbidden_documents, id, question, user"
# Le prompt « nexiste » : son nom, le dossier et les prompts livrés.
NO_PROMPT = f"prompt introuvable : nexiste dans {PROMPTS_DIR} (connus : answer, answer-v2)"
UNREACHABLE = "service IA injoignable (http://127.0.0.1:9). Lancez-le d'abord : python -m ai_service"
NOT_TEXT = "chaîne qui n'est pas du texte : surrogate UTF-16 isolé (\\ud800 à \\udfff sans sa paire)"
TOO_LONG = "nombre entier de plus de 4300 chiffres"

# Jeux de questions mal formés et ce que dit le message, après « <fichier> : jeu de questions mal formé ».
MALFORMED = [
    ('[1, 2]', "un objet JSON avec une liste « questions » est attendu"),
    ('{"questions": {}}', "un objet JSON avec une liste « questions » est attendu"),
    ('{"questions": [1]}', "question n° 1 : un objet JSON est attendu"),
    ('{"questions": [{"id": 1, "question": "?"}]}', "question n° 1 : champ « id » manquant ou non textuel"),
    ('{"questions": [{"id": "a", "question": "?"}, {"id": "b"}]}',
     "question n° 2 : champ « question » manquant ou non textuel"),
    ('{"questions": [{"id": "a", "question": 12}]}', "question n° 1 : champ « question » manquant ou non textuel"),
    ('{"questions": [{"id": "a", "question": "?", "user": 3}]}', "question n° 1 : « user » doit être un texte"),
    ('{"questions": [{"id": "a", "question": "?", "answerable": "false"}]}',
     "question n° 1 : « answerable » doit valoir true ou false"),
    ('{"questions": [{"id": "a", "question": "?", "expected_documents": "teletravail"}]}',
     "question n° 1 : « expected_documents » doit être une liste de textes"),
    ('{"questions": [{"id": "a", "question": "?", "expected_keywords": [1]}]}',
     "question n° 1 : « expected_keywords » doit être une liste de textes"),
    ('{"questions": [{"id": "a", "question": "?", "expected_document": ["x"]}]}',
     f"question n° 1 : clé(s) inconnue(s) [expected_document] (connues : {KNOWN_KEYS})"),
    ('{"questions": [{"id": "a", "id": "b", "question": "?"}]}', "clé « id » en double"),
    ('{"questions": [{"id": {"x": 1, "x": 2}, "id": "a", "question": "?"}]}', "clé « x » en double"),   # l'objet intérieur d'abord
    ('{"questions": [{"id": "a", "question": "?"}, {"id": "a", "question": "?"}]}',
     "question n° 2 : identifiant « a » déjà utilisé"),
    ('{"x": ' + "[" * 900 + "]" * 900 + ', "questions": []}', "JSON trop imbriqué : plus de 900 niveaux"),
    ("[" * 100_000 + "]" * 100_000, "JSON trop imbriqué : plus de 900 niveaux"),
    ('{"x": NaN, "questions": [{"id": "a", "question": "?"}]}', "« NaN » n'est pas du JSON"),   # json.loads le lirait
    # « \ud800 » isolé : du JSON que json.loads lit, mais pas du texte. Partout, commentaire compris. Les chaînes sont
    # vérifiées après json.loads : ce qu'il refuse en lisant (clé en double, NaN, entier trop long) est dit avant.
    ('{"questions": [{"id": "a", "question": "\\ud800"}]}', NOT_TEXT),
    ('{"questions": [{"id": "a", "question": "?", "user": "\\udc00"}]}', NOT_TEXT),
    ('{"questions": [{"id": "\\ud800", "question": "?"}]}', NOT_TEXT),
    ('{"questions": [{"id": "a", "question": "?", "expected_documents": ["\\ud83d"]}]}', NOT_TEXT),
    ('{"questions": [{"id": "a", "question": "?", "\\ud800": 1}]}', NOT_TEXT),
    ('{"questions": [{"id": "a", "question": "?", "\\ud800": 1, "\\ud800": 2}]}', "clé « \ud800 » en double"),
    ('{"questions": [{"id": "a", "question": "?", "_note": "\\ud800"}]}', NOT_TEXT),
    ('{"questions": [{"id": "\\ud800", "question": "?", "_x": ' + "[" * 70 + "]" * 70 + "}]}", NOT_TEXT),
    ('{"questions": [{"_x": ' + "[" * 900 + "]" * 900 + ', "id": "\\ud800", "question": "?"}]}',
     "JSON trop imbriqué : plus de 900 niveaux"),   # refusé avant même json.loads
    ('{"questions": [{"id": "\\ud800", "question": "?", "_x": NaN}]}', "« NaN » n'est pas du JSON"),
    # Plus de 4300 chiffres : int() refuserait de le convertir, avec un message en anglais.
    ('{"questions": [{"id": "a", "question": "?", "_n": ' + "1" * 4301 + "}]}", TOO_LONG),
    ('{"questions": [{"id": -' + "1" * 4301 + ', "question": "?"}]}', TOO_LONG),
    ('{"questions": [{"id": "\\ud800", "question": "?", "_n": ' + "1" * 4301 + "}]}", TOO_LONG),
]


@contextlib.contextmanager
def connection_refused(*ports: int):
    """Les connexions à 127.0.0.1 sur ces ports refusées aussitôt, comme quand rien n'y écoute : sous Windows, un vrai
    refus coûte 2 s (la demande de connexion y est réessayée). Le refus part de socket.create_connection, d'où urllib
    le reçoit ; les autres connexions passent. Le vrai refus, un seul pour les tests de l'application : test_cli.py
    (test_an_unreachable_ai_service_is_an_error_before_any_folder)."""
    connect = socket.create_connection

    def refusing(address, *args, **kwargs):
        if address[0] == "127.0.0.1" and address[1] in ports:
            raise ConnectionRefusedError(f"connexion refusée (simulée) : 127.0.0.1:{address[1]}")
        return connect(address, *args, **kwargs)

    with mock.patch("socket.create_connection", refusing):
        yield


class QuestionSetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, content: str) -> Path:
        path = Path(self.tmp.name) / "questions.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_the_shipped_question_sets_load_with_real_booleans(self):
        for path in sorted((ROOT / "eval").glob("questions*.json")):
            with self.subTest(path=path.name):
                questions = load_questions(path)
                self.assertTrue(all(isinstance(q.answerable, bool) for q in questions))
                self.assertTrue(any(not q.answerable for q in questions))

    def test_malformed_question_sets_name_the_file_and_the_field(self):
        for content, detail in MALFORMED:
            with self.subTest(content=content[:80]):
                path = self.write(content)
                with self.assertRaises(ValueError) as caught:
                    load_questions(path)
                self.assertEqual(str(caught.exception), f"{path} : jeu de questions mal formé ({detail})")

    def test_invalid_json_names_the_file(self):
        path = self.write("{pas json")
        with self.assertRaises(ValueError) as caught:
            load_questions(path)
        self.assertTrue(str(caught.exception).startswith(f"{path} : jeu de questions mal formé (JSON invalide : "),
                        caught.exception)

    def test_a_question_set_that_is_not_utf8_names_the_byte_and_its_position(self):
        """L'erreur était en anglais (« 'utf-8' codec can't decode… »). Avec ou sans marque d'ordre des octets, la
        position se compte après elle."""
        content = '{"questions": [{"id": "a", "question": "Congés ?"}]}'
        path = Path(self.tmp.name) / "questions.json"
        for prefix in (b"", b"\xef\xbb\xbf"):
            with self.subTest(prefix=prefix):
                path.write_bytes(prefix + content.encode("latin-1"))
                with self.assertRaises(ValueError) as caught:
                    load_questions(path)
                self.assertEqual(str(caught.exception), f"{path} : jeu de questions mal formé "
                                                        f"(pas en UTF-8 (octet 0xe9 à la position {content.index('é')}))")
        path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
        self.assertEqual([q.question for q in load_questions(path)], ["Congés ?"])

    def test_a_key_that_starts_with_an_underscore_is_a_comment(self):
        # JSON n'a pas de commentaires : « _note » se lit sans effet ; les autres clés inconnues restent
        # refusées.
        path = self.write('{"questions": [{"id": "a", "_note": "à revoir", "question": "?", "_x": [1]}]}')
        self.assertEqual([q.id for q in load_questions(path)], ["a"])
        path = self.write('{"questions": [{"id": "a", "note": "à revoir", "question": "?"}]}')
        with self.assertRaisesRegex(ValueError, r"clé\(s\) inconnue\(s\) \[note\]"):
            load_questions(path)

    def test_a_complete_surrogate_pair_and_an_integer_of_4300_digits_are_read(self):
        # Les bornes de ce que refuse MALFORMED : un emoji écrit en paire (« \ud83d\ude00 ») est du texte, et
        # 4300 chiffres, signe non compté, int() les convertit.
        path = self.write('{"questions": [{"id": "a", "question": "\\ud83d\\ude00 ?", "_n": -' + "9" * 4300 + "}]}")
        self.assertEqual([q.question for q in load_questions(path)], ["\U0001F600 ?"])

    def test_an_empty_question_set_is_refused(self):
        path = self.write('{"questions": []}')
        with self.assertRaisesRegex(ValueError, "^aucune question dans "):
            load_questions(path)

    def test_limit_keeps_the_first_questions_and_refuses_zero_or_less(self):
        questions = [EvalQuestion(str(i), "?", "alice", True, (), (), ()) for i in range(3)]
        self.assertEqual(limit_questions(questions, None), questions)
        self.assertEqual([q.id for q in limit_questions(questions, 2)], ["0", "1"])
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError) as caught:
                limit_questions(questions, limit)
            self.assertEqual(str(caught.exception), f"--limit doit valoir au moins 1, pas {limit}")


class BenchmarkOptionsTest(unittest.TestCase):
    """Une option fausse est refusée avant tout travail : ni dossier de résultats, ni indexation."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "jamais"
        # Aucun service IA ne répond ici : une option vérifiée trop tard échouerait autrement.
        self.config = replace(AppConfig.load(ROOT / "config" / "app.toml"), ai_base_url="http://127.0.0.1:9")
        self.questions = [EvalQuestion("tt", "Combien de jours de télétravail ?", "alice", True,
                                       ("teletravail",), (), ())]

    def refused(self, error, **options):
        arguments = {"runs": 1, **options}
        config = arguments.pop("config", self.config)
        questions = arguments.pop("questions", self.questions)
        with self.assertRaises(error) as caught:
            run_benchmark(config, ["hashing"], ["extractive"], questions, out_dir=self.out, log=lambda *_: None,
                          **arguments)
        self.assertFalse(self.out.exists())
        return str(caught.exception)

    def test_runs_must_be_at_least_one(self):
        self.assertEqual(self.refused(ValueError, runs=0), "--runs doit valoir au moins 1, pas 0")
        self.assertEqual(self.refused(ValueError, runs=-2), "--runs doit valoir au moins 1, pas -2")

    def test_min_score_must_be_a_threshold_of_the_configuration(self):
        for mode in ("abc", "nan", "inf", "5", "-1.5"):
            with self.subTest(mode=mode):
                self.assertEqual(self.refused(ValueError, min_score_mode=mode),
                                 f"--min-score attend config, auto ou un nombre de [-1, 1] (ex. 0.6), pas « {mode} »")

    def test_splitter_and_prompt_are_checked_before_indexing(self):
        self.assertEqual(self.refused(ValueError, splitter_overrides={"max_chars": 50}),
                         "max_chars doit valoir au moins 100")
        self.assertEqual(self.refused(ValueError, splitter_overrides={"overlap_chars": 900}),
                         "overlap_chars doit être compris entre 0 et max_chars / 2")
        self.assertEqual(self.refused(PromptNotFoundError, config=replace(self.config, prompt_name="nexiste")),
                         NO_PROMPT)

    def test_an_empty_prompt_is_the_configured_one_like_in_the_other_commands(self):
        """--prompt "" vaut le prompt de la configuration, comme pour ask, status et snapshot record (build) : les
        options passent, et le banc s'arrête au service IA, absent ici. Il disait « prompt introuvable :  dans … »."""
        with connection_refused(9):
            self.assertEqual(self.refused(AIServiceError, prompt_name=""), UNREACHABLE)

    def test_users_of_both_question_sets_are_checked_before_indexing(self):
        intruder = [replace(self.questions[0], user="mallory")]
        self.assertIn("mallory", self.refused(UnknownUserError, questions=intruder))
        self.assertIn("mallory", self.refused(UnknownUserError, validation=intruder))

    def test_fixed_min_score(self):
        self.assertIsNone(fixed_min_score("config"))
        self.assertIsNone(fixed_min_score("auto"))
        self.assertEqual([fixed_min_score(v) for v in ("0.6", "-1", "1")], [0.6, -1.0, 1.0])


class ExperimentsTest(unittest.TestCase):
    """Une expérience refuse une option fausse comme `python -m assistant` : « Erreur : … », code 1, pas
    de trace, et avant le moindre travail (dossier, index, instantané « avant »)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.out = self.tmp / "exp"

    def run_script(self, script, *options: str, config: Path | None = ROOT / "config" / "app.toml") -> str:
        """Le message de sortie ; ce qui s'est écrit sur la sortie d'erreur avant lui reste dans self.stderr.
        Sans `config` (None), pas de --config."""
        given = ("--config", str(config)) if config is not None else ()
        argv = [f"{script.__name__}.py", *given, "--out", str(self.out), *options]
        self.stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(self.stderr), \
                self.assertRaises(SystemExit) as caught:
            _commun.run(script.main)
        self.assertFalse(self.out.exists())
        return str(caught.exception.code)

    def without_threshold(self, ai_url: str | None = None) -> Path:
        """La configuration livrée, sans seuil pour « hashing » (le seuil `default` s'applique)."""
        text = (ROOT / "config" / "app.toml").read_text(encoding="utf-8").replace("hashing = 0.15", "")
        if ai_url is not None:
            text = text.replace("http://127.0.0.1:8100", ai_url)
        config = self.tmp / "app.toml"
        config.write_text(text, encoding="utf-8")
        self.assertFalse(AppConfig.load(config).has_threshold_for("hashing"))
        return config

    def test_an_invalid_splitter_is_refused_before_the_first_index(self):
        self.assertEqual(self.run_script(cace_decoupage, "--max-chars", "50"),
                         "Erreur : max_chars doit valoir au moins 100")

    def test_an_unknown_prompt_is_refused_before_the_first_index(self):
        # Son nom, le dossier et les prompts connus : plus de « fichier ou dossier inaccessible ».
        self.assertEqual(self.run_script(prompt_v2, "--other", "nexiste"), f"Erreur : {NO_PROMPT}")

    def test_the_main_configuration_is_checked_before_the_first_index(self):
        """Pas seulement ce que l'expérience change : le prompt de la configuration elle-même."""
        config = self.tmp / "app.toml"
        config.write_text((ROOT / "config" / "app.toml").read_text(encoding="utf-8")
                          .replace('prompt = "answer"', 'prompt = "nexiste"'), encoding="utf-8")
        self.assertEqual(AppConfig.load(config).prompt_name, "nexiste")
        self.assertEqual(self.run_script(stabilite, config=config), f"Erreur : {NO_PROMPT}")

    def test_limit_runs_and_users_are_checked_before_any_work(self):
        self.assertEqual(self.run_script(stabilite, "--limit", "0"), "Erreur : --limit doit valoir au moins 1, pas 0")
        self.assertEqual(self.run_script(stabilite, "--runs", "1"),
                         "Erreur : --runs doit valoir au moins 2 : il faut deux passages pour mesurer une dérive")
        questions = self.tmp / "questions.json"
        questions.write_text('{"questions": [{"id": "q", "user": "mallory", "question": "?"}]}', encoding="utf-8")
        self.assertIn("utilisateur inconnu : mallory", self.run_script(stabilite, "--questions", str(questions)))

    def test_a_wrong_option_is_refused_after_the_configuration_and_before_any_work(self):
        """Les options sont vérifiées à un seul endroit, la construction de l'expérience : après la configuration
        (absente, c'est elle qui est dite), avant tout travail (ni dossier, ni appel au service IA, ni avertissement
        du seuil `default`). Le faux service IA note chaque requête reçue."""
        requests = []

        class Counting(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Counting)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        config = self.without_threshold(f"http://127.0.0.1:{server.server_address[1]}")
        cases = [
            (stabilite, ("--runs", "1"), "--runs doit valoir au moins 2 : il faut deux passages pour mesurer une dérive"),
            (stabilite, ("--limit", "0"), "--limit doit valoir au moins 1, pas 0"),
            (cace_decoupage, ("--max-chars", "50"), "max_chars doit valoir au moins 100"),
            # 150 : la moitié des 300 caractères par défaut, la première valeur refusée.
            (cace_decoupage, ("--overlap-chars", "150"), "overlap_chars doit être compris entre 0 et max_chars / 2"),
            (prompt_v2, ("--other", "nexiste"), NO_PROMPT),
        ]
        for script, options, message in cases:
            with self.subTest(script=script.__name__, options=options):
                self.assertIn("absente.toml", self.run_script(script, *options, config=self.tmp / "absente.toml"))
                self.assertEqual(self.run_script(script, *options, config=config), f"Erreur : {message}")
                self.assertEqual(self.stderr.getvalue(), "")
        self.assertEqual(requests, [])

    def test_without_config_the_header_names_the_default_files(self):
        """Sans --config ni --questions : le fichier de configuration lu, config/app.toml sous la racine du projet,
        en chemin complet, et le jeu de questions par défaut tel quel (« eval/questions.json »)."""
        args = _commun.parser("test").parse_args(["--out", str(self.out), "--limit", "1"])
        with contextlib.chdir(ROOT), mock.patch.dict(os.environ):
            os.environ.pop("ASSISTANT_CONFIG", None)
            exp = _commun.Experiment("test", args)
        self.assertIn(f"Configuration `{(ROOT / 'config' / 'app.toml').as_posix()}` · 1 questions de "
                      "`eval/questions.json` · corpus `solveo`.", exp.lines)
        self.assertFalse(self.out.exists())

    def test_without_config_assistant_config_is_read(self):
        """Sans --config, la variable ASSISTANT_CONFIG désigne la configuration, comme pour `python -m assistant` :
        les scripts lisaient config/app.toml sans rien dire. L'en-tête donne alors le fichier lu, tel que la
        variable le nomme."""
        config = self.without_threshold("http://127.0.0.1:9")
        args = _commun.parser("test").parse_args(["--out", str(self.out), "--limit", "1"])
        with contextlib.chdir(ROOT), mock.patch.dict(os.environ, {"ASSISTANT_CONFIG": str(config)}):
            os.environ.pop("AI_SERVICE_URL", None)
            exp = _commun.Experiment("test", args)
        self.assertEqual(exp.config.ai_base_url, "http://127.0.0.1:9")
        self.assertIn(f"Configuration `{config.as_posix()}` · 1 questions de `eval/questions.json` · corpus `solveo`.",
                      exp.lines)
        # Absente, elle est dite absente : pas de repli silencieux sur config/app.toml.
        with mock.patch.dict(os.environ, {"ASSISTANT_CONFIG": str(self.tmp / "absente.toml")}):
            message = self.run_script(stabilite, config=None)
        self.assertTrue(message.startswith("Erreur : fichier ou dossier inaccessible — "), message)
        self.assertIn("absente.toml", message)

    def test_the_default_threshold_is_announced_on_the_console_and_in_the_report(self):
        """Le seuil `default` ne s'applique pas en silence (ADR 0004), comme en ligne de commande et au banc ;
        mais pas avant les vérifications : la construction n'en dit rien (start l'annonce, voir test_end_to_end)."""
        args = _commun.parser("test").parse_args(["--config", str(self.without_threshold()), "--out", str(self.out),
                                                  "--limit", "1"])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exp = _commun.Experiment("test", args)
            self.assertEqual(stderr.getvalue(), "")
            exp.warn_if_default_threshold("hashing-512")   # seuil configuré : rien à dire
            exp.warn_if_default_threshold("hashing")
        warning = ("Attention : aucun seuil de pertinence configuré pour « hashing » : valeur `default` 0.4 "
                   "(ADR 0004 : lancer le banc d'essai)")
        self.assertEqual(stderr.getvalue().splitlines(), [warning])
        self.assertIn(f"> {warning}", exp.lines)
        self.assertFalse(self.out.exists())   # rien n'est créé avant la première indexation

    def test_the_default_threshold_is_announced_after_the_checks(self):
        # Une option fausse, ou un service IA injoignable : l'erreur seule, pas d'avertissement avant elle.
        self.assertEqual(self.run_script(cace_decoupage, "--max-chars", "50", config=self.without_threshold()),
                         "Erreur : max_chars doit valoir au moins 100")
        self.assertEqual(self.stderr.getvalue(), "")
        # « Erreur : » et le message du service injoignable, comme toute erreur (il était sans préfixe).
        with connection_refused(9):
            self.assertEqual(self.run_script(stabilite, config=self.without_threshold("http://127.0.0.1:9")),
                             f"Erreur : {UNREACHABLE}")
        self.assertEqual(self.stderr.getvalue(), "")

    def test_nothing_is_indexed_before_start(self):
        # Service IA injoignable, pour ne rien indexer : sans la garde, le dossier serait créé et l'erreur serait la sienne.
        args = _commun.parser("test").parse_args(["--config", str(self.without_threshold("http://127.0.0.1:9")),
                                                  "--out", str(self.out)])
        with self.assertRaisesRegex(RuntimeError, r"^Experiment\.start\(\) d'abord"):
            _commun.Experiment("test", args).index("avant")
        self.assertFalse(self.out.exists())

    def serving(self, models: str, status: int = 200) -> str:
        """L'adresse d'un faux service IA qui répond `models` (texte JSON tel quel), avec ce statut, à toute
        requête GET (/v1/models, /health)."""
        class Models(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = models.encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Models)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def stalling(self, start: bytes, cut: bool) -> str:
        """L'adresse d'un faux service IA qui envoie `start` (ses en-têtes et un début de corps) à toute requête GET,
        puis se tait, connexion ouverte, jusqu'à ce que le client abandonne ; ou coupe la connexion (cut)."""
        class Stalling(http.server.BaseHTTPRequestHandler):
            timeout = 10   # au pire, le faux service abandonne lui-même

            def do_GET(self):
                self.wfile.write(start)
                if not cut:
                    self.rfile.read()   # se tait, jusqu'à ce que le client ferme
                self.close_connection = True

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stalling)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_the_list_of_served_models_must_be_readable(self):
        """Injoignable, ou hors contrat (un alias qui n'est pas un texte, une clé en double) : le même message,
        pas une trace, ni « la dernière valeur l'emporte ». La réponse est
        lue en JSON strict, comme un fichier, même là où rien n'est lu : « nomic » y est servi."""
        served = '{"embedding": [{"alias": "nomic", "x": %s}], "generation": []}'
        for url in ("http://127.0.0.1:9/",
                    self.serving(json.dumps({"embedding": [{"alias": 1}], "generation": []})),
                    self.serving('{"embedding": [{"alias": "a"}], "embedding": [{"alias": "nomic"}], "generation": []}'),
                    self.serving(served % '{"b": 1, "b": 2}'),
                    self.serving(served % ("1" * 4301)),
                    self.serving(served % '"\\ud800"')):
            with self.subTest(url=url), connection_refused(9):
                with self.assertRaises(AIServiceError) as caught:
                    check_served_model(url, "embedding", "nomic")
                self.assertEqual(str(caught.exception),
                                 f"le service IA ne donne pas la liste de ses modèles ({url.rstrip('/')}/v1/models)")
                self.assertFalse(caught.exception.transient)
        check_served_model(self.serving(served % "1"), "embedding", "nomic")   # sans défaut : accepté
        # Les octets, lus comme toute réponse du service IA (docs/contrat-http.md) : une marque d'ordre des octets est
        # acceptée (elle était refusée) ; un octet qui n'est pas de l'UTF-8 (latin-1) rend la réponse hors contrat.
        check_served_model(self.serving("\ufeff" + served % "1"), "embedding", "nomic")
        latin1 = io.BytesIO((served % '"é"').encode("latin-1"))
        with mock.patch("urllib.request.urlopen", return_value=latin1) as urlopen:
            with self.assertRaises(AIServiceError) as caught:
                check_served_model("http://127.0.0.1:9/", "embedding", "nomic")
        urlopen.assert_called_once()   # le faux service a bien répondu
        self.assertEqual(str(caught.exception),
                         "le service IA ne donne pas la liste de ses modèles (http://127.0.0.1:9/v1/models)")

    def test_an_empty_list_of_served_models_says_aucun(self):
        # « aucun » plutôt que « (servis : ) » : un message qui se lit.
        url = self.serving(json.dumps({"embedding": [], "generation": [{"alias": "nomic"}]}))
        with self.assertRaises(ValueError) as caught:
            check_served_model(url, "embedding", "nomic")   # servi, mais pour la génération
        self.assertEqual(str(caught.exception),
                         "le service IA ne sert pas « nomic » comme modèle d'embeddings (servis : aucun)")

    def test_an_unreachable_or_failing_ai_service_is_said_with_its_address(self):
        """Deux messages : injoignable (« Lancez-le d'abord »), ou joignable mais en erreur, avec l'adresse
        interrogée ; pas « Service IA injoignable sur … (<urlopen error …>) »."""
        # Adresse invalide ou autre protocole que HTTP : le même message. Un dossier qui contient un fichier
        # « health » : urllib le lisait (« joignable »).
        (self.tmp / "health").write_text("ok", encoding="utf-8")
        # Le port 99999 : urllib le passe au système, qui le réduit modulo 65536 (34463) ; refus simulé, comme pour 9.
        for base_url in ("http://127.0.0.1:9", "http://127.0.0.1:99999", "http://[::1", "ftp://127.0.0.1:9",
                         "file:///c:/x", self.tmp.as_uri()):
            with self.subTest(base_url=base_url), connection_refused(9, 99999):
                with self.assertRaises(AIServiceError) as caught:
                    check_ai_service(base_url)
                self.assertEqual((str(caught.exception), caught.exception.transient),
                                 (UNREACHABLE.replace("http://127.0.0.1:9", base_url), False))
        url = self.serving('{"error": "en panne"}', status=500)
        with self.assertRaises(AIServiceError) as caught:
            check_ai_service(url + "/")   # une barre finale : la même adresse, sans « // »
        self.assertEqual((str(caught.exception), caught.exception.transient),
                         (f"le service IA répond HTTP 500 sur {url}/health", False))
        check_ai_service(self.serving("ok"))   # joignable : le corps est lu, sans être interprété

    def test_a_health_answer_cut_short_or_silent_is_unreachable(self):
        """Le service envoie ses en-têtes, puis se tait (délai dépassé) ou coupe, au milieu du corps : injoignable,
        d'un 200 comme d'une erreur (le corps est lu en entier) ; ni « joignable » (le banc partait),
        ni « répond HTTP 500 »."""
        ok = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"
        error = b"HTTP/1.1 500 Erreur\r\nContent-Length: 100\r\n\r\n{"
        cases = {
            "200, puis silence": (ok, False),
            "200 coupé": (ok, True),
            "500, puis silence": (error, False),
            "500 coupé": (error, True),
            "500 en morceaux, coupé": (b"HTTP/1.1 500 Erreur\r\nTransfer-Encoding: chunked\r\n\r\n10\r\n{", True),
        }
        urlopen = urllib.request.urlopen
        for case, (start, cut) in cases.items():
            url = self.stalling(start, cut)
            # Le délai de 5 s ramené à 0,2 s : « puis silence » le dépasse.
            with self.subTest(case), mock.patch("urllib.request.urlopen",
                                                lambda address, timeout: urlopen(address, timeout=0.2)):
                with self.assertRaises(AIServiceError) as caught:
                    check_ai_service(url)
                self.assertEqual((str(caught.exception), caught.exception.transient),
                                 (UNREACHABLE.replace("http://127.0.0.1:9", url), False))

    def test_the_mean_length_of_the_answers_counts_characters(self):
        """prompt-v2 : en caractères (len, points de code) : trois emoji comptent pour 3. Les seules réponses données,
        moyenne arrondie à égalité vers le pair ; None sans réponse donnée."""
        def snapshot(*entries):
            return Snapshot("s", "2026-09-12T00:00:00+00:00", {}, tuple(
                SnapshotEntry(f"q{i}", "alice", "?", status, (), text, 1) for i, (status, text) in enumerate(entries)))

        self.assertEqual(prompt_v2.mean_length(snapshot(("answered", "\U0001F600" * 3), ("answered", "a"),
                                                        ("no_relevant_source", "un refus"))), 2)
        self.assertEqual(prompt_v2.mean_length(snapshot(("answered", "ab"), ("answered", "abc"))), 2)   # 2,5 → 2
        self.assertIsNone(prompt_v2.mean_length(snapshot(("no_relevant_source", "un refus"))))

    def test_all_lists_what_the_scripts_import(self):
        """__all__ de _commun couvre tout ce que les scripts voisins importent."""
        imported = set()
        for script in EXPERIMENTS.glob("*.py"):
            for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module == "_commun":
                    imported |= {alias.name for alias in node.names}
        self.assertTrue(imported)
        self.assertLessEqual(imported, set(_commun.__all__))
        self.assertTrue(all(hasattr(_commun, name) for name in _commun.__all__))


class DefaultOutputFolderTest(unittest.TestCase):
    """Sans --out (banc, expériences) : sous la racine du projet, d'où qu'on lance la commande, daté AAAAMMJJ-HHMMSS,
    en chemin complet. Il était pris dans le dossier courant."""

    def test_the_folder_is_dated_under_the_project_root_as_a_full_path(self):
        now = datetime(2026, 10, 1, 16, 41, 52)
        self.assertEqual(results_dir("", now), PROJECT_ROOT / "eval" / "resultats" / "20261001-164152")
        self.assertEqual(results_dir("exp-prompt-v2-", now),
                         PROJECT_ROOT / "eval" / "resultats" / "exp-prompt-v2-20261001-164152")
        self.assertTrue(results_dir("", now).is_absolute())

    def test_the_benchmark_and_the_experiments_use_it_from_another_folder(self):
        # Ni config/, ni eval/ dans le dossier courant : seuls les chemins absolus ci-dessous se lisent.
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            config, questions = str(ROOT / "config" / "app.toml"), str(ROOT / "eval" / "questions.json")
            args = _commun.parser("test").parse_args(["--config", config, "--questions", questions, "--limit", "1"])
            with mock.patch.object(benchmark, "run_benchmark") as run:
                benchmark.main_benchmark(SimpleNamespace(
                    config=config, seed=None, questions=questions, limit=1, validate_with=None, max_chars=None,
                    overlap_chars=None, out=None, embedding=None, generation=None, runs=1, min_score="config",
                    prompt=None))
            for out, name in ((_commun.Experiment("test", args).out, r"exp-test-\d{8}-\d{6}"),
                              (run.call_args.kwargs["out_dir"], r"\d{8}-\d{6}")):
                with self.subTest(out=out):
                    self.assertEqual((Path(tmp) / out).resolve().parent, PROJECT_ROOT / "eval" / "resultats")
                    self.assertRegex(out.name, rf"\A{name}\Z")
                    self.assertFalse((Path(tmp) / out).exists())   # rien n'est créé avant le travail
            self.assertTrue(run.call_args.kwargs["new_out_dir"])   # jamais un dossier qui existe déjà

    def test_a_folder_that_already_exists_gets_a_suffix(self):
        """Daté à la seconde, le dossier par défaut peut déjà exister : deux bancs lancés dans la même seconde
        écrivaient dans le même dossier, et le second écrasait le rapport du premier. Le suivant reçoit « -2 », puis
        « -3 »… ; un fichier de ce nom compte aussi."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "eval" / "resultats" / "20261001-164152"   # dossiers parents absents : créés
            self.assertEqual(create_new_dir(base), base)
            self.assertEqual(create_new_dir(base), base.with_name("20261001-164152-2"))
            base.with_name("20261001-164152-3").write_text("", encoding="utf-8")
            self.assertEqual(create_new_dir(base), base.with_name("20261001-164152-4"))
            self.assertEqual(sorted(p.name for p in base.parent.iterdir() if p.is_dir()),
                             ["20261001-164152", "20261001-164152-2", "20261001-164152-4"])

    def test_a_parent_that_is_not_a_folder_is_an_error(self):
        """eval/resultats existe, mais c'est un fichier : l'erreur remonte (OSError), et rien n'est créé. Le piège :
        sous Windows, mkdir(parents=True) lève alors FileExistsError pour ce parent ; dans la boucle, il passerait
        pour un nom déjà pris, et create_new_dir essaierait « -2 », « -3 »… sans fin."""
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "resultats"
            parent.write_text("", encoding="utf-8")
            with self.assertRaises(OSError):
                create_new_dir(parent / "20261001-164152")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["resultats"])

    def test_folders_created_at_the_same_instant_are_all_different(self):
        """mkdir sans exist_ok échoue si le dossier existe, même créé à l'instant ailleurs : huit fils lancés ensemble
        reçoivent huit dossiers différents, comme deux processus. Vérifier puis créer pouvait en donner deux pareils."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "20261001-164152"
            created, start = [], threading.Barrier(8)

            def create():
                start.wait()
                created.append(create_new_dir(base))

            threads = [threading.Thread(target=create) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(p.name for p in created),
                             sorted(["20261001-164152"] + [f"20261001-164152-{n}" for n in range(2, 9)]))


if __name__ == "__main__":
    unittest.main()
