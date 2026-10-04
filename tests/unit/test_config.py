"""Configuration de l'application : une erreur de saisie se voit au chargement, avec le fichier et la clé."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant.composition import AppConfig, ConfigError
from assistant.infrastructure.splitter import ParagraphSplitter

ROOT = Path(__file__).resolve().parents[2]
VALID = """
[ai_service]
base_url = "http://127.0.0.1:8100"
embedding_model = "hashing"
generation_model = "extractive"
[corpus]
directory = "corpus/solveo"
[index]
path = "data/index.json"
[retrieval.min_score]
default = 0.40
hashing = 0.15
[decorators]
validate_output = true
"""


def _ai(line: str) -> str:
    return VALID.replace('generation_model = "extractive"\n', f'generation_model = "extractive"\n{line}\n')


def _retrieval(line: str) -> str:
    return VALID.replace("[retrieval.min_score]", f"[retrieval]\n{line}\n[retrieval.min_score]")


def _decorators(line: str) -> str:
    return VALID.replace("validate_output = true", f"validate_output = true\n{line}")


# Le message suit le nom du fichier.
MESSAGES = (
    (VALID.replace('embedding_model = "hashing"\n', ""), " [ai_service] : clé « embedding_model » obligatoire"),
    (VALID.replace('[corpus]\ndirectory = "corpus/solveo"\n', ""), " : section [corpus] absente"),
    (_ai("timeout_seconds = 0"), " [ai_service] : « timeout_seconds » = 0, doit être compris entre 1 et 86400"),
    (_ai("timeout_seconds = 86401"), " [ai_service] : « timeout_seconds » = 86401, doit être compris entre 1 et 86400"),
    (_retrieval("top_k = 0"), " [retrieval] : « top_k » = 0, doit valoir au moins 1"),
    (_retrieval("top_k = true"), " [retrieval] : « top_k » doit être un entier, pas true"),
    (_retrieval("topk = 8"), " [retrieval] : clé(s) inconnue(s) [topk] (connues : min_score, top_k)"),
    (VALID.replace("hashing = 0.15", "hashing = 1.5"),
     " [retrieval.min_score] : « hashing » = 1.5, doit être compris entre -1 et 1"),
    (VALID.replace("hashing = 0.15", "hashing = -1.5"),
     " [retrieval.min_score] : « hashing » = -1.5, doit être compris entre -1 et 1"),
    (VALID.replace("[retrieval.min_score]\ndefault = 0.40\nhashing = 0.15", "[retrieval]\nmin_score = 0.4"),
     " : section [retrieval.min_score] mal formée"),
    (VALID + "[generation]\ntemperature = 2.5\n", " [generation] : « temperature » = 2.5, doit être compris entre 0 et 2"),
    (VALID + '[generation]\ntemperature = "chaud"\n', ' [generation] : « temperature » doit être un nombre, pas "chaud"'),
    (VALID + '[generation]\ntemperature = "chaud\\u00a0"\n',   # espace insécable telle quelle
     ' [generation] : « temperature » doit être un nombre, pas "chaud\xa0"'),
    (VALID + "[generation]\nmax_tokens = 0\n", " [generation] : « max_tokens » = 0, doit valoir au moins 1"),
    (VALID + "[generation]\nmax_attempts = 0\n", " [generation] : « max_attempts » = 0, doit valoir au moins 1"),
    (VALID + '[generation]\nseed = "42"\n', ' [generation] : « seed » doit être un entier, pas "42"'),
    (VALID + "[generation]\nprompt = 3\n", " [generation] : « prompt » doit être une chaîne, pas 3"),
    (VALID.replace("validate_output = true", 'validate_output = "oui"'),
     ' [decorators] : « validate_output » doit être un booléen, pas "oui"'),
    (_decorators("max_output_chars = 0"), " [decorators] : « max_output_chars » = 0, doit valoir au moins 1"),
    (_decorators("retries = -1"), " [decorators] : « retries » = -1, doit valoir au moins 0"),
    (VALID + "[splitter]\nmax_chars = 99\n", " [splitter] : « max_chars » = 99, doit valoir au moins 100"),
    (VALID + "[splitter]\noverlap_chars = 400\n",
     " [splitter] : « overlap_chars » = 400, doit être compris entre 0 et 399"),
    (VALID + "[splitter]\nmax_chars = 200\n",
     " [splitter] : « overlap_chars » = 120 (valeur par défaut), doit être compris entre 0 et 99"),
    (VALID + '[splitter]\ninclude_title = "désactivé"\n',
     ' [splitter] : « include_title » doit être un booléen, pas "désactivé"'),
    (VALID + '[users.alice]\ngroup = ["rh"]\n', " [users.alice] : clé(s) inconnue(s) [group] (connues : groups)"),
    (VALID + '[users.alice]\ngroups = [1, "rh"]\n',
     ' [users.alice] : « groups » doit être une liste de chaînes, pas [1,"rh"]'),
    (VALID + '[users.alice]\ngroups = "rh"\n', ' [users.alice] : « groups » doit être une liste de chaînes, pas "rh"'),
    (VALID + '[users.alice]\ngroups = [["rh"]]\n',
     ' [users.alice] : « groups » doit être une liste de chaînes, pas [["rh"]]'),
    (VALID + '[users]\nalice = "tous"\n', " : section [users.alice] mal formée"),
)

# Des entiers démesurés. Plus de 4300 chiffres : refusé à la lecture (int() refuserait de le convertir, en anglais et
# sans dire où) ; moins, à une clé réelle, l'infini, hors bornes (float() levait OverflowError, une trace).
HUGE = "1" + "0" * 5000
TOO_LONG = " : TOML invalide — nombre entier de plus de 4300 chiffres"
HUGE_INTEGERS = (
    (_retrieval(f"top_k = {HUGE}"), TOO_LONG),
    (VALID + f"[generation]\nseed = -{HUGE}\n", TOO_LONG),
    (_retrieval("top_k = +1_" + "0" * 5000), TOO_LONG),
    (VALID + f'[users.alice]\ngroups = [{HUGE}, "rh"]\n', TOO_LONG),
    (_ai("timeout_seconds = 1" + "0" * 400),
     " [ai_service] : « timeout_seconds » = 1" + "0" * 400 + ", doit être compris entre 1 et 86400"),
    (_ai(f"timeout_seconds = {HUGE}.5"), " [ai_service] : « timeout_seconds » = Infinity, doit être compris entre 1 et 86400"),
)


class AppConfigTest(unittest.TestCase):
    def load(self, text: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.toml"
            path.write_text(text, encoding="utf-8")
            return AppConfig.load(path)

    def assert_refused(self, text: str, *fragments: str) -> None:
        with self.assertRaises(ConfigError) as caught:
            self.load(text)
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_the_shipped_configurations_load(self):
        for name in ("app.toml", "app-ollama.toml"):
            with self.subTest(name):
                self.assertTrue(AppConfig.load(ROOT / "config" / name).users)

    def test_errors_name_the_key(self):
        cases = {
            "embedding_model": VALID.replace('embedding_model = "hashing"\n', ""),
            "validate_ouput": VALID.replace("validate_output = true", "validate_ouput = true"),
            "validate_output": VALID.replace("validate_output = true", 'validate_output = "oui"'),
            "max_attempts": VALID + "[generation]\nmax_attempts = 0\n",
            "top_k": VALID.replace("[retrieval.min_score]", "[retrieval]\ntop_k = 0\n[retrieval.min_score]"),
            "topk": VALID.replace("[retrieval.min_score]", "[retrieval]\ntopk = 8\n[retrieval.min_score]"),
            "max_char": VALID + "[splitter]\nmax_char = 300\n",
            "overlap_chars": VALID + "[splitter]\noverlap_chars = \"120\"\n",
            "timeout_seconds": VALID.replace('generation_model = "extractive"\n',
                                             'generation_model = "extractive"\ntimeout_seconds = 0\n'),
            "retreival": VALID + "[retreival]\ntop_k = 8\n",
        }
        for key, text in cases.items():
            with self.subTest(key):
                self.assert_refused(text, key, "app.toml")

    def test_each_error_says_where_and_what_is_wrong(self):
        for text, expected in MESSAGES:
            with self.subTest(expected):
                self.assert_refused(text, f"app.toml{expected}")

    def test_huge_integers_are_refused_at_load_with_a_message(self):
        for number, (text, expected) in enumerate(HUGE_INTEGERS):
            with self.subTest(number):   # l'indice : quatre cas ont le même message
                with self.assertRaises(ConfigError) as caught:
                    self.load(text)
                self.assertTrue(str(caught.exception).endswith(f"app.toml{expected}"), str(caught.exception)[:200])

    def test_a_huge_integer_is_refused_only_where_toml_reads_it(self):
        """Les mêmes chiffres dans une chaîne restent tels quels. En hexadécimal, en octal ou en binaire, tomllib
        convertit l'entier sans limite : refusé lui aussi au-delà de 4300 chiffres décimaux, écrit en deçà."""
        self.assertEqual(self.load(VALID + f'[generation]\nprompt = "{HUGE}"\n').prompt_name, HUGE)
        power = 10 ** 4300   # « 1 » suivi de 4300 zéros : 4301 chiffres
        for written in (hex(power), oct(power), bin(power)):
            with self.subTest(written[:2]):
                self.assert_refused(_retrieval(f"top_k = {written}"), f"app.toml{TOO_LONG}")
        self.assert_refused(VALID + f'[users.alice]\ngroups = [{hex(power)}, "rh"]\n', f"app.toml{TOO_LONG}")   # dans une liste
        self.assert_refused(_ai(f"timeout_seconds = {hex(power - 1)}"),
                            f"app.toml [ai_service] : « timeout_seconds » = {'9' * 4300}, doit être compris entre 1 et 86400")

    def test_an_absurd_timeout_is_refused_at_load(self):
        # Sinon, dès 3e6 s, le socket refuse le délai par une trace (OverflowError), sans fichier ni clé.
        for written, shown in (("3000000", "3000000"), ("1e300", "1e+300")):   # valeur lue, réécrite en JSON
            with self.subTest(written):
                self.assert_refused(_ai(f"timeout_seconds = {written}"),
                                    f"app.toml [ai_service] : « timeout_seconds » = {shown}, doit être compris entre 1 et 86400")

    def test_nan_is_out_of_bounds(self):
        # TOML connaît nan (JSON non) : toute comparaison avec nan est fausse, un seuil nan refuserait tout.
        for text, expected in ((VALID.replace("hashing = 0.15", "hashing = nan"),
                                "[retrieval.min_score] : « hashing » = NaN, doit être compris entre -1 et 1"),
                               (_ai("timeout_seconds = nan"),
                                "[ai_service] : « timeout_seconds » = NaN, doit être compris entre 1 et 86400")):
            with self.subTest(expected):
                self.assert_refused(text, expected)

    def test_a_toml_date_is_shown_as_text(self):
        # TOML connaît les dates (JSON non) : json.dumps ne sait pas les écrire, il lui faut default=str.
        self.assert_refused(_ai("timeout_seconds = 1979-05-27"),
                            'app.toml [ai_service] : « timeout_seconds » doit être un nombre, pas "1979-05-27"')

    def test_the_splitter_bounds_are_those_of_the_splitter(self):
        # (max_chars, overlap_chars, accepté) ; None : clé absente, donc valeur par défaut.
        cases = ((100, 49, True), (100, 50, False), (99, 0, False), (801, 399, True), (801, 400, False),
                 (None, 399, True), (None, 400, False), (242, None, True), (241, None, False))
        for max_chars, overlap_chars, accepted in cases:
            settings = {key: value for key, value in (("max_chars", max_chars), ("overlap_chars", overlap_chars))
                        if value is not None}
            text = VALID + "[splitter]\n" + "".join(f"{key} = {value}\n" for key, value in settings.items())
            with self.subTest(**settings):
                if accepted:   # le découpeur que construira build() est celui des valeurs écrites
                    self.assertEqual(ParagraphSplitter(**self.load(text).splitter).describe(),
                                     ParagraphSplitter(**settings).describe())
                else:          # refusé par le découpeur, donc dès le chargement
                    self.assertRaises(ValueError, ParagraphSplitter, **settings)
                    self.assertRaises(ConfigError, self.load, text)

    def test_a_malformed_file_is_a_config_error_that_names_it(self):
        # Clé en double ou « \ud800 » isolé : tomllib les refuse.
        cases = {
            "syntaxe": VALID + "[corpus\n",
            "clé en double": VALID.replace("hashing = 0.15", "hashing = 0.15\nhashing = 0.2"),
            "surrogate isolé": VALID.replace('"corpus/solveo"', '"corpus/solveo\\ud800"'),
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.assert_refused(text, "app.toml : TOML invalide")

    def test_a_file_that_is_not_utf8_names_the_file_and_the_byte(self):
        """L'erreur était en anglais, sans le fichier. La position se compte après la marque d'ordre des octets ;
        avec elle, un fichier en UTF-8 se lit."""
        text = VALID.replace("corpus/solveo", "corpus/solvéo")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.toml"
            for prefix in (b"", b"\xef\xbb\xbf"):
                with self.subTest(prefix=prefix):
                    path.write_bytes(prefix + text.encode("latin-1"))
                    with self.assertRaises(ConfigError) as caught:
                        AppConfig.load(path)
                    self.assertEqual(str(caught.exception),
                                     f"{path} : pas en UTF-8 (octet 0xe9 à la position {text.index('é')})")
            path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
            self.assertEqual(AppConfig.load(path).corpus_dir.name, "solvéo")

    def test_a_deeply_nested_file_is_a_config_error_not_a_crash(self):
        # tomllib lève RecursionError (pas une TOMLDecodeError) : c'était une trace d'erreur.
        self.assert_refused(VALID + "[generation]\nseed = " + "[" * 100_000 + "]" * 100_000 + "\n",
                            "app.toml : TOML invalide — trop de niveaux d'imbrication")

    def test_keys_that_differ_only_by_case_are_two_keys(self):
        # « Alice » et « alice » sont deux utilisateurs, pas une clé en double.
        self.assertEqual(sorted(self.load(VALID + "[users.alice]\n[users.Alice]\n").users), ["Alice", "alice"])

    def test_a_user_without_groups_is_in_the_public_group_only(self):
        self.assertEqual(self.load(VALID + "[users.alice]\n").user("alice").groups, frozenset({"tous"}))

    def test_the_ai_service_variable_does_not_skip_the_check_of_base_url(self):
        with mock.patch.dict(os.environ, {"AI_SERVICE_URL": "http://machine-gpu:8100"}):
            self.assertEqual(self.load(VALID).ai_base_url, "http://machine-gpu:8100")
            self.assert_refused(VALID.replace('"http://127.0.0.1:8100"', "8100"),
                                "app.toml [ai_service] : « base_url » doit être une chaîne, pas 8100")
            self.assert_refused(VALID.replace('base_url = "http://127.0.0.1:8100"\n', ""),
                                "app.toml [ai_service] : clé « base_url » obligatoire")

    def test_a_leading_underscore_is_not_a_comment_in_toml(self):
        # TOML a ses commentaires (#) : la règle « _ » des jeux de questions n'existe que parce que JSON n'en a pas.
        self.assert_refused(_decorators('_note = "x"'), "app.toml [decorators] : clé(s) inconnue(s) [_note]")

    def test_threshold_per_model_and_uncalibrated_fallback(self):
        config = self.load(VALID)
        self.assertEqual(config.min_score_for("hashing"), 0.15)
        self.assertTrue(config.has_threshold_for("hashing"))
        self.assertEqual(config.min_score_for("nomic"), 0.40)        # repli sur « default »…
        self.assertFalse(config.has_threshold_for("nomic"))          # … signalé (ligne de commande, banc)


if __name__ == "__main__":
    unittest.main()
