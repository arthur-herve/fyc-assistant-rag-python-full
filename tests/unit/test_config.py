"""Configuration de l'application : une erreur de saisie se voit au chargement, avec le fichier et la clé."""

import tempfile
import unittest
from pathlib import Path

from assistant.composition import AppConfig, ConfigError

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


class AppConfigTest(unittest.TestCase):
    def load(self, text: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.toml"
            path.write_text(text, encoding="utf-8")
            return AppConfig.load(path)

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
            with self.subTest(key), self.assertRaises(ConfigError) as caught:
                self.load(text)
            self.assertIn(key, str(caught.exception))
            self.assertIn("app.toml", str(caught.exception))

    def test_threshold_per_model_and_uncalibrated_fallback(self):
        config = self.load(VALID)
        self.assertEqual(config.min_score_for("hashing"), 0.15)
        self.assertTrue(config.has_threshold_for("hashing"))
        self.assertEqual(config.min_score_for("nomic"), 0.40)        # repli sur « default »…
        self.assertFalse(config.has_threshold_for("nomic"))          # … signalé (ligne de commande, banc)


if __name__ == "__main__":
    unittest.main()
