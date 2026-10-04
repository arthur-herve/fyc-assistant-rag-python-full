import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from assistant.composition import AppConfig
from assistant.interface import benchmark
from assistant.interface.benchmark import (
    CSV_FIELDS, BenchmarkRow, BenchmarkSummary, EvalQuestion, GenerationSummary, IndexingFailure, RetrievalScore,
    RetrievalSummary, ThresholdValidation, keyword_coverage, run_benchmark, suggest_threshold, summarize,
    validate_threshold,
)

ROOT = Path(__file__).resolve().parents[2]
# Ce que le banc écrit dans synthese.json pour la synthèse de test_synthesis_is_written_as_frozen.
SYNTHESIS = (
    '{\n'
    '  "retrieval": [\n'
    '    {\n'
    '      "embedding": "hashing",\n'
    '      "model_id": "hashing:256@abc",\n'
    '      "dimension": 256,\n'
    '      "chunks": 15,\n'
    '      "index_seconds": 0.3,\n'
    '      "hit@1": 1.0,\n'
    '      "hit@k": 1.0,\n'
    '      "top_k": 4,\n'
    '      "top1_median_answerable": 0.5,\n'
    '      "top1_median_unanswerable": 0.25,\n'
    '      "configured_threshold": 0.4,\n'
    '      "configured_threshold_is_default": true,\n'
    '      "suggested_threshold": 0.375,\n'
    '      "separation_accuracy": 1.0,\n'
    '      "threshold_used": 0.4,\n'
    '      "validation": {\n'
    '        "questions": 2,\n'
    '        "hit@1": 1.0,\n'
    '        "kept_answerable": 1.0,\n'
    '        "correct_refusals": 0.0,\n'
    '        "threshold": 0.4\n'
    '      }\n'
    '    },\n'
    '    {\n'
    '      "embedding": "nomic",\n'
    '      "error": "Service IA : HTTP 502 — « modèle absent »"\n'
    '    }\n'
    '  ],\n'
    '  "generation": [\n'
    '    {\n'
    '      "embedding": "hashing",\n'
    '      "generation": "extractive",\n'
    '      "calls": 4,\n'
    '      "errors": 0,\n'
    '      "answer_rate": 1.0,\n'
    '      "source_hit_rate": 0.5,\n'
    '      "keyword_coverage": null,\n'
    '      "unsourced_rate": 0.0,\n'
    '      "correct_refusal_rate": 1.0,\n'
    '      "forbidden_leaks": 0,\n'
    '      "stability": null,\n'
    '      "mean_attempts": 1.0,\n'
    '      "latency_median_ms": 12.0,\n'
    '      "latency_p90_ms": 15.0,\n'
    '      "refusals_without_generation": 2\n'
    '    }\n'
    '  ]\n'
    '}'
)


class BenchmarkHelpersTest(unittest.TestCase):
    def test_keyword_coverage_ignores_case_and_accents(self):
        self.assertEqual(keyword_coverage("DEUX JOURS, indemnite de 30 euros", ("deux jours", "40 euros", "indemnité")), 2 / 3)
        self.assertIsNone(keyword_coverage("texte", ()))

    def test_threshold_separates_two_populations(self):
        threshold, accuracy = suggest_threshold([0.8, 0.7, 0.75], [0.3, 0.4])
        self.assertTrue(0.4 < threshold < 0.7)
        self.assertEqual(accuracy, 1.0)

    def test_threshold_with_overlap(self):
        threshold, accuracy = suggest_threshold([0.8, 0.5], [0.6, 0.2])
        self.assertLess(accuracy, 1.0)
        self.assertIsNotNone(threshold)

    def test_no_threshold_without_both_populations(self):
        self.assertEqual(suggest_threshold([0.5], []), (None, None))

    def test_csv_columns_are_those_of_the_reference_results(self):
        """Mêmes colonnes, dans le même ordre, que les résultats de référence."""
        references = sorted((ROOT / "eval" / "resultats").glob("*/resultats.csv"))
        self.assertTrue(references)
        for path in references:
            with self.subTest(path=path.parent.name):
                header = path.read_text(encoding="utf-8").splitlines()[0]
                self.assertEqual(header.split(","), CSV_FIELDS)


class ValidateThresholdTest(unittest.TestCase):
    def test_measures_what_a_threshold_keeps_and_refuses_on_unseen_questions(self):
        def q(qid, answerable, expected=()):
            return EvalQuestion(qid, "?", "alice", answerable, tuple(expected), (), ())
        rows = [
            RetrievalScore(q("a1", True, ["d"]), top1=0.80, hit=True, hit1=True),
            RetrievalScore(q("a2", True, ["d"]), top1=0.55, hit=True, hit1=False),
            RetrievalScore(q("u1", False), top1=0.40, hit=None, hit1=None),
            RetrievalScore(q("u2", False), top1=0.70, hit=None, hit1=None),
        ]
        checked = validate_threshold(rows, 0.60)
        self.assertEqual(checked.questions, 4)
        self.assertEqual(checked.hit_at_1, 0.5)
        self.assertEqual(checked.kept_answerable, 0.5)
        self.assertEqual(checked.correct_refusals, 0.5)


class SummaryTest(unittest.TestCase):
    def test_latency_is_measured_on_generated_answers_only(self):
        rows = [
            BenchmarkRow("e", "g", 1, "q1", True, "answered", attempts=1, latency_ms=900),
            # Refus sans appel au modèle : quasi immédiat, il ne compte pas dans la latence.
            BenchmarkRow("e", "g", 1, "q2", False, "no_relevant_source", attempts=0, latency_ms=2),
        ]
        (generation,) = summarize(rows)
        self.assertEqual(generation.latency_median_ms, 900.0)
        self.assertEqual(generation.latency_p90_ms, 900.0)
        self.assertEqual(generation.mean_attempts, 1.0)
        self.assertEqual(generation.refusals_without_generation, 1)

    def test_synthesis_is_written_under_fixed_names_in_a_fixed_order(self):
        """synthese.json : les noms des champs et leur ordre, figés ici ; ceux des résultats de référence
        (eval/resultats) y sont, dans le même ordre."""
        validation = ThresholdValidation(16, 0.923, 0.923, 1.0, 0.65)
        retrieval = RetrievalSummary("bge-m3", "ollama:bge-m3@7907", 1024, 740, 19.9, 0.969, 0.969, 4, 0.737, 0.535,
                                     0.65, False, 0.635, 1.0, 0.65, validation)
        generation = GenerationSummary("bge-m3", "llama3-2-3b", 42, 0, 0.969, 1.0, 0.871, 0.024, 0.9, 0, None, 1.1,
                                       850.0, 1200.0, 3)
        self.assertEqual(list(retrieval.to_json()), [
            "embedding", "model_id", "dimension", "chunks", "index_seconds", "hit@1", "hit@k", "top_k",
            "top1_median_answerable", "top1_median_unanswerable", "configured_threshold",
            "configured_threshold_is_default", "suggested_threshold", "separation_accuracy", "threshold_used",
            "validation"])
        self.assertEqual(retrieval.to_json()["validation"], {
            "questions": 16, "hit@1": 0.923, "kept_answerable": 0.923, "correct_refusals": 1.0, "threshold": 0.65})
        self.assertNotIn("validation", replace(retrieval, validation=None).to_json())   # sans --validate-with
        self.assertEqual(IndexingFailure("nomic", "panne").to_json(), {"embedding": "nomic", "error": "panne"})
        self.assertEqual(list(generation.to_json()), [
            "embedding", "generation", "calls", "errors", "answer_rate", "source_hit_rate", "keyword_coverage",
            "unsourced_rate", "correct_refusal_rate", "forbidden_leaks", "stability", "mean_attempts",
            "latency_median_ms", "latency_p90_ms", "refusals_without_generation"])
        self.assertIsNone(generation.to_json()["stability"])   # « sans objet » s'écrit null

    def test_synthesis_is_written_as_frozen(self):
        """Les octets de synthese.json qu'écrit le banc (run_benchmark) : json.dumps(…, ensure_ascii=False, indent=2).
        La synthèse est donnée (BenchmarkSummary remplacé) : ni service IA, ni modèle, ni question."""
        summary = BenchmarkSummary(
            (RetrievalSummary("hashing", "hashing:256@abc", 256, 15, 0.3, 1.0, 1.0, 4, 0.5, 0.25, 0.4, True, 0.375, 1.0,
                              0.4, ThresholdValidation(2, 1.0, 1.0, 0.0, 0.4)),
             IndexingFailure("nomic", "Service IA : HTTP 502 — « modèle absent »")),
            (GenerationSummary("hashing", "extractive", 4, 0, 1.0, 0.5, None, 0.0, 1.0, 0, None, 1.0, 12.0, 15.0, 2),))
        config = AppConfig.load(ROOT / "config" / "app.toml")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(benchmark, "check_ai_service"), \
                mock.patch.object(benchmark, "BenchmarkSummary", return_value=summary):
            run_benchmark(config, [], [], [], runs=1, out_dir=Path(tmp), log=lambda *_: None)
            # Les octets, pas read_text, qui ramènerait à « \n » les « \r\n » qu'écrirait le mode texte sous Windows.
            self.assertEqual((Path(tmp) / "synthese.json").read_bytes().decode("utf-8"), SYNTHESIS)


if __name__ == "__main__":
    unittest.main()
