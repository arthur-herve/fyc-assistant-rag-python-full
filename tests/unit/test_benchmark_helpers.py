import unittest
from pathlib import Path

from assistant.interface.benchmark import (
    CSV_FIELDS, EvalQuestion, RetrievalScore, keyword_coverage, suggest_threshold, validate_threshold,
)

ROOT = Path(__file__).resolve().parents[2]


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
        """Mêmes colonnes, dans le même ordre, que les résultats de référence (et que la version C#)."""
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
        self.assertEqual(checked["questions"], 4)
        self.assertEqual(checked["hit@1"], 0.5)
        self.assertEqual(checked["kept_answerable"], 0.5)
        self.assertEqual(checked["correct_refusals"], 0.5)


if __name__ == "__main__":
    unittest.main()
