import unittest

from assistant.interface.benchmark import (
    EvalQuestion, keyword_coverage, suggest_threshold, validate_threshold,
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


class ValidateThresholdTest(unittest.TestCase):
    def test_measures_what_a_threshold_keeps_and_refuses_on_unseen_questions(self):
        def q(qid, answerable, expected=()):
            return EvalQuestion(qid, "?", "alice", answerable, tuple(expected), (), ())
        rows = [
            {"question": q("a1", True, ["d"]), "top1": 0.80, "hit": True, "hit1": True},
            {"question": q("a2", True, ["d"]), "top1": 0.55, "hit": True, "hit1": False},
            {"question": q("u1", False), "top1": 0.40, "hit": None, "hit1": None},
            {"question": q("u2", False), "top1": 0.70, "hit": None, "hit1": None},
        ]
        checked = validate_threshold(rows, 0.60)
        self.assertEqual(checked["questions"], 4)
        self.assertEqual(checked["hit@1"], 0.5)
        self.assertEqual(checked["kept_answerable"], 0.5)
        self.assertEqual(checked["correct_refusals"], 0.5)


if __name__ == "__main__":
    unittest.main()
