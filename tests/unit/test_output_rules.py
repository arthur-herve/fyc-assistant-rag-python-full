"""La forme d'une réponse : règle déterministe autour d'un modèle bavard (S3.1, S4.1)."""

import unittest

from assistant.domain.output_rules import check_output

QWEN_LEAK = (
    "Okay, let's see. The user is asking how many days of remote work per week. "
    "First, I need to check the provided passages. Passage [1] says two days per week."
)


class OutputRulesTest(unittest.TestCase):
    def test_a_short_french_answer_with_a_citation_is_valid(self):
        self.assertTrue(check_output("Vous pouvez télétravailler deux jours par semaine [1].").is_valid)

    def test_empty_output_is_rejected(self):
        self.assertEqual(check_output("   ").problems, ("réponse vide",))

    def test_leaked_reasoning_is_rejected_even_with_a_citation(self):
        check = check_output(QWEN_LEAK)
        self.assertFalse(check.is_valid)
        self.assertTrue(any("raisonnement" in p for p in check.problems), check.problems)
        self.assertTrue(any("langue" in p for p in check.problems), check.problems)

    def test_english_answer_is_rejected(self):
        check = check_output("The employee is allowed to work from home two days a week and the manager agrees [1].")
        self.assertIn("réponse dans une autre langue que le français", check.problems)

    def test_french_answer_with_a_few_english_words_is_accepted(self):
        text = "Le salarié peut demander un « time off » ; the request is faite auprès du manager, dans les délais du passage [1]."
        self.assertTrue(check_output(text).is_valid, check_output(text).problems)

    def test_french_words_that_look_like_reasoning_markers_are_accepted(self):
        for text in ("Le billet me semble clair : 25 jours calendaires de congé [1].",
                     "Passage [1] et passage [2] répondent à cette question : oui, sous conditions."):
            self.assertTrue(check_output(text).is_valid, check_output(text).problems)

    def test_short_english_answer_is_rejected(self):
        self.assertFalse(check_output("The answer is two days [1].").is_valid)

    def test_too_long_output_is_rejected(self):
        check = check_output("Le salarié a droit à des congés. " * 60, max_chars=500)
        self.assertTrue(any("trop longue" in p for p in check.problems))

    def test_think_tags_are_rejected(self):
        self.assertFalse(check_output("<think>je réfléchis</think> Deux jours [1].").is_valid)


if __name__ == "__main__":
    unittest.main()
