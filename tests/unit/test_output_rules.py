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

    def test_the_length_is_counted_in_code_points(self):
        """Un emoji compte pour un caractère, comme dans len(). Une moitié de paire isolée compte pour un caractère.
        Ce sont des points de code, pas des caractères perçus (graphèmes) : un « é » décomposé ou un drapeau en
        comptent deux. Et c'est la longueur de la réponse rognée."""
        def too_long(length):
            return (f"réponse trop longue ({length} caractères, 500 au plus)",)

        cases = [
            ("\U0001F600" * 500, ()),   # 1 000 unités UTF-16
            ("\U0001F600" * 501, too_long(501)),
            ("\ud800" * 501, too_long(501)),
            ("\udc00" * 501, too_long(501)),
            ("e\u0301" * 251, too_long(502)),   # « é » décomposé : 251 graphèmes, 502 points de code
            ("\U0001F1EB\U0001F1F7" * 251, too_long(502)),   # drapeau : 251 graphèmes, 502 points de code
            ("  " + "a" * 500 + "\n", ()),   # 503 avant rognage, 500 après
        ]
        for text, problems in cases:
            with self.subTest(character=f"U+{ord(text[0]):04X}", length=len(text)):
                self.assertEqual(check_output(text, max_chars=500).problems, problems)

    def test_the_blanks_are_those_of_dotnet(self):
        """Le rognage et les marqueurs de raisonnement voient les blancs d'Unicode (blanks.py), ceux de
        char.IsWhiteSpace en .NET, d'où le nom du test. str.strip() et le \\s de Python y ajoutent les séparateurs
        \\x1c à \\x1f, qui n'en sont pas ici."""
        def reasoning(found):
            return (f"raisonnement du modèle déversé dans la réponse (« {found} »)",)

        too_long = ("réponse trop longue (1501 caractères, 1500 au plus)",)
        separators = "\x1c\x1d\x1e\x1f"
        # Tous dans le plan de base (U+0000 à U+FFFF) : le parcourir suffit.
        blanks = {chr(c) for c in range(0x110000) if chr(c).isspace()} - set(separators)
        plane = [chr(c) for c in range(0x10000)]
        # Rognés, ou entre « ok, » et « let » : ces blancs, ni plus ni moins (pas U+200B, par exemple).
        self.assertEqual({c for c in plane if check_output(c * 3).problems == ("réponse vide",)}, blanks)
        self.assertEqual({c for c in plane if check_output(f"ok,{c}let").problems}, blanks)
        for blank in sorted(blanks):
            with self.subTest(blank=f"U+{ord(blank):04X}"):
                self.assertEqual(check_output(blank + "a" * 1500 + blank).problems, ())
                self.assertEqual(check_output(f"ok,{blank}let").problems, reasoning(f"ok,{blank}let"))
                self.assertEqual(check_output(f"first,{blank}i need").problems, reasoning(f"first,{blank}i need"))
                self.assertEqual(check_output(f"wait,{blank}x").problems, reasoning("wait,"))
        for separator in separators:   # des blancs pour str.isspace(), pas pour cette règle
            with self.subTest(separator=f"U+{ord(separator):04X}"):
                self.assertEqual(check_output(separator * 3).problems, ())
                self.assertEqual(check_output(separator + "a" * 1500).problems, too_long)
                self.assertEqual(check_output("a" * 1500 + separator).problems, too_long)
                for text in (f"ok,{separator}let", f"first,{separator}i need", f"wait,{separator}x"):
                    self.assertEqual(check_output(text).problems, ())

    def test_think_tags_are_rejected(self):
        self.assertFalse(check_output("<think>je réfléchis</think> Deux jours [1].").is_valid)


if __name__ == "__main__":
    unittest.main()
