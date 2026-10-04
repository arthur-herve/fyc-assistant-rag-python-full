"""JSON lu strictement (json_text.parse) : index, instantanés, jeux de questions, réponses du service IA, corps des
requêtes HTTP."""

import json
import tracemalloc
import unittest

from assistant.infrastructure.json_text import parse


def nested(levels: int) -> str:
    """Des listes imbriquées sur `levels` niveaux."""
    return "[" * levels + "]" * levels


NOT_TEXT = "chaîne qui n'est pas du texte : surrogate UTF-16 isolé (\\ud800 à \\udfff sans sa paire)"
TOO_DEEP = "JSON trop imbriqué : plus de 900 niveaux"
TOO_LONG = "nombre entier de plus de 4300 chiffres"

# Au-delà de 900 niveaux, le texte est refusé d'emblée, avant json.loads (qui abandonne plus ou moins loin selon la
# version de Python et la pile d'appels), même si un autre défaut vient avant ; les crochets et les accolades d'une
# chaîne n'y comptent pas (un antislash y garde le caractère qui le suit, quel qu'il soit ; hors d'une chaîne, il ne
# garde rien). Un défaut au fond de 900 niveaux, ou juste après, est signalé sous toute version de Python : json.loads
# y lit encore, même sous 3.11 (qui abandonne vers 982 à 990 niveaux).
FIRST_DEFECT = (
    ('{"a": 1, "a": 2, "x": ' + nested(100_000) + "}", TOO_DEEP),
    ('[{"a": 1, "a": 2}, ' + nested(900) + "]", TOO_DEEP),                      # 901 niveaux
    ('[{"a": 1, "a": 2}, ' + nested(899) + "]", "clé « a » en double"),       # 900 niveaux
    ("[" * 900 + "NaN" + "]" * 900, "« NaN » n'est pas du JSON"),       # au fond de 900 niveaux
    ("[" * 901 + "NaN" + "]" * 901, TOO_DEEP),                          # au fond de 901 niveaux
    ("[" * 990 + "NaN" + "]" * 990, TOO_DEEP),                          # Python 3.13 y lisait le NaN
    ("[" + nested(899) + ", NaN]", "« NaN » n'est pas du JSON"),        # après 900 niveaux
    ("[" + nested(900) + ", NaN]", TOO_DEEP),                           # après 901 niveaux
    ("[NaN, " + nested(1000) + "]", TOO_DEEP),
    ("[1 2, " + nested(1000) + "]", TOO_DEEP),   # même une erreur de syntaxe
    ('{"x": ' + nested(2000) + ', "a": 1, "a": 2}', TOO_DEEP),   # Python 3.13 les lisait
    ('["' + "[" * 1001 + '", {"a": 1, "a": 2}]', "clé « a » en double"),       # des crochets dans une chaîne
    ('["\\"' + "[" * 1001 + '", {"a": 1, "a": 2}]', "clé « a » en double"),    # après un guillemet échappé
    ('["\\\\", ' + nested(1000) + ', {"a": 1, "a": 2}]', TOO_DEEP),   # « \\ » ferme la chaîne
    ('["]}", NaN, ' + nested(1000) + "]", TOO_DEEP),   # des fermants dans une chaîne
    ('["\\\n", ' + nested(1000) + "]", TOO_DEEP),   # « \ » garde même un saut de ligne
    ("\\" + nested(1001), TOO_DEEP),   # hors chaîne, « \ » ne garde rien
)

# NaN, Infinity et -Infinity à la place d'une valeur : ce message ; ailleurs (None), une erreur de syntaxe
# (JSONDecodeError).
CONSTANTS = (
    ("NaN", "« NaN » n'est pas du JSON"),
    (" \r\n NaN", "« NaN » n'est pas du JSON"),
    ('{"a" : NaN}', "« NaN » n'est pas du JSON"),
    ("[1 , Infinity]", "« Infinity » n'est pas du JSON"),
    ('{"a": [-Infinity]}', "« -Infinity » n'est pas du JSON"),
    ("[NaNx]", "« NaN » n'est pas du JSON"),   # json.loads reconnaît NaN avant de voir le x
    ('[{"a": 1}, NaN]', "« NaN » n'est pas du JSON"),
    ("{NaN: 1}", None),
    ('{"a": 1, NaN: 2}', None),
    ('{"a": [1], NaN: 2}', None),
    ("[1 NaN]", None),
    ("[-NaN]", None),
    ("[nan]", None),
    ("1, NaN", None),
)


class StrictReadingTest(unittest.TestCase):
    """Ce que json.loads lirait de travers (clé en double, NaN) ou refuserait par une trace (RecursionError) ou en
    anglais (entier de plus de 4300 chiffres) : une ValueError, avec un message en français."""

    def test_a_duplicate_key_is_refused(self):
        for text, message in (
            ('{"a": 1, "a": {"x": 1, "x": 2}}', "clé « x » en double"),   # l'objet imbriqué se ferme le premier
            ('{"a": 1, "b": 1, "b": 2, "a": 2}', "clé « b » en double"),   # le premier doublon de l'objet
            ('[{"a": 1}, {"a": 1, "a": 1}]', "clé « a » en double"),
        ):
            with self.subTest(text), self.assertRaises(ValueError) as caught:
                parse(text)
            self.assertEqual(str(caught.exception), message)

    def test_nine_hundred_levels_are_read_not_nine_hundred_and_one(self):
        # 900 niveaux se lisent, pas 901. Avec strings aussi : les clés des objets sont un emoji écrit en paire,
        # leurs chaînes sont donc parcourues ; celles des listes, sans échappement, ne le sont pas.
        for depth, objects, read in ((900, False, True), (901, False, False), (900, True, True), (901, True, False),
                                     (1000, False, False), (1000, True, False), (1001, False, False),
                                     (100_000, False, False)):
            text = ('{"\\ud83d\\ude00": ' * (depth - 1) + "{}" + "}" * (depth - 1) if objects
                    else "[" * depth + "]" * depth)
            for strings in (False, True):
                with self.subTest(depth=depth, objects=objects, strings=strings):
                    if read:
                        self.assertIsNotNone(parse(text, strings=strings))
                        continue
                    with self.assertRaises(ValueError) as caught:
                        parse(text, strings=strings)
                    self.assertEqual(str(caught.exception), TOO_DEEP)

    def test_nan_and_infinity_are_not_json(self):
        for text, message in CONSTANTS:
            with self.subTest(text):
                if message is None:
                    self.assertRaises(json.JSONDecodeError, parse, text)
                    continue
                with self.assertRaises(ValueError) as caught:
                    parse(text)
                self.assertEqual(str(caught.exception), message)

    def test_the_depth_is_checked_before_any_other_defect(self):
        for text, message in FIRST_DEFECT:
            for strings in (False, True):
                with self.subTest(text[:60], strings=strings):
                    with self.assertRaises(ValueError) as caught:
                        parse(text, strings=strings)
                    self.assertEqual(str(caught.exception), message)

    def test_invalid_syntax_is_still_a_json_decode_error(self):
        # Une chaîne non fermée court jusqu'à la fin du texte : la pré-lecture n'en compte pas les crochets.
        for text in ("{pas json", "[1,]", '["' + "[" * 1001):
            with self.subTest(text[:12]):
                self.assertRaises(json.JSONDecodeError, parse, text)

    def test_a_string_full_of_escapes_is_skipped_without_memory(self):
        # La pré-lecture saute les chaînes par une expression régulière : sans répétition possessive, re gardait plus
        # de 100 octets par échappement (1,7 Gio pour un corps HTTP de 16 Mio). Le pic reste de l'ordre du texte.
        text = '["' + "a\\n" * 70_000 + '"]'
        tracemalloc.start()
        try:
            value = parse(text, strings=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(value, ["a\n" * 70_000])
        self.assertLess(peak, 4 * len(text))

    def test_an_integer_of_more_than_4300_digits_has_a_french_message(self):
        # int() le refuserait de toute façon, avec un message en anglais. 4300 chiffres, signe non compté,
        # passent ; un réel n'est pas converti par int().
        for text in ("[" + "1" * 4301 + "]", "[-" + "1" * 4301 + "]", '{"a": ' + "9" * 5000 + "}"):
            with self.subTest(text[:12]), self.assertRaises(ValueError) as caught:
                parse(text)
            self.assertEqual(str(caught.exception), TOO_LONG)
        for text in ("[" + "1" * 4300 + "]", "[-" + "1" * 4300 + "]", "[" + "1" * 5000 + ".5]"):
            with self.subTest(text[:12]):
                self.assertEqual(len(parse(text)), 1)

    def test_a_string_that_is_not_text_is_refused_only_when_asked(self):
        # L'index, les instantanés, les jeux de questions, les corps des requêtes HTTP et GET /v1/models le demandent ;
        # les autres réponses du service IA le laissent à leurs propres contrôles (_is_text).
        self.assertEqual(parse('{"a": "x\\ud800"}'), {"a": "x\ud800"})
        # Les chiffres de l'échappement en majuscules aussi (« \uD800 »), comme les écrivent d'autres outils que Python.
        for text in ('{"a": "x\\ud800"}', '{"x\\udc00": 1}', '["\\ud83d"]', '"\\ude00"', '["\\uD800"]', '{"\\uDFFF": 1}',
                     '["a\\uDc00"]'):
            with self.subTest(text), self.assertRaises(ValueError) as caught:
                parse(text, strings=True)
            self.assertEqual(str(caught.exception), NOT_TEXT)
        self.assertEqual(parse('["\\ud83d\\ude00"]', strings=True), ["\U0001F600"])   # une paire complète


if __name__ == "__main__":
    unittest.main()
