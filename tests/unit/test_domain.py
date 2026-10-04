import unittest

from assistant.domain.access import AccessPolicy
from assistant.domain.citations import check_citations
from assistant.domain.model import Chunk, User


def chunk(groups):
    return Chunk("d#0", "d", "D", "texte", 0, frozenset(groups))


class AccessPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = AccessPolicy()

    def test_public_document_is_readable_by_everyone(self):
        self.assertTrue(self.policy.can_read(User("x", frozenset()), chunk(["tous"])))

    def test_restricted_document_requires_a_shared_group(self):
        restricted = chunk(["rh", "direction"])
        self.assertFalse(self.policy.can_read(User("alice", frozenset({"tous"})), restricted))
        self.assertTrue(self.policy.can_read(User("bruno", frozenset({"tous", "rh"})), restricted))


class CitationsTest(unittest.TestCase):
    def test_valid_citations(self):
        check = check_citations("Deux jours [1]. Indemnité de 30 euros [2, 3].", passage_count=3)
        self.assertEqual(check.cited, (1, 2, 3))
        self.assertTrue(check.is_valid)

    def test_no_citation_is_invalid(self):
        self.assertFalse(check_citations("Deux jours par semaine.", 3).is_valid)

    def test_citation_to_a_passage_that_was_not_provided_is_invalid(self):
        check = check_citations("Deux jours [1] et [7].", passage_count=2)
        self.assertEqual(check.invalid, (7,))
        self.assertFalse(check.is_valid)

    def test_duplicates_are_counted_once(self):
        self.assertEqual(check_citations("[2] puis [2,1]", 2).cited, (2, 1))

    def test_a_huge_citation_number_is_invalid_not_an_exception(self):
        # Des milliers de chiffres, que int() refuse de convertir : une citation invalide, sans exception.
        check = check_citations("Deux jours [1], voir [" + "9" * 5000 + "].", passage_count=2)
        self.assertEqual(check.cited, (1,))
        self.assertEqual(len(check.invalid), 1)
        self.assertFalse(check.is_valid)


class UnusualCitationsTest(unittest.TestCase):
    # (sortie du modèle, citations valides, citations invalides) pour 2 passages.
    UNUSUAL = [
        ("[1,\xa02]", (1, 2), ()),           # espace insécable, U+2028, \x85 : des blancs pour \s
        ("[1,\u20282]", (1, 2), ()),
        ("[1\x85, 2]", (1, 2), ()),
        ("[33612345678] puis [44612345678]", (), (33612345678, 44612345678)),   # trop grands : tels quels
        ("[00000000002]", (2,), ()),
        ("[1, 00000000002]", (1, 2), ()),   # un blanc puis onze chiffres : rogné, puis lu comme 2
        ("[0]", (), (0,)),
        ("[\u0661]", (), ()),
    ]

    def test_unusual_outputs_give_citations_without_an_exception(self):
        """Jamais d'exception : une ValueError ici faisait échouer la question (500 avec serve)."""
        for text, cited, invalid in self.UNUSUAL:
            with self.subTest(text=text[:40]):
                check = check_citations(text, 2)
                self.assertEqual((check.cited, check.invalid), (cited, invalid))

    def test_only_latin_digits_count_as_a_citation(self):
        """« [١] » (chiffre arabe) n'est pas une citation : les passages sont numérotés en chiffres latins."""
        self.assertFalse(check_citations("Deux jours [\u0661].", 1).is_valid)
        self.assertTrue(check_citations("Deux jours [1].", 1).is_valid)


if __name__ == "__main__":
    unittest.main()
