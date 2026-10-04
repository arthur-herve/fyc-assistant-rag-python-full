import unittest

from coeur.access import AccessPolicy
from coeur.citations import check_citations
from coeur.model import Chunk, User


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


if __name__ == "__main__":
    unittest.main()
