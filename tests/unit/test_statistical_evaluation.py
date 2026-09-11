"""Séquence 3.1 : quand le composant est probabiliste, le test devient statistique.

On ne vérifie plus « la réponse est X », mais « sur N essais, au moins une
proportion p respecte la règle ». Le seuil est choisi pour que la probabilité
d'un faux échec soit négligeable, et ce choix est documenté.
"""

import random
import unittest

from assistant.application.ports import Generation
from assistant.domain.model import AnswerStatus, User
from tests.fakes import ScriptedGenerator
from tests.unit.test_ask_question import indexed, use_case

TRIALS = 50
FORGET_RATE = 0.3          # le « modèle » oublie de citer 3 fois sur 10
MIN_ANSWER_RATE = 0.75     # attendu ≈ 0,91 avec 2 tentatives ; faux échec < 0,1 %


class ForgetfulGenerator(ScriptedGenerator):
    def __init__(self):
        super().__init__("")
        self._random = random.Random()  # volontairement non initialisé

    def generate(self, request):
        self.requests.append(request)
        cited = self._random.random() >= FORGET_RATE
        return Generation("forgetful", "Deux jours [1]." if cited else "Deux jours.")


class StatisticalEvaluationTest(unittest.TestCase):
    def test_answer_rate_stays_above_the_tolerance(self):
        index = indexed()
        ask = use_case(ForgetfulGenerator(), index=index, max_attempts=2)
        user = User("alice", frozenset({"tous"}))

        statuses = [ask.execute(user, "jours de télétravail").status for _ in range(TRIALS)]
        rate = statuses.count(AnswerStatus.ANSWERED) / TRIALS

        self.assertGreaterEqual(rate, MIN_ANSWER_RATE, f"taux de réponse sourcée : {rate:.2f}")
        # La règle métier, elle, est déterministe : jamais de réponse non sourcée affichée.
        self.assertTrue(set(statuses) <= {AnswerStatus.ANSWERED, AnswerStatus.UNSOURCED})


if __name__ == "__main__":
    unittest.main()
