"""Décorateurs sur les ports : ajouter un comportement sans toucher au cœur (S4.1)."""

import logging
import unittest

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.errors import AIServiceError, ModelOutputRejectedError
from assistant.application.guards import OutputValidatingGenerator
from assistant.application.index_corpus import IndexCorpus
from assistant.application.ports import Generation, GenerationRequest
from assistant.composition import decorate
from assistant.domain.model import AnswerStatus, User
from assistant.infrastructure.decorators import (
    CachedEmbedder, LoggingGenerator, RetryingEmbedder, RetryingGenerator,
)
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, ScriptedGenerator, StaticPrompts,
    WholeDocumentSplitter, make_document,
)

REQUEST = GenerationRequest("système", "Passages :\n[1] x\n\nQuestion : ?", 0.2, 100)
SILENT = logging.getLogger("test.silencieux")
SILENT.addHandler(logging.NullHandler())
SILENT.propagate = False
ALICE = User("alice", frozenset({"tous"}))
LEAK = ("Okay, let's see. The user is asking about remote work. The passage [1] says "
        "two days per week, so the answer should be two days.")


class FlakyGenerator:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise AIServiceError("injoignable")
        return Generation("llm", "Deux jours [1].")


class OutputValidatingGeneratorTest(unittest.TestCase):
    def test_passes_a_valid_answer_through(self):
        generator = OutputValidatingGenerator(ScriptedGenerator("Deux jours [1]."))
        self.assertEqual(generator.generate(REQUEST).text, "Deux jours [1].")

    def test_rejects_leaked_reasoning_with_the_reasons(self):
        generator = OutputValidatingGenerator(ScriptedGenerator(LEAK, model="qwen"))
        with self.assertRaises(ModelOutputRejectedError) as caught:
            generator.generate(REQUEST)
        self.assertEqual(caught.exception.model, "qwen")
        self.assertTrue(any("raisonnement" in p for p in caught.exception.problems))

    def test_use_case_counts_a_rejection_as_a_failed_attempt_and_retries(self):
        index = FakeIndex()
        IndexCorpus(ListSource([make_document("tt", "Deux jours de télétravail.")]),
                    WholeDocumentSplitter(), KeywordEmbedder(), index, FixedClock()).execute()
        inner = ScriptedGenerator(LEAK, "Deux jours par semaine [1].")
        ask = AskQuestion(KeywordEmbedder(), index, OutputValidatingGenerator(inner),
                          StaticPrompts(), AskSettings(min_score=0.1, max_attempts=2))
        answer = ask.execute(ALICE, "Combien de jours de télétravail ?")
        self.assertEqual(answer.status, AnswerStatus.ANSWERED)
        self.assertEqual(answer.trace.attempts, 2)
        self.assertTrue(answer.trace.raw_outputs[0].startswith("<rejetée : "))

    def test_use_case_gives_up_after_all_attempts(self):
        index = FakeIndex()
        IndexCorpus(ListSource([make_document("tt", "Deux jours de télétravail.")]),
                    WholeDocumentSplitter(), KeywordEmbedder(), index, FixedClock()).execute()
        ask = AskQuestion(KeywordEmbedder(), index, OutputValidatingGenerator(ScriptedGenerator(LEAK)),
                          StaticPrompts(), AskSettings(min_score=0.1, max_attempts=2))
        answer = ask.execute(ALICE, "Combien de jours de télétravail ?")
        self.assertEqual(answer.status, AnswerStatus.UNSOURCED)
        self.assertEqual(len(answer.trace.raw_outputs), 2)


class CachedEmbedderTest(unittest.TestCase):
    def test_same_text_is_embedded_once(self):
        inner = KeywordEmbedder()
        cached = CachedEmbedder(inner)
        first = cached.embed_query("télétravail")
        second = cached.embed_query("télétravail")
        self.assertIs(first, second)
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual((cached.hits, cached.misses), (1, 1))

    def test_documents_are_cached_by_batch(self):
        inner = KeywordEmbedder()
        cached = CachedEmbedder(inner)
        cached.embed_documents(["a", "b"])
        cached.embed_documents(["a", "b"])
        cached.embed_documents(["a"])
        self.assertEqual(len(inner.calls), 2)


class RetryingTest(unittest.TestCase):
    def test_retries_on_ai_service_error_then_succeeds(self):
        inner = FlakyGenerator(failures=1)
        slept: list[float] = []
        generator = RetryingGenerator(inner, attempts=1, delay_seconds=0.5, sleep=slept.append,
                                      logger=SILENT)
        self.assertEqual(generator.generate(REQUEST).text, "Deux jours [1].")
        self.assertEqual(inner.calls, 2)
        self.assertEqual(slept, [0.5])

    def test_does_not_retry_a_refused_request(self):
        class Refusing:
            calls = 0

            def generate(self, request):
                self.calls += 1
                raise AIServiceError("HTTP 404 — modèle inconnu", transient=False)

        inner = Refusing()
        with self.assertRaises(AIServiceError):
            RetryingGenerator(inner, attempts=3, sleep=lambda _: None, logger=SILENT).generate(REQUEST)
        self.assertEqual(inner.calls, 1)

    def test_gives_up_after_the_configured_attempts(self):
        inner = FlakyGenerator(failures=5)
        generator = RetryingGenerator(inner, attempts=2, sleep=lambda _: None, logger=SILENT)
        with self.assertRaises(AIServiceError):
            generator.generate(REQUEST)
        self.assertEqual(inner.calls, 3)

    def test_embedder_variant(self):
        class Flaky(KeywordEmbedder):
            def __init__(self):
                super().__init__()
                self.failed = False

            def embed_query(self, text):
                if not self.failed:
                    self.failed = True
                    raise AIServiceError("injoignable")
                return super().embed_query(text)

        embedder = RetryingEmbedder(Flaky(), attempts=1, sleep=lambda _: None, logger=SILENT)
        self.assertEqual(embedder.embed_query("télétravail").dimension, 8)


class LoggingTest(unittest.TestCase):
    def test_generation_is_logged_with_model_and_duration(self):
        logger = logging.getLogger("test.assistant")
        with self.assertLogs(logger, level="INFO") as logs:
            LoggingGenerator(ScriptedGenerator("Deux jours [1].", model="llm-x"), logger).generate(REQUEST)
        self.assertIn("llm-x", logs.output[0])
        self.assertIn("ms", logs.output[0])


class CompositionTest(unittest.TestCase):
    def test_status_uses_the_raw_embedder_not_the_cache(self):
        """Un cache d'embeddings masquerait un changement de modèle servi (revue du 11/09)."""
        import inspect

        from assistant import composition
        self.assertIn("CheckStatus(source, splitter, raw_embedder", inspect.getsource(composition.build))

    def test_decorators_are_stacked_from_configuration_only(self):
        embedder, generator = decorate(KeywordEmbedder(), ScriptedGenerator("x"), {})
        self.assertIsInstance(generator, OutputValidatingGenerator)   # règle métier : activée par défaut
        self.assertIsInstance(embedder, KeywordEmbedder)              # rien de technique sans le demander

        embedder, generator = decorate(KeywordEmbedder(), ScriptedGenerator("x"),
                                       {"cache_embeddings": True, "retries": 2, "log": True,
                                        "validate_output": False})
        self.assertIsInstance(embedder, CachedEmbedder)
        self.assertNotIsInstance(generator, OutputValidatingGenerator)


if __name__ == "__main__":
    unittest.main()
