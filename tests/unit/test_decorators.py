"""Décorateurs sur les ports : ajouter un comportement sans toucher au cœur (S4.1)."""

import logging
import unittest
from pathlib import Path

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.errors import AIServiceError, ModelOutputRejectedError
from assistant.application.guards import OutputValidatingGenerator
from assistant.application.index_corpus import IndexCorpus
from assistant.application.ports import Generation, GenerationRequest, IndexManifest
from assistant.composition import AppConfig, build, decorate
from assistant.domain.model import AnswerStatus, User
from assistant.infrastructure.decorators import (
    CachedEmbedder, LoggingGenerator, RetryingEmbedder, RetryingGenerator,
)
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, ScriptedGenerator, StaticPrompts,
    WholeDocumentSplitter, make_document,
)

ROOT = Path(__file__).resolve().parents[2]
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


def manifest(model: str = "fake-keywords", dimension: int = 8) -> IndexManifest:
    return IndexManifest("idx", model, dimension, "empreinte", {"type": "whole"}, 1, 1, "2026-09-21T12:00:00")


class CachedEmbedderTest(unittest.TestCase):
    def test_same_question_is_embedded_once(self):
        inner, index = KeywordEmbedder(), manifest()
        cached = CachedEmbedder(inner, current_index=lambda: index)
        first = cached.embed_query("télétravail")
        second = cached.embed_query("télétravail")
        self.assertIs(first, second)
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual((cached.hits, cached.misses), (1, 1))

    def test_documents_are_never_cached(self):
        """Une réindexation doit refléter le modèle servi maintenant (S4.2)."""
        inner, index = KeywordEmbedder(), manifest()
        cached = CachedEmbedder(inner, current_index=lambda: index)
        cached.embed_documents(["a", "b"])
        cached.embed_documents(["a", "b"])
        self.assertEqual(len(inner.calls), 2)

    def test_a_new_index_empties_the_cache_even_with_the_same_model_name(self):
        """Préfixes changés, moteur sans empreinte : les vecteurs changent, pas le nom du modèle."""
        inner, current = KeywordEmbedder(), {"index": manifest()}
        cached = CachedEmbedder(inner, current_index=lambda: current["index"])
        cached.embed_query("télétravail")
        current["index"] = manifest()   # réindexé : nouveau manifeste, mêmes valeurs
        cached.embed_query("télétravail")
        self.assertEqual(len(inner.calls), 2)

    def test_vectors_that_do_not_match_the_index_are_not_kept(self):
        """Sinon, une question posée pendant que le service servait un autre modèle
        resterait en erreur après le retour du service au modèle de l'index."""
        inner, index = KeywordEmbedder("modele-b"), manifest("modele-a")
        cached = CachedEmbedder(inner, current_index=lambda: index)
        self.assertEqual(cached.embed_query("télétravail").model, "modele-b")   # AskQuestion : erreur
        inner.model = "modele-a"                                                # le service revient
        self.assertEqual(cached.embed_query("télétravail").model, "modele-a")
        cached.embed_query("télétravail")
        self.assertEqual((len(inner.calls), cached.hits), (2, 1))

    def test_vectors_of_another_dimension_are_not_kept(self):
        """Même nom de modèle, autre dimension (réglage du moteur changé) : rien à garder pour cet index."""
        inner, index = KeywordEmbedder(), manifest(dimension=16)
        cached = CachedEmbedder(inner, current_index=lambda: index)
        cached.embed_query("télétravail")
        cached.embed_query("télétravail")
        self.assertEqual((len(inner.calls), cached.hits), (2, 0))

    def test_a_vector_computed_for_the_previous_index_is_not_kept_for_the_new_one(self):
        """Réindexé avec un autre modèle pendant l'appel au service : le vecteur de l'ancien modèle
        n'entre pas dans le cache du nouvel index, sinon la question y resterait refusée (409)."""
        current = {"index": manifest("modele-a")}

        class ReindexedDuringTheCall(KeywordEmbedder):
            def embed_query(self, text):
                batch = super().embed_query(text)
                if len(self.calls) == 1:   # une autre requête voit déjà le nouvel index et le nouveau modèle
                    self.model, current["index"] = "modele-b", manifest("modele-b")
                    cached.embed_query("congés")
                return batch

        inner = ReindexedDuringTheCall("modele-a")
        cached = CachedEmbedder(inner, current_index=lambda: current["index"])
        self.assertEqual(cached.embed_query("télétravail").model, "modele-a")   # calculé pour l'ancien index
        self.assertEqual(cached.embed_query("télétravail").model, "modele-b")

    def test_the_oldest_question_leaves_first(self):
        inner, index = KeywordEmbedder(), manifest()
        cached = CachedEmbedder(inner, current_index=lambda: index, max_entries=1)
        for question in ("télétravail", "congés", "télétravail"):
            cached.embed_query(question)
        self.assertEqual(len(inner.calls), 3)


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
    def test_decorators_are_stacked_from_configuration_only(self):
        no_index = lambda: None  # noqa: E731
        embedder, generator = decorate(KeywordEmbedder(), ScriptedGenerator("x"), {}, current_index=no_index)
        self.assertIsInstance(generator, OutputValidatingGenerator)   # règle métier : activée par défaut
        self.assertIsInstance(embedder, KeywordEmbedder)              # rien de technique sans le demander

        embedder, generator = decorate(KeywordEmbedder(), ScriptedGenerator("x"),
                                       {"cache_embeddings": True, "retries": 2, "log": True,
                                        "validate_output": False}, current_index=no_index)
        self.assertIsInstance(embedder, CachedEmbedder)
        self.assertNotIsInstance(generator, OutputValidatingGenerator)

    def test_a_rejected_output_is_refused_and_logged_whatever_else_is_stacked(self):
        options = {"cache_embeddings": True, "retries": 2, "log": True}
        _, generator = decorate(KeywordEmbedder(), ScriptedGenerator(LEAK), options, current_index=lambda: None)
        with self.assertLogs("assistant", level="WARNING") as logs, self.assertRaises(ModelOutputRejectedError):
            generator.generate(REQUEST)
        self.assertIn("rejetée", "\n".join(logs.output))   # la validation est sous le journal des générations
        _, unchecked = decorate(KeywordEmbedder(), ScriptedGenerator(LEAK), {**options, "validate_output": False},
                                current_index=lambda: None)
        self.assertEqual(unchecked.generate(REQUEST).text, LEAK)

    def test_the_threshold_follows_the_embedding_model_chosen_on_the_command_line(self):
        config = AppConfig.load(ROOT / "config" / "app.toml")
        self.assertNotEqual(config.min_score_for("nomic"), config.min_score_for(config.embedding_model))
        self.assertEqual(build(config, embedding_model="nomic").settings.min_score, config.min_score_for("nomic"))
        self.assertEqual(build(config, embedding_model="nomic", min_score=0.9).settings.min_score, 0.9)


if __name__ == "__main__":
    unittest.main()
