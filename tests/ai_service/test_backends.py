import math
import unittest

from ai_service.backends.base import BackendError
from ai_service.backends.extractive import ExtractiveGenerationBackend
from ai_service.backends.hashing import HashingEmbeddingBackend
from ai_service.backends.ollama import OllamaEmbeddingBackend, OllamaGenerationBackend
from ai_service.backends.openai_compatible import (
    OpenAICompatibleEmbeddingBackend, OpenAICompatibleGenerationBackend,
)
from ai_service.registry import ConfigError, ModelRegistry, UnknownModelError
from tests.ai_service.stub_servers import StubServer, ollama_routes, openai_routes

PROMPT = """Passages :

[1] Télétravail
Télétravail
Chaque salarié peut télétravailler deux jours par semaine.

[2] Congés
Congés
Chaque salarié acquiert vingt-cinq jours de congés payés.

Question : Combien de jours de télétravail ?"""


class HashingBackendTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        backend = HashingEmbeddingBackend(dimension=64)
        first, second = backend.embed(["télétravail"]), backend.embed(["télétravail"])
        self.assertEqual(first.vectors, second.vectors)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in first.vectors[0])), 1.0)
        self.assertEqual(first.model_id, "hashing-64-stem6")

    def test_accents_do_not_matter(self):
        backend = HashingEmbeddingBackend()
        self.assertEqual(backend.embed(["Congés"]).vectors, backend.embed(["conges"]).vectors)


class ExtractiveBackendTest(unittest.TestCase):
    def test_picks_the_closest_sentence_and_cites_it(self):
        _, text = ExtractiveGenerationBackend().generate("", PROMPT, 0.2, 100, None)
        self.assertEqual(text, "Chaque salarié peut télétravailler deux jours par semaine. [1]")

    def test_noisy_mode_sometimes_forgets_citations(self):
        backend = ExtractiveGenerationBackend(citation_failure_rate=0.5, randomize=True)
        outputs = {backend.generate("", PROMPT, 0.2, 100, seed)[1] for seed in range(40)}
        self.assertTrue(any("[" not in o for o in outputs))
        self.assertTrue(any("[" in o for o in outputs))

    def test_same_seed_same_output(self):
        backend = ExtractiveGenerationBackend(citation_failure_rate=0.5, randomize=True)
        self.assertEqual(backend.generate("", PROMPT, 0.2, 100, 3), backend.generate("", PROMPT, 0.2, 100, 3))


class OllamaBackendTest(unittest.TestCase):
    def test_embeddings_request_format_and_model_digest(self):
        with StubServer(ollama_routes()) as stub:
            vectors = OllamaEmbeddingBackend("nomic-embed-text", base_url=stub.url).embed(["a", "b"])
            _, path, body = next(r for r in stub.requests if r[1] == "/api/embed")
        self.assertEqual(body, {"model": "nomic-embed-text", "input": ["a", "b"]})
        self.assertEqual(vectors.model_id, "ollama:nomic-embed-text@0a109f422b47")
        self.assertEqual(vectors.dimension, 3)

    def test_chat_request_format_and_think_tags_removed(self):
        with StubServer(ollama_routes()) as stub:
            model_id, text = OllamaGenerationBackend("qwen3:1.7b", base_url=stub.url, think=False).generate(
                "système", "prompt", 0.2, 150, 42)
            _, _, body = next(r for r in stub.requests if r[1] == "/api/chat")
        self.assertEqual(text, "Deux jours [1].")
        self.assertEqual(model_id, "ollama:qwen3:1.7b@8f68893c685c")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "système"})
        self.assertEqual(body["options"], {"temperature": 0.2, "num_predict": 150, "seed": 42})
        self.assertIs(body["think"], False)
        self.assertIs(body["stream"], False)

    def test_thinking_models_get_a_separate_token_budget_and_thinking_is_discarded(self):
        with StubServer(ollama_routes(thinking=True)) as stub:
            _, text = OllamaGenerationBackend(
                "qwen3:4b", base_url=stub.url, think=True, thinking_tokens=1000,
            ).generate("système", "prompt", 0.2, 150, None)
            _, _, body = next(r for r in stub.requests if r[1] == "/api/chat")
        self.assertEqual(text, "Deux jours [1].")
        self.assertIs(body["think"], True)
        self.assertEqual(body["options"]["num_predict"], 1150)

    def test_thinking_budget_is_ignored_when_thinking_is_off(self):
        backend = OllamaGenerationBackend("x", think=False, thinking_tokens=1000)
        self.assertEqual(backend.thinking_tokens, 0)

    def test_unreachable_ollama_is_a_backend_error(self):
        with self.assertRaises(BackendError):
            OllamaEmbeddingBackend("x", base_url="http://127.0.0.1:1", timeout=2).embed(["a"])


class OpenAICompatibleBackendTest(unittest.TestCase):
    def test_embeddings_are_reordered_by_index(self):
        with StubServer(openai_routes()) as stub:
            vectors = OpenAICompatibleEmbeddingBackend("m", base_url=stub.url + "/v1").embed(["a", "b"])
        self.assertEqual(vectors.vectors, [[0.0, 1.0], [1.0, 1.0]])

    def test_chat(self):
        with StubServer(openai_routes()) as stub:
            _, text = OpenAICompatibleGenerationBackend("m", base_url=stub.url + "/v1").generate(
                "s", "p", 0.1, 50, None)
            _, _, body = stub.requests[0]
        self.assertEqual(text, "Réponse [1]")
        self.assertNotIn("seed", body)


class RegistryTest(unittest.TestCase):
    def test_prefixes_are_applied_by_the_service_not_the_application(self):
        registry = ModelRegistry.from_dict({"embedding": {"e5": {
            "backend": "hashing", "dimension": 16, "query_prefix": "query: ", "document_prefix": "passage: "}}})
        model = registry.embedding("e5")
        direct = HashingEmbeddingBackend(dimension=16)
        self.assertEqual(model.embed(["bonjour"], "query").vectors, direct.embed(["query: bonjour"]).vectors)
        self.assertEqual(model.embed(["bonjour"], "document").vectors, direct.embed(["passage: bonjour"]).vectors)

    def test_unknown_model(self):
        with self.assertRaises(UnknownModelError):
            ModelRegistry.from_dict({}).generation("gpt-9")

    def test_unknown_backend(self):
        with self.assertRaises(ConfigError):
            ModelRegistry.from_dict({"generation": {"x": {"backend": "magie"}}})

    def test_project_configuration_is_valid(self):
        from pathlib import Path
        registry = ModelRegistry.from_file(Path(__file__).parents[2] / "config/ai_service.toml")
        self.assertIn("hashing", [m["alias"] for m in registry.describe()["embedding"]])


if __name__ == "__main__":
    unittest.main()
