"""Contrat entre l'application et le service IA, vérifié sans réseau."""

import unittest

from assistant.application.errors import AIServiceError
from assistant.application.ports import GenerationRequest
from assistant.infrastructure.http_ai_client import HttpEmbedder, HttpGenerator


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.response


class HttpEmbedderContractTest(unittest.TestCase):
    def test_query_request_and_response(self):
        transport = RecordingTransport({"model": "ollama:nomic@abc", "dimension": 2, "vectors": [[0.1, 0.2]]})
        batch = HttpEmbedder("http://ia:8100/", "nomic", transport=transport).embed_query("bonjour")

        url, payload = transport.calls[0]
        self.assertEqual(url, "http://ia:8100/v1/embeddings")
        self.assertEqual(payload, {"model": "nomic", "input_type": "query", "inputs": ["bonjour"]})
        self.assertEqual((batch.model, batch.dimension), ("ollama:nomic@abc", 2))

    def test_documents_are_sent_as_documents(self):
        transport = RecordingTransport({"model": "m", "dimension": 1, "vectors": [[1], [2]]})
        HttpEmbedder("http://ia", "m", transport=transport).embed_documents(["a", "b"])
        self.assertEqual(transport.calls[0][1]["input_type"], "document")

    def test_incomplete_response_is_an_error(self):
        transport = RecordingTransport({"vectors": [[1.0]]})
        with self.assertRaises(AIServiceError):
            HttpEmbedder("http://ia", "m", transport=transport).embed_query("x")


class HttpGeneratorContractTest(unittest.TestCase):
    def test_request_and_response(self):
        transport = RecordingTransport({"model": "ollama:qwen3:4b", "text": "Réponse [1]"})
        generation = HttpGenerator("http://ia", "qwen3-4b", transport=transport).generate(
            GenerationRequest(system="s", prompt="p", temperature=0.2, max_tokens=100, seed=7))

        url, payload = transport.calls[0]
        self.assertEqual(url, "http://ia/v1/generate")
        self.assertEqual(payload, {"model": "qwen3-4b", "system": "s", "prompt": "p",
                                   "temperature": 0.2, "max_tokens": 100, "seed": 7})
        self.assertEqual(generation.text, "Réponse [1]")


if __name__ == "__main__":
    unittest.main()
