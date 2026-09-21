"""Contrat entre l'application et le service IA, vérifié sans réseau."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    def test_vectors_that_do_not_match_the_announced_dimension_are_an_error(self):
        for vectors in ([[0.1]], [[0.1, 0.2], [0.3, 0.4]]):   # trop courts ; trop nombreux
            with self.subTest(vectors=vectors):
                transport = RecordingTransport({"model": "m", "dimension": 2, "vectors": vectors})
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder("http://ia", "m", transport=transport).embed_query("x")
                self.assertFalse(caught.exception.transient)

    def test_a_502_marked_not_retryable_is_not_transient(self):
        """Le service dit « réessayer ne changera rien » (modèle absent) : pas de nouvelle tentative."""
        class Refusing(BaseHTTPRequestHandler):
            retryable = True

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                data = json.dumps({"error": {"code": "backend_error", "message": "x",
                                             "retryable": Refusing.retryable}}).encode()
                self.send_response(502)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Refusing)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            for retryable in (True, False):
                with self.subTest(retryable=retryable):
                    Refusing.retryable = retryable
                    with self.assertRaises(AIServiceError) as caught:
                        HttpEmbedder(url, "m", timeout=5).embed_query("x")
                    self.assertEqual(caught.exception.transient, retryable)
        finally:
            server.shutdown()
            server.server_close()

    def test_a_response_that_is_not_json_is_an_error(self):
        class NotJson(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                data = b"<html>proxy</html>"
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), NotJson)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(AIServiceError) as caught:
                HttpEmbedder(f"http://127.0.0.1:{server.server_address[1]}", "m", timeout=5).embed_query("x")
            self.assertIn("illisible", str(caught.exception))
        finally:
            server.shutdown()
            server.server_close()


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
