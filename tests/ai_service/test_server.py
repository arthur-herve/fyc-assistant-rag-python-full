import http.client
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_service.registry import ModelRegistry
from ai_service.server import create_server

CONFIG = {
    "embedding": {"hashing": {"backend": "hashing", "dimension": 32},
                  "ollama-absent": {"backend": "ollama", "model": "x",
                                    "base_url": "http://127.0.0.1:1", "timeout": 2}},
    "generation": {"extractive": {"backend": "extractive"}},
}


class AIServiceServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(ModelRegistry.from_dict(CONFIG), "127.0.0.1", 0, quiet=True)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, payload):
        request = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_other_methods_get_a_json_405(self):
        for method in ("PUT", "OPTIONS", "TRACE"):
            request = urllib.request.Request(self.url + "/v1/embeddings", method=method)
            with self.subTest(method), self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            with caught.exception as error:
                self.assertEqual((error.code, json.loads(error.read())["error"]["code"]),
                                 (405, "method_not_allowed"))

    def test_a_negative_content_length_gets_an_answer(self):
        """read(-1) attendrait la fin de la connexion : le fil resterait bloqué, sans réponse."""
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        try:
            connection.putrequest("POST", "/v1/embeddings")
            connection.putheader("Content-Length", "-1")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual((response.status, json.loads(response.read())["error"]["code"]),
                             (400, "invalid_request"))
        finally:
            connection.close()

    def test_embeddings(self):
        status, body = self.post("/v1/embeddings", {"model": "hashing", "input_type": "query", "inputs": ["a", "b"]})
        self.assertEqual(status, 200)
        self.assertEqual((body["model"], body["alias"], body["dimension"], len(body["vectors"])),
                         ("hashing-32-stem6", "hashing", 32, 2))

    def test_generate(self):
        status, body = self.post("/v1/generate", {"model": "extractive", "prompt": "rien", "seed": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "extractive")

    def test_validation_errors(self):
        cases = [
            ({"model": "hashing", "inputs": []}, "/v1/embeddings"),
            ({"model": "hashing", "inputs": ["a"], "input_type": "autre"}, "/v1/embeddings"),
            ({"model": "extractive"}, "/v1/generate"),
            ({"model": "extractive", "prompt": "p", "temperature": 5}, "/v1/generate"),
            ({"model": "extractive", "prompt": "p", "max_tokens": True}, "/v1/generate"),
        ]
        for payload, path in cases:
            with self.subTest(payload=payload):
                status, body = self.post(path, payload)
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["code"], "invalid_request")

    def test_unknown_model_is_404(self):
        status, body = self.post("/v1/generate", {"model": "inconnu", "prompt": "p"})
        self.assertEqual((status, body["error"]["code"]), (404, "unknown_model"))

    def test_backend_failure_is_502(self):
        status, body = self.post("/v1/embeddings", {"model": "ollama-absent", "inputs": ["a"]})
        self.assertEqual((status, body["error"]["code"]), (502, "backend_error"))
        self.assertTrue(body["error"]["retryable"])   # Ollama injoignable : peut revenir

    def test_an_engine_answer_that_is_not_json_is_502_not_400(self):
        """La faute est chez le moteur, pas chez le client : jamais « votre requête est invalide »."""
        class Html(BaseHTTPRequestHandler):
            def do_GET(self):
                self._html()

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self._html()

            def _html(self):
                data = b"<html>proxy</html>"
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        engine = ThreadingHTTPServer(("127.0.0.1", 0), Html)
        threading.Thread(target=engine.serve_forever, daemon=True).start()
        service = create_server(ModelRegistry.from_dict({"embedding": {"o": {
            "backend": "openai-compatible", "model": "m", "base_url": f"http://127.0.0.1:{engine.server_address[1]}"}}}),
            "127.0.0.1", 0, quiet=True)
        threading.Thread(target=service.serve_forever, daemon=True).start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{service.server_address[1]}/v1/embeddings",
                data=json.dumps({"model": "o", "inputs": ["a"]}).encode(), method="POST")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            with caught.exception as error:
                body = json.loads(error.read())
            self.assertEqual((caught.exception.code, body["error"]["code"], body["error"]["retryable"]),
                             (502, "backend_error", False))
        finally:
            for server in (service, engine):
                server.shutdown()
                server.server_close()

    def test_models_listing(self):
        with urllib.request.urlopen(self.url + "/v1/models", timeout=5) as response:
            body = json.loads(response.read())
        self.assertEqual([m["alias"] for m in body["generation"]], ["extractive"])


if __name__ == "__main__":
    unittest.main()
