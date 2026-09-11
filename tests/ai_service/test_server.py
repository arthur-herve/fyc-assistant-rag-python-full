import json
import threading
import unittest
import urllib.error
import urllib.request

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
            return error.code, json.loads(error.read())

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

    def test_models_listing(self):
        with urllib.request.urlopen(self.url + "/v1/models", timeout=5) as response:
            body = json.loads(response.read())
        self.assertEqual([m["alias"] for m in body["generation"]], ["extractive"])


if __name__ == "__main__":
    unittest.main()
