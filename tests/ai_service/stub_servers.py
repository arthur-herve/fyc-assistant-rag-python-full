"""Faux serveurs Ollama et « compatible OpenAI » : vérifient le format des requêtes."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubServer:
    def __init__(self, routes):
        self.requests: list[tuple[str, str, dict | None]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self, method):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else None
                stub.requests.append((method, self.path, body))
                route = routes.get((method, self.path))
                status, payload = route(body) if route else (404, {"error": "not found"})
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def ollama_routes(thinking: bool = False):
    """`thinking=True` imite un modèle à réflexion : Ollama renvoie alors le
    raisonnement dans `message.thinking`, à côté de `message.content`."""
    message = {"role": "assistant", "content": "<think>je réfléchis</think>\nDeux jours [1]."}
    if thinking:
        message = {"role": "assistant", "thinking": "je réfléchis longuement",
                   "content": "Deux jours [1]."}
    return {
        ("GET", "/api/tags"): lambda body: (200, {"models": [
            {"name": "nomic-embed-text:latest", "digest": "0a109f422b47e3a3"},
            {"name": "qwen3:1.7b", "digest": "8f68893c685ceaa1"},
        ]}),
        ("POST", "/api/embed"): lambda body: (200, {
            "model": body["model"],
            "embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]],
        }),
        ("POST", "/api/chat"): lambda body: (200, {
            "model": body["model"],
            "message": {"role": "assistant",
                        "content": "<think>je réfléchis</think>\nDeux jours [1]."},
            "done": True,
        }),
    }


def openai_routes():
    return {
        ("POST", "/v1/embeddings"): lambda body: (200, {"data": [
            {"index": i, "embedding": [float(i), 1.0]} for i, _ in reversed(list(enumerate(body["input"])))
        ]}),
        ("POST", "/v1/chat/completions"): lambda body: (200, {
            "choices": [{"message": {"role": "assistant", "content": " Réponse [1] "}}]
        }),
    }
