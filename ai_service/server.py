"""Serveur HTTP du service IA (bibliothèque standard uniquement).

Ce processus est destiné à tourner sur la machine qui dispose des ressources
de calcul. Il ne connaît rien au métier : ni documents, ni droits, ni citations.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends.base import BackendError
from .registry import ModelRegistry, UnknownModelError

MAX_INPUTS = 256
CONTRACT_VERSION = "1"


class RequestError(ValueError):
    pass


def _field(payload: dict[str, Any], name: str, kind, required: bool = True, default=None):
    if name not in payload or payload[name] is None:
        if required:
            raise RequestError(f"champ '{name}' obligatoire")
        return default
    value = payload[name]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise RequestError(f"champ '{name}' : type {kind.__name__} attendu")
    return value


def handle_embeddings(registry: ModelRegistry, payload: dict[str, Any]) -> dict[str, Any]:
    alias = _field(payload, "model", str)
    input_type = _field(payload, "input_type", str, required=False, default="document")
    if input_type not in ("query", "document"):
        raise RequestError("input_type doit valoir 'query' ou 'document'")
    inputs = _field(payload, "inputs", list)
    if not inputs or len(inputs) > MAX_INPUTS or not all(isinstance(t, str) for t in inputs):
        raise RequestError(f"inputs : liste de 1 à {MAX_INPUTS} chaînes attendue")
    model = registry.embedding(alias)
    start = time.perf_counter()
    result = model.embed(inputs, input_type)
    return {
        "model": result.model_id,
        "alias": alias,
        "dimension": result.dimension,
        "vectors": result.vectors,
        "duration_ms": round((time.perf_counter() - start) * 1000),
    }


def handle_generate(registry: ModelRegistry, payload: dict[str, Any]) -> dict[str, Any]:
    alias = _field(payload, "model", str)
    system = _field(payload, "system", str, required=False, default="")
    prompt = _field(payload, "prompt", str)
    temperature = _field(payload, "temperature", float, required=False, default=0.2)
    max_tokens = _field(payload, "max_tokens", int, required=False, default=400)
    seed = _field(payload, "seed", int, required=False, default=None)
    if not 0.0 <= temperature <= 2.0:
        raise RequestError("temperature doit être comprise entre 0 et 2")
    if not 1 <= max_tokens <= 8192:
        raise RequestError("max_tokens doit être compris entre 1 et 8192")
    model = registry.generation(alias)
    start = time.perf_counter()
    model_id, text = model.generate(system, prompt, temperature, max_tokens, seed)
    return {
        "model": model_id,
        "alias": alias,
        "text": text,
        "duration_ms": round((time.perf_counter() - start) * 1000),
    }


def make_handler(registry: ModelRegistry, quiet: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "fyc-ai-service/" + CONTRACT_VERSION

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "contract_version": CONTRACT_VERSION})
            elif self.path == "/v1/models":
                self._send(200, registry.describe())
            else:
                self._error(404, "not_found", f"route inconnue : {self.path}")

        def do_POST(self):
            routes = {"/v1/embeddings": handle_embeddings, "/v1/generate": handle_generate}
            route = routes.get(self.path)
            if route is None:
                return self._error(404, "not_found", f"route inconnue : {self.path}")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise RequestError("un objet JSON est attendu")
                self._send(200, route(registry, payload))
            except (RequestError, json.JSONDecodeError, UnicodeDecodeError) as error:
                self._error(400, "invalid_request", str(error))
            except UnknownModelError as error:
                self._error(404, "unknown_model", str(error.args[0]))
            except BackendError as error:
                self._error(502, "backend_error", str(error))
            except Exception as error:  # noqa: BLE001 — on renvoie toujours du JSON
                traceback.print_exc()
                self._error(500, "internal_error", f"{type(error).__name__}: {error}")

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": {"code": code, "message": message}})

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(f"[service-ia] {self.address_string()} {fmt % args}\n")

    return Handler


def create_server(registry: ModelRegistry, host: str, port: int,
                  quiet: bool = False) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(registry, quiet))
