"""API HTTP de l'application (adaptateur entrant, bibliothèque standard).

    GET  /health
    GET  /v1/status                     -> l'index est-il cohérent avec le corpus et le modèle servi ?
    POST /v1/index                      -> reconstruit l'index
    POST /v1/ask  {"user": "alice", "question": "..."}
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from assistant.application.errors import (
    AIServiceError,
    IndexModelMismatchError,
    IndexNotBuiltError,
)
from assistant.composition import Container, UnknownUserError
from assistant.domain.errors import DomainError

from .presenter import answer_to_dict, manifest_to_dict, status_to_dict


def make_handler(container: Container, quiet: bool = False):
    index_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "fyc-assistant/1"

        def do_GET(self):
            try:
                if self.path == "/health":
                    manifest = container.index.manifest()
                    self._send(200, {
                        "status": "ok",
                        "index": manifest_to_dict(manifest) if manifest else None,
                    })
                elif self.path == "/v1/status":
                    report = container.check_status.execute()
                    status = 200 if report.up_to_date else (503 if report.unverified else 409)
                    self._send(status, status_to_dict(report))
                else:
                    self._send(404, {"error": {"code": "not_found", "message": self.path}})
            except (ValueError, OSError) as error:   # corpus mal formé, prompt ou index illisible
                self._error(500, "unreadable_state", str(error))

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == "/v1/index":
                    with index_lock:
                        manifest = container.index_corpus.execute()
                    return self._send(200, manifest_to_dict(manifest))
                if self.path == "/v1/ask":
                    user = container.config.user(str(payload.get("user", "")))
                    answer = container.ask_question.execute(user, str(payload.get("question", "")))
                    return self._send(200, answer_to_dict(answer))
                self._send(404, {"error": {"code": "not_found", "message": self.path}})
            except json.JSONDecodeError as error:
                self._error(400, "invalid_json", str(error))
            except UnknownUserError as error:
                self._error(403, "unknown_user", str(error.args[0]))
            except DomainError as error:
                self._error(400, "invalid_question", str(error))
            except (IndexNotBuiltError, IndexModelMismatchError) as error:
                self._error(409, "index_unusable", str(error))
            except AIServiceError as error:
                self._error(502, "ai_service_error", str(error))

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
                sys.stderr.write(f"[application] {self.address_string()} {fmt % args}\n")

    return Handler


def create_server(container: Container, host: str, port: int,
                  quiet: bool = False) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(container, quiet))
