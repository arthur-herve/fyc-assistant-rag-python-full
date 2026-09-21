"""API HTTP de l'application (adaptateur entrant, bibliothèque standard).

    GET  /health
    GET  /v1/status                     -> l'index est-il cohérent avec le corpus et le modèle servi ?
    POST /v1/index                      -> reconstruit l'index
    POST /v1/ask  {"user": "alice", "question": "..."}

Toujours une réponse JSON, avec les mêmes codes que la version C# (liste dans le README,
« API HTTP de l'application »). L'utilisateur est celui que déclare l'appelant : pas
d'authentification (voir le README).
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit

from assistant.application.errors import (
    AIServiceError,
    ApplicationError,
    IndexModelMismatchError,
    IndexNotBuiltError,
    IndexReplacedError,
)
from assistant.composition import Container, UnknownUserError
from assistant.domain.errors import DomainError

from .presenter import answer_to_dict, manifest_to_dict, status_to_dict


class InvalidRequestError(ValueError):
    """Corps de requête illisible ou mal formé : la faute est chez l'appelant (400)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ReadWriteLock:
    """Plusieurs questions en parallèle, mais une réindexation seule : une question ne voit
    jamais l'index changer entre le contrôle du modèle et la recherche. Une réindexation qui
    attend passe avant les nouvelles questions : un flux continu ne la repousse pas sans fin."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writing = False
        self._writers_waiting = 0

    @contextmanager
    def reading(self) -> Iterator[None]:
        with self._condition:
            while self._writing or self._writers_waiting:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    @contextmanager
    def writing(self) -> Iterator[None]:
        with self._condition:
            self._writers_waiting += 1
            while self._writing or self._readers:
                self._condition.wait()
            self._writers_waiting -= 1
            self._writing = True
        try:
            yield
        finally:
            with self._condition:
                self._writing = False
                self._condition.notify_all()


def _text_field(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if value is None:   # absent ou null : comme la version C#
        return ""
    if not isinstance(value, str):
        raise InvalidRequestError("invalid_request", f"le champ « {name} » doit être une chaîne")
    return value


def make_handler(container: Container, quiet: bool = False):
    index_lock = ReadWriteLock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "fyc-assistant/1"

        def do_GET(self):
            self._handle(self._get)

        def do_POST(self):
            self._handle(self._post)

        def _method_not_allowed(self):
            self._drain()
            self._error(405, "method_not_allowed", self.command)

        def _drain(self) -> None:
            """Lire le corps avant de répondre : fermer une connexion sur des octets non lus peut la
            réinitialiser (Windows), et le client ne verrait jamais la réponse."""
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                length = 0
            if length > 0:
                self.rfile.read(length)

        do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed   # 405 en JSON, comme en C#

        def _handle(self, route) -> None:
            path = urlsplit(self.path).path
            try:
                status, body = route(path)
            except InvalidRequestError as error:
                status, body = 400, _error_body(error.code, str(error))
            except UnknownUserError as error:
                status, body = 403, _error_body("unknown_user", str(error.args[0]))
            except DomainError as error:
                status, body = 400, _error_body("invalid_question", str(error))
            except (IndexNotBuiltError, IndexModelMismatchError, IndexReplacedError) as error:
                status, body = 409, _error_body("index_unusable", str(error))
            except AIServiceError as error:
                status, body = 502, _error_body("ai_service_error", str(error))
            except (ApplicationError, ValueError, OSError) as error:
                # corpus mal formé ou vide, prompt ou index illisible…
                status, body = 500, _error_body("unreadable_state", str(error))
            except Exception as error:  # noqa: BLE001 — jamais de connexion coupée sans réponse
                traceback.print_exc()
                status, body = 500, _error_body("internal_error", f"{type(error).__name__}: {error}")
            self._send(status, body)

        def _get(self, path: str) -> tuple[int, dict[str, Any]]:
            if path == "/health":
                manifest = container.index.manifest()
                return 200, {"status": "ok", "index": manifest_to_dict(manifest) if manifest else None}
            if path == "/v1/status":
                report = container.check_status.execute()
                return (200 if report.up_to_date else (503 if report.unverified else 409)), status_to_dict(report)
            return 404, _error_body("not_found", path)

        def _post(self, path: str) -> tuple[int, dict[str, Any]]:
            payload = self._read_json()
            if path == "/v1/index":
                with index_lock.writing():
                    return 200, manifest_to_dict(container.index_corpus.execute())
            if path == "/v1/ask":
                user = container.config.user(_text_field(payload, "user"))
                question = _text_field(payload, "question")
                with index_lock.reading():
                    return 200, answer_to_dict(container.ask_question.execute(user, question))
            return 404, _error_body("not_found", path)

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0:   # read(-1) attendrait la fin de la connexion : jamais de réponse
                raise InvalidRequestError("invalid_request", "Content-Length invalide")
            try:
                # utf-8-sig : un corps précédé d'une marque d'ordre des octets est accepté, comme en C#.
                payload = json.loads(self.rfile.read(length).decode("utf-8-sig") or "{}")
            except (ValueError, UnicodeDecodeError) as error:   # JSONDecodeError est une ValueError
                raise InvalidRequestError("invalid_json", str(error)) from error
            if not isinstance(payload, dict):
                raise InvalidRequestError("invalid_json", "le corps doit être un objet JSON")
            return payload

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, _error_body(code, message))

        def send_error(self, code, message=None, explain=None):
            """Ce que la bibliothèque standard refuse elle-même (méthode inconnue, requête
            mal formée) répond aussi en JSON. Une méthode inconnue est un 405, comme en C#."""
            self.close_connection = True
            if code == 501:
                self._error(405, "method_not_allowed", self.command or "")
            else:
                self._error(code, "invalid_request" if 400 <= code < 500 else "internal_error",
                            message or self.responses.get(code, ("",))[0])

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":   # HEAD : les en-têtes seuls
                self.wfile.write(data)

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(f"[application] {self.address_string()} {fmt % args}\n")

    return Handler


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def create_server(container: Container, host: str, port: int,
                  quiet: bool = False) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(container, quiet))
