"""API HTTP de l'application (adaptateur entrant, bibliothèque standard).

    GET  /health
    GET  /v1/status                     -> l'index est-il cohérent avec le corpus et le modèle servi ?
    POST /v1/index                      -> reconstruit l'index
    POST /v1/ask  {"user": "alice", "question": "..."}

Une réponse JSON à toute requête lue (liste des codes dans le README, « API HTTP de l'application ») :
route inconnue 404 ; autre méthode 405, avec l'en-tête Allow (HEAD est permis partout où GET l'est, sans
corps) ; corps qui n'est pas un objet JSON strict 400 (invalid_json) ; requête mal formée 400
(invalid_request : Content-Length illisible, envoi en morceaux, corps plus court que son Content-Length…) ;
question vide 400 ; utilisateur inconnu 403 ; index absent, d'un autre modèle ou remplacé pendant la
question 409 ; corps de plus de 16 Mio 413 ; ligne de requête trop longue 414, en-têtes trop longs ou trop
nombreux 431, version HTTP non prise en charge 505 (invalid_request, refusés par la bibliothèque standard) ;
état illisible, index impossible à écrire ou erreur imprévue 500 ; service IA en échec 502.
GET /v1/status rend le rapport de status en corps (pas {"error": …}) : 200 à jour, 409 à refaire, 503 non
vérifié (le service IA a échoué) ; ses autres codes sont ceux de la liste (500 pour un état illisible…).
L'utilisateur est celui que déclare l'appelant : pas d'authentification (voir le README).
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import traceback
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit

from assistant.application.errors import (
    AIServiceError,
    ApplicationError,
    IndexModelMismatchError,
    IndexNotBuiltError,
    IndexReplacedError,
    IndexWriteError,
)
from assistant.composition import Container, UnknownUserError, read_json_text
from assistant.domain.errors import DomainError

from .presenter import answer_to_dict, manifest_to_dict, status_to_dict

_ROUTES = {"/health": "GET", "/v1/status": "GET", "/v1/index": "POST", "/v1/ask": "POST"}
_ALLOWED = {"GET": ("GET", "HEAD"), "POST": ("POST",)}   # HEAD permis partout où GET l'est (RFC 9110)
_MAX_BODY = 16 * 1024 * 1024           # corps lu au plus (16 Mio) : au-delà, 413
_TOO_LARGE = f"corps de requête trop volumineux : {_MAX_BODY} octets au plus (16 Mio)"
_PIECE = 64 * 1024                     # le corps est lu par morceaux de 64 Kio
# Content-Length : chiffres seuls, zéros devant permis. Le groupe commence par un chiffre non nul, ou vaut
# « 0 » : avec « 0*([0-9]+) », un refus reviendrait en arrière zéro par zéro, en temps quadratique (une
# ligne d'en-tête de 64 Kio occuperait un fil plus de dix secondes).
_DECIMAL = re.compile("0*([1-9][0-9]*|0)")
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")   # moitié de paire UTF-16 : pas du texte


class InvalidRequestError(ValueError):
    """Requête illisible ou mal formée, ou corps trop volumineux : la faute est chez l'appelant (400, ou 413)."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class _ClientTimeout(TimeoutError):
    """Le client n'envoie plus rien (délai d'inactivité). C'est un TimeoutError : la bibliothèque standard
    l'attrape comme tel et abandonne la connexion, sans réponse. Un délai dépassé ailleurs (service IA…)
    n'est pas celui du client : il reçoit sa réponse, comme toute autre erreur."""


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
    if value is None:   # absent ou null : une chaîne vide (utilisateur inconnu, question vide)
        return ""
    if not isinstance(value, str):
        raise InvalidRequestError("invalid_request", f"le champ « {name} » doit être une chaîne")
    return value


def _parse_json(raw: bytes) -> dict[str, Any]:
    """Le corps d'un POST : un objet JSON strict, sinon 400 invalid_json. Lu par le contrôleur des fichiers
    (read_json_text, de la racine de composition), qui refuse ce que json.loads accepterait : clé en
    double, NaN ou Infinity, « \\ud800 » isolé (la réponse qui le reprendrait ne pourrait pas être
    encodée) ; plus de 900 niveaux et un entier de plus de 4300 chiffres sont refusés aussi, en français."""
    try:
        # utf-8-sig : un corps précédé d'une marque d'ordre des octets est accepté, comme un fichier (text_files).
        payload = read_json_text(raw.decode("utf-8-sig") or "{}")
    except ValueError as error:   # UnicodeDecodeError, JSONDecodeError, clé en double… sont des ValueError
        raise InvalidRequestError("invalid_json", str(error)) from error
    if not isinstance(payload, dict):
        raise InvalidRequestError("invalid_json", "le corps doit être un objet JSON")
    return payload


def _size(digits: str) -> int:
    """Un Content-Length en octets. Au-delà de 19 chiffres, c'est plus que tout ce qu'un client enverra :
    10**19 suffit à le dire (et int() refuserait de convertir plus de 4300 chiffres)."""
    return int(digits) if len(digits) <= 19 else 10**19


def _read(pieces: Iterator[bytes], size: int) -> bytes:
    """Le corps à traiter, 16 Mio au plus : au-delà, 413, sans rien lire. Plus court que son Content-Length
    (le client a fermé son envoi avant la fin) : 400, plutôt que de traiter un corps tronqué. Seule la
    connexion du client est lue ici : un délai dépassé y est le sien (_ClientTimeout)."""
    if size > _MAX_BODY:
        raise InvalidRequestError("payload_too_large", _TOO_LARGE, 413)
    try:
        body = b"".join(pieces)
    except TimeoutError as error:
        raise _ClientTimeout(*error.args) from error
    if len(body) < size:
        raise InvalidRequestError("invalid_request", _incomplete(len(body), size))
    return body


def _incomplete(received: int, size: int) -> str:
    return f"corps de requête incomplet : {received} octets reçus sur {size} annoncés (Content-Length)"


def make_handler(container: Container, quiet: bool = False):
    index_lock = ReadWriteLock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "fyc-assistant/1"
        # Version supposée tant que la ligne de requête n'est pas lue : avec HTTP/0.9 (défaut de la
        # bibliothèque standard), le refus d'une requête « HTTP/2.0 » partirait sans ligne de statut.
        default_request_version = "HTTP/1.0"
        # Délai d'inactivité : un client qui n'envoie plus rien (requête, ou corps annoncé et jamais envoyé)
        # ne retient pas un fil plus de 30 s. La bibliothèque standard abandonne alors la connexion, sans réponse.
        timeout = 30

        def _handle(self) -> None:
            """Toutes les méthodes passent par ici, même celles que la bibliothèque standard ne
            connaît pas (voir send_error) : route inconnue → 404, autre méthode → 405 avec Allow."""
            path = urlsplit(self.path).path
            method = _ROUTES.get(path)
            pieces = self._until_closed()   # tant que la fin du corps est inconnue
            try:
                pieces, size = self._body()
                # Seul un POST sur sa route lit le corps (16 Mio au plus) ; ailleurs, il est jeté.
                if not (method == self.command == "POST" and size <= _MAX_BODY):
                    pieces = self._pieces(self._arrived(size))
                if method is None:
                    status, body = 404, _error_body("not_found", path)
                elif self.command not in _ALLOWED[method]:
                    status, body = 405, _error_body("method_not_allowed", self.command)
                elif method == "GET":   # ou HEAD : mêmes en-têtes, sans corps (voir _send)
                    status, body = self._get(path)
                else:
                    status, body = self._post(path, _parse_json(_read(pieces, size)))
            except _ClientTimeout:   # corps qui n'arrive plus : connexion abandonnée, sans réponse
                raise
            except InvalidRequestError as error:
                status, body = error.status, _error_body(error.code, str(error))
            except UnknownUserError as error:
                status, body = 403, _error_body("unknown_user", str(error.args[0]))
            except DomainError as error:
                status, body = 400, _error_body("invalid_question", str(error))
            except (IndexNotBuiltError, IndexModelMismatchError, IndexReplacedError) as error:
                status, body = 409, _error_body("index_unusable", str(error))
            except AIServiceError as error:
                status, body = 502, _error_body("ai_service_error", str(error))
            except IndexWriteError as error:   # l'index en service reste le précédent
                status, body = 500, _error_body("index_write_failed", str(error))
            except (ApplicationError, ValueError, OSError) as error:
                # corpus mal formé ou vide, prompt ou index illisible…
                status, body = 500, _error_body("unreadable_state", str(error))
            except Exception as error:  # noqa: BLE001 — jamais de connexion coupée sans réponse
                traceback.print_exc()
                status, body = 500, _error_body("internal_error", f"{type(error).__name__}: {error}")
            self._send(status, body, {"Allow": ", ".join(_ALLOWED[method])} if status == 405 else None)
            self._drain(pieces)

        do_GET = do_HEAD = do_POST = _handle

        def _get(self, path: str) -> tuple[int, dict[str, Any]]:
            if path == "/health":
                manifest = container.index.manifest()
                return 200, {"status": "ok", "index": manifest_to_dict(manifest) if manifest else None}
            report = container.check_status.execute()   # /v1/status, l'autre route GET
            return (200 if report.up_to_date else (503 if report.unverified else 409)), status_to_dict(report)

        def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            if path == "/v1/index":
                with index_lock.writing():
                    return 200, manifest_to_dict(container.index_corpus.execute())
            user = container.config.user(_text_field(payload, "user"))   # /v1/ask, l'autre route POST
            question = _text_field(payload, "question")
            with index_lock.reading():
                return 200, answer_to_dict(container.ask_question.execute(user, question))

        def _body(self) -> tuple[Iterator[bytes], int]:
            """Le corps, lu à la demande, morceau par morceau : un Content-Length démesuré n'alloue
            rien d'avance. Avec sa taille, que Content-Length annonce (0 sans lui). Un envoi en morceaux
            (Transfer-Encoding) n'est pas lu : refusé, en disant quoi envoyer à la place."""
            # Répété à l'identique (espaces autour mis à part), il est fondu en un ; deux valeurs différentes
            # sont refusées plutôt que de garder la première.
            if len({value.strip(" \t") for value in self.headers.get_all("Content-Length", [])}) > 1:
                raise InvalidRequestError("invalid_request", "Content-Length en double")
            if "Transfer-Encoding" in self.headers:
                raise InvalidRequestError(
                    "invalid_request", "Content-Length obligatoire : l'application ne lit pas les envois en morceaux "
                    f"(Transfer-Encoding: {self.headers['Transfer-Encoding']})")
            length = _DECIMAL.fullmatch(self.headers.get("Content-Length", "0").strip(" \t"))
            if length is None:   # int() accepterait « -1 », « +1 » ou « 1_0 »
                raise InvalidRequestError("invalid_request", "Content-Length invalide")
            size = _size(length[1])
            return self._pieces(size), size

        def _pieces(self, length: int) -> Iterator[bytes]:
            while length > 0 and (piece := self.rfile.read(min(length, _PIECE))):
                length -= len(piece)
                yield piece

        def _until_closed(self) -> Iterator[bytes]:
            """Tout ce qui arrive jusqu'à ce que le client ferme : un corps dont on ignore la fin."""
            return iter(lambda: self.rfile.read(_PIECE), b"")

        def _arrived(self, size: int) -> int:
            """Un corps jeté (route inconnue, autre méthode, GET, plus de 16 Mio annoncés) : ce qui en est déjà
            arrivé est lu, sans attendre, 64 Kio au plus. Renvoie ce qui reste à recevoir, lu après la réponse
            (_drain). Si le client a déjà fermé son envoi avant la fin annoncée, 400. Plus loin que ce qui est
            arrivé, rien n'est vu : la réponse est celle du corps jeté (404, 405, 413)."""
            budget, received = min(size, _PIECE), 0
            if budget:
                self.connection.settimeout(0)   # sans attendre : None quand rien d'autre n'est arrivé
                try:
                    while received < budget and (piece := self.rfile.read(budget - received)) is not None:
                        if not piece:   # envoi fermé avant la fin annoncée
                            raise InvalidRequestError("invalid_request", _incomplete(received, size))
                        received += len(piece)
                finally:
                    self.connection.settimeout(self.timeout)
            return size - received

        def _drain(self, pieces: Iterator[bytes]) -> None:
            """Après la réponse : la fin de l'envoi part (le client sait que la réponse est complète), puis ce qui
            reste du corps, que personne ne lit (route inconnue, autre méthode, GET, corps refusé…), est lu
            jusqu'au bout. Fermée sur des octets non lus, la connexion serait réinitialisée, et le client, qui
            envoie souvent tout son corps avant de lire, pourrait perdre la réponse (RFC 9112, § 9.6)."""
            with suppress(OSError):   # client parti, ou muet (délai d'inactivité)
                self.connection.shutdown(socket.SHUT_WR)
                for _ in pieces:
                    pass

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, _error_body(code, message))

        def send_error(self, code, message=None, explain=None):
            """Ce que la bibliothèque standard refuse elle-même répond aussi en JSON. Une méthode
            qu'elle ne connaît pas (501 : PUT, TRACE…) suit le chemin commun : 404 ou 405. Le reste
            (ligne de requête ou en-têtes trop longs : 414, 431 ; version : 400, 505) est la faute du client."""
            if code == 501:
                return self._handle()
            self._error(code, "invalid_request", message or self.responses.get(code, ("",))[0])

        def _send(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            # Un surrogate isolé (venu du service IA…) ne s'encode pas en UTF-8 : remplacé par U+FFFD,
            # plutôt qu'une connexion coupée sans réponse.
            data = _LONE_SURROGATE.sub("\ufffd", json.dumps(body, ensure_ascii=False)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":   # HEAD : les en-têtes seuls
                self.wfile.write(data)

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(f"[application] {self.address_string()} {fmt % args}\n")

    return Handler


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


class _Server(ThreadingHTTPServer):
    # Sous Windows, SO_REUSEADDR (que http.server active) laisse un second serveur écouter un port déjà
    # pris, sans erreur ; ailleurs, il permet seulement de relancer aussitôt sur un port en TIME_WAIT.
    allow_reuse_address = sys.platform != "win32"


def create_server(container: Container, host: str, port: int,
                  quiet: bool = False) -> ThreadingHTTPServer:
    return _Server((host, port), make_handler(container, quiet))
