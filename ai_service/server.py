"""Serveur HTTP du service IA (bibliothèque standard uniquement).

Ce processus est destiné à tourner sur la machine qui dispose des ressources
de calcul. Il ne connaît rien au métier : ni documents, ni droits, ni citations.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import time
import traceback
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, NoReturn

from .backends.base import BackendError
from .registry import ModelRegistry, UnknownModelError

MAX_INPUTS = 256
CONTRACT_VERSION = "1"
ROUTES = {"/health": "GET", "/v1/models": "GET", "/v1/embeddings": "POST", "/v1/generate": "POST"}
ALLOWED = {"GET": ("GET", "HEAD"), "POST": ("POST",)}   # HEAD permis partout où GET l'est (RFC 9110)
MAX_DEPTH = 64                        # niveaux d'imbrication d'un corps JSON, au plus
MAX_DIGITS = 4300                     # chiffres d'un entier JSON : au-delà, int() refuse de le convertir
MAX_BODY = 16 * 1024 * 1024           # corps lu au plus (16 Mio) : au-delà, 413 sans le lire
_PIECE = 64 * 1024                    # le corps est lu par morceaux de 64 Kio
# Content-Length : chiffres seuls, zéros devant permis. Le groupe commence par un chiffre non nul, ou vaut
# « 0 » : avec « 0*([0-9]+) », un refus reviendrait en arrière zéro par zéro, en temps quadratique (une
# ligne d'en-tête de 64 Kio occuperait un fil plus de dix secondes).
_DECIMAL = re.compile("0*([1-9][0-9]*|0)")
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")   # moitié de paire UTF-16 : pas du texte


class RequestError(ValueError):
    pass


def _size(digits: str) -> int:
    """Un Content-Length en octets. Au-delà de 19 chiffres, c'est plus que tout ce qu'un client enverra :
    10**19 suffit à le dire (et int() refuserait de convertir plus de 4300 chiffres)."""
    return int(digits) if len(digits) <= 19 else 10**19


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Une clé en double est refusée : sinon la dernière valeur l'emporterait sans rien dire."""
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RequestError(f"clé « {key} » en double")
        payload[key] = value
    return payload


def _not_json(constant: str) -> NoReturn:
    """NaN, Infinity et -Infinity : json.loads les accepte, mais ce n'est pas du JSON (RFC 8259)."""
    raise RequestError(f"« {constant} » n'est pas du JSON")


def _integer(text: str) -> int:
    """Au-delà de 4300 chiffres, int() refuse de convertir : refus explicite, plutôt que son message."""
    if len(text.lstrip("-")) > MAX_DIGITS:
        raise RequestError(f"nombre entier de plus de {MAX_DIGITS} chiffres")
    return int(text)


def _check_json(value: Any, depth: int = 1) -> None:
    """Contrôles faits après json.loads : un « \\ud800 » isolé (du JSON valide, mais pas du texte :
    la réponse qui le reprendrait ne pourrait pas être encodée) et plus de 64 niveaux d'imbrication."""
    if isinstance(value, str):
        if _LONE_SURROGATE.search(value):
            raise RequestError("chaîne qui n'est pas du texte : surrogate UTF-16 isolé (\\ud800 à \\udfff sans sa paire)")
    elif isinstance(value, (dict, list)):
        if depth > MAX_DEPTH:
            raise RequestError(f"JSON trop imbriqué : plus de {MAX_DEPTH} niveaux")
        for item in [*value, *value.values()] if isinstance(value, dict) else value:
            _check_json(item, depth + 1)


def parse_json(body: bytes) -> dict[str, Any]:
    """Le corps d'une requête POST : un objet JSON strict, sinon RequestError (400) : ni clé en double,
    ni NaN ou Infinity (ce n'est pas du JSON), ni « \\ud800 » isolé, ni plus de 64 niveaux ; ni
    entier de plus de 4300 chiffres."""
    try:
        payload = json.loads(body.decode("utf-8") or "{}", object_pairs_hook=_unique_keys,
                             parse_constant=_not_json, parse_int=_integer)
    except RecursionError:   # des milliers de niveaux : json.loads abandonne avant _check_json
        raise RequestError(f"JSON trop imbriqué : plus de {MAX_DEPTH} niveaux") from None
    except ValueError as error:   # JSONDecodeError, UnicodeDecodeError… : la faute du client, jamais un 500
        raise RequestError(str(error)) from error
    if not isinstance(payload, dict):
        raise RequestError("un objet JSON est attendu")
    _check_json(payload)
    return payload


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
        # Version supposée tant que la ligne de requête n'est pas lue : avec HTTP/0.9 (défaut de la
        # bibliothèque standard), le refus d'une requête « HTTP/2.0 » partirait sans ligne de statut.
        default_request_version = "HTTP/1.0"
        # Délai d'inactivité : un client qui n'envoie plus rien (requête, ou corps annoncé et jamais envoyé)
        # ne retient pas un fil plus de 30 s. La bibliothèque standard abandonne alors la connexion, sans réponse.
        timeout = 30

        def _handle(self) -> None:
            """Toutes les méthodes passent par ici, même celles que la bibliothèque standard ne
            connaît pas (voir send_error) : route inconnue → 404, autre méthode → 405 avec Allow."""
            pieces = self._until_closed()   # tant que la fin du corps est inconnue
            try:
                pieces, size = self._body()
            except RequestError as error:   # Content-Length illisible ou en double, ou absent (envoi en morceaux)
                self._error(400, "invalid_request", str(error))
            else:
                self._answer(ROUTES.get(self.path), pieces, size)
            self._drain(pieces)

        do_GET = do_HEAD = do_POST = _handle

        def _answer(self, method: str | None, pieces: Iterator[bytes], size: int) -> None:
            if method is None:
                return self._error(404, "not_found", f"route inconnue : {self.path}")
            if self.command not in ALLOWED[method]:
                allow = ", ".join(ALLOWED[method])
                return self._error(405, "method_not_allowed",
                                   f"méthode {self.command} non permise sur {self.path} (permises : {allow})",
                                   headers={"Allow": allow})
            if self.path == "/health":   # GET ou HEAD : mêmes en-têtes, sans corps (voir _send)
                return self._send(200, {"status": "ok", "contract_version": CONTRACT_VERSION})
            if self.path == "/v1/models":
                return self._send(200, registry.describe())
            if size > MAX_BODY:   # refusé sans lire le corps
                return self._error(413, "payload_too_large",
                                   f"corps de requête trop volumineux : {MAX_BODY} octets au plus (16 Mio)")
            route = {"/v1/embeddings": handle_embeddings, "/v1/generate": handle_generate}[self.path]
            # Lu hors du try : si le corps n'arrive plus (délai d'inactivité), le TimeoutError va jusqu'à la
            # bibliothèque standard, qui abandonne la connexion sans réponse. Celui d'un moteur, levé dans le
            # try, reçoit une réponse, comme toute autre erreur.
            body = b"".join(pieces)
            try:
                self._send(200, route(registry, parse_json(body)))
            except RequestError as error:
                self._error(400, "invalid_request", str(error))
            except UnknownModelError as error:
                self._error(404, "unknown_model", str(error.args[0]))
            except BackendError as error:
                self._error(502, "backend_error", str(error), retryable=error.retryable)
            except Exception as error:  # noqa: BLE001 — on renvoie toujours du JSON
                traceback.print_exc()
                self._error(500, "internal_error", f"{type(error).__name__}: {error}")

        def _body(self) -> tuple[Iterator[bytes], int]:
            """Le corps, lu à la demande, morceau par morceau : un Content-Length démesuré n'alloue
            rien d'avance. Content-Length obligatoire : les clients du service l'envoient toujours.
            Avec sa taille annoncée."""
            # Répété à l'identique (espaces autour mis à part), il est fondu en un, comme le permet la RFC 9110
            # (§ 8.6) ; deux valeurs différentes sont refusées plutôt que de garder la première.
            if len({value.strip(" \t") for value in self.headers.get_all("Content-Length", [])}) > 1:
                raise RequestError("Content-Length en double")
            if "Transfer-Encoding" in self.headers:
                raise RequestError("Content-Length obligatoire : le service ne lit pas les envois en morceaux "
                                   f"(Transfer-Encoding: {self.headers['Transfer-Encoding']})")
            length = _DECIMAL.fullmatch(self.headers.get("Content-Length", "0").strip(" \t"))
            if length is None:   # int() accepterait « -1 », « +1 » ou « 1_0 »
                raise RequestError("Content-Length invalide")
            size = _size(length[1])
            return self._pieces(size), size

        def _pieces(self, length: int) -> Iterator[bytes]:
            while length > 0 and (piece := self.rfile.read(min(length, _PIECE))):
                length -= len(piece)
                yield piece

        def _until_closed(self) -> Iterator[bytes]:
            """Tout ce qui arrive jusqu'à ce que le client ferme : un corps dont on ignore la fin."""
            return iter(lambda: self.rfile.read(_PIECE), b"")

        def _drain(self, pieces: Iterator[bytes]) -> None:
            """Après la réponse : la fin de l'envoi part (le client sait que la réponse est complète), puis ce qui
            reste du corps, que personne ne lit (route inconnue, autre méthode, GET, corps refusé…), est lu
            jusqu'au bout. Fermée sur des octets non lus, la connexion serait réinitialisée, et le client, qui
            envoie souvent tout son corps avant de lire, pourrait perdre la réponse (RFC 9112, § 9.6)."""
            with suppress(OSError):   # client parti, ou muet (délai d'inactivité)
                self.connection.shutdown(socket.SHUT_WR)
                for _ in pieces:
                    pass

        def _error(self, status: int, code: str, message: str, headers: dict[str, str] | None = None,
                   **extra: Any) -> None:
            self._send(status, {"error": {"code": code, "message": message, **extra}}, headers)

        def send_error(self, code, message=None, explain=None):
            """Ce que la bibliothèque standard refuse elle-même répond aussi en JSON. Une méthode
            qu'elle ne connaît pas (501 : PUT, TRACE…) suit le chemin commun : 404 ou 405. Le reste
            (ligne de requête ou en-têtes trop longs : 414, 431 ; version : 400, 505) est la faute du client."""
            if code == 501:
                return self._handle()
            self._error(code, "invalid_request", message or self.responses.get(code, ("",))[0])

        def _send(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            # Un surrogate isolé venu d'un moteur ne s'encode pas en UTF-8 : remplacé par U+FFFD,
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
                sys.stderr.write(f"[service-ia] {self.address_string()} {fmt % args}\n")

    return Handler


class _Server(ThreadingHTTPServer):
    # Sous Windows, SO_REUSEADDR (que http.server active) laisse un second serveur écouter un port déjà
    # pris, sans erreur ; ailleurs, il permet seulement de relancer aussitôt sur un port en TIME_WAIT.
    allow_reuse_address = sys.platform != "win32"


def create_server(registry: ModelRegistry, host: str, port: int,
                  quiet: bool = False) -> ThreadingHTTPServer:
    return _Server((host, port), make_handler(registry, quiet))
