import http.client
import io
import json
import os
import queue
import re
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from unittest import mock

from ai_service import __main__ as entry_point
from ai_service.backends.hashing import HashingEmbeddingBackend
from ai_service.registry import ModelRegistry
from ai_service.server import create_server

from .stub_servers import silent_after   # relatif : ce fichier est le même dans tests/ et tests_python/ (C#)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "embedding": {"hashing": {"backend": "hashing", "dimension": 32},
                  "ollama-absent": {"backend": "ollama", "model": "x",
                                    "base_url": "http://127.0.0.1:1", "timeout": 2}},
    "generation": {"extractive": {"backend": "extractive"}},
}
MAX_BODY = 16 * 1024 * 1024   # 16 Mio : au-delà, 413


def serve(server: ThreadingHTTPServer) -> ThreadingHTTPServer:
    # Arrêt vérifié toutes les 50 ms (0,5 s par défaut) : sinon shutdown() attend jusqu'à une demi-seconde.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


def post(url: str, payload) -> tuple[int, dict]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())


def read_all(sock: socket.socket) -> bytes:
    """Tout ce qui arrive jusqu'à ce que le service ferme la connexion (requête HTTP/1.0, corps non lu…)."""
    received = b""
    while chunk := sock.recv(65536):
        received += chunk
    return received


def read_response(stream: BinaryIO, head: bool = False) -> tuple[str, dict[str, str], bytes]:
    """La réponse lue dans `stream` (sock.makefile("rb")) : ligne de statut, en-têtes, et corps selon son
    Content-Length (aucun pour HEAD)."""
    status_line = stream.readline().decode("latin-1").rstrip("\r\n")
    if not status_line:
        raise AssertionError("connexion fermée sans réponse")
    headers = {}
    while (line := stream.readline().decode("latin-1").rstrip("\r\n")):
        name, value = line.split(": ", 1)
        headers[name] = value
    return status_line, headers, b"" if head else stream.read(int(headers["Content-Length"]))


@contextmanager
def service_behind(answer: bytes):
    """Un service IA branché sur un faux moteur « compatible OpenAI » qui répond `answer` (200) à
    toute requête ; alias « o » en embeddings comme en génération. Renvoie l'adresse du service."""
    class Engine(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, *args):
            pass

    engine = serve(ThreadingHTTPServer(("127.0.0.1", 0), Engine))
    alias = {"o": {"backend": "openai-compatible", "model": "m",
                   "base_url": f"http://127.0.0.1:{engine.server_address[1]}"}}
    service = serve(create_server(ModelRegistry.from_dict({"embedding": alias, "generation": alias}),
                                  "127.0.0.1", 0, quiet=True))
    try:
        yield f"http://127.0.0.1:{service.server_address[1]}"
    finally:
        for server in (service, engine):
            server.shutdown()
            server.server_close()


class AIServiceServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve(create_server(ModelRegistry.from_dict(CONFIG), "127.0.0.1", 0, quiet=True))
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, payload):
        return post(self.url + path, payload)

    def connect(self) -> socket.socket:
        return socket.create_connection(("127.0.0.1", self.server.server_address[1]), timeout=5)

    def raw(self, request: bytes) -> tuple[int, dict[str, str], bytes]:
        """Requête en octets bruts (ce qu'urllib n'enverrait pas) : statut, en-têtes et corps de la réponse."""
        with self.connect() as sock, sock.makefile("rb") as stream:
            sock.sendall(request)
            status_line, headers, body = read_response(stream, head=request.startswith(b"HEAD "))
        return int(status_line.split()[1]), headers, body

    def test_other_methods_get_a_json_405(self):
        for method in ("PUT", "OPTIONS", "TRACE"):
            request = urllib.request.Request(self.url + "/v1/embeddings", method=method)
            with self.subTest(method), self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            with caught.exception as error:
                self.assertEqual((error.code, json.loads(error.read())["error"]["code"]),
                                 (405, "method_not_allowed"))

    def test_a_known_route_with_another_method_is_405_with_allow(self):
        """Ce qu'annonce le contrat : 405 et l'en-tête Allow sur une route connue, 404 sur une route inconnue."""
        cases = [("GET", "/v1/embeddings", "POST"), ("POST", "/health", "GET, HEAD"), ("PUT", "/v1/generate", "POST"),
                 ("DELETE", "/v1/models", "GET, HEAD"), ("TRACE", "/health", "GET, HEAD")]
        for method, path, allowed in cases:
            with self.subTest(f"{method} {path}"):
                status, headers, body = self.raw(f"{method} {path} HTTP/1.0\r\nContent-Length: 2\r\n\r\n{{}}".encode())
                self.assertEqual((status, headers.get("Allow"), json.loads(body)["error"]),
                                 (405, allowed, {"code": "method_not_allowed",
                                                 "message": f"méthode {method} non permise sur {path} "
                                                            f"(permises : {allowed})"}))
        for method in ("GET", "POST", "PUT", "TRACE"):
            with self.subTest(f"{method} /inconnue"):
                status, headers, body = self.raw(f"{method} /inconnue HTTP/1.0\r\n\r\n".encode())
                self.assertEqual((status, json.loads(body)["error"]["code"]), (404, "not_found"))
                self.assertNotIn("Allow", headers)

    def test_head_is_accepted_wherever_get_is_without_the_body(self):
        """RFC 9110 : mêmes statut et en-têtes que GET (Date mise à part), sans corps ; ailleurs, 405 ou
        404, sans corps non plus."""
        for path in ("/health", "/v1/models"):
            with self.subTest(path):
                status, headers, body = self.raw(f"HEAD {path} HTTP/1.0\r\n\r\n".encode())
                get_status, get_headers, get_body = self.raw(f"GET {path} HTTP/1.0\r\n\r\n".encode())
                del headers["Date"], get_headers["Date"]
                self.assertEqual((status, headers, body), (get_status, get_headers, b""))
                self.assertEqual((status, int(headers["Content-Length"])), (200, len(get_body)))
        for path, expected, allowed in (("/v1/embeddings", 405, "POST"), ("/inconnue", 404, None)):
            with self.subTest(path):
                status, headers, body = self.raw(f"HEAD {path} HTTP/1.0\r\n\r\n".encode())
                self.assertEqual((status, headers.get("Allow"), body), (expected, allowed, b""))
                self.assertGreater(int(headers["Content-Length"]), 0)

    def test_a_thrown_away_body_does_not_delay_the_answer(self):
        """La réponse et la fin de l'envoi partent sans attendre un corps que personne ne lira (route
        inconnue, autre méthode, GET), même annoncé et jamais envoyé."""
        for line, length, expected in ((b"POST /inconnue", 4, 404), (b"PUT /v1/embeddings", 4, 405),
                                       (b"GET /health", 4, 200), (b"PUT /v1/embeddings", 10**12, 405)):
            with self.subTest(line=line, length=length), self.connect() as sock:
                sock.sendall(line + b" HTTP/1.0\r\nContent-Length: %d\r\n\r\n{}" % length)
                self.assertTrue(read_all(sock).startswith(b"HTTP/1.0 %d " % expected))

    def test_a_big_thrown_away_body_still_gets_its_answer(self):
        """Le corps jeté est lu jusqu'au bout après la réponse : fermée sur des octets non lus, la
        connexion serait réinitialisée avant que le client, qui envoie tout son corps avant de lire,
        ne voie la réponse (sans cette lecture, sous Windows comme sous Linux, chacun de ces cas perdait sa
        réponse). Un envoi en morceaux, refusé, est lu aussi."""
        body = b"x" * 20_000_000   # bien plus que ce que retiennent les tampons du système
        for method, path, chunked, expected in (("PUT", "/v1/embeddings", False, 405), ("GET", "/health", False, 200),
                                                ("POST", "/v1/embeddings", True, 400)):
            with self.subTest(f"{method} {path}"):
                connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=30)
                try:   # un itérable part en morceaux (Transfer-Encoding: chunked)
                    connection.request(method, path, body=iter([body]) if chunked else body)
                    self.assertEqual(connection.getresponse().status, expected)
                finally:
                    connection.close()

    def test_a_content_length_is_required_and_made_of_digits(self):
        """Les clients du service envoient toujours Content-Length : un envoi en morceaux est refusé,
        explicitement, plutôt que lu comme un corps vide. int() accepterait « +37 » ou « 3_7 »."""
        payload = b'{"model": "hashing", "inputs": ["a"]}'   # 37 octets

        def sized(length: bytes) -> bytes:
            return b"POST /v1/embeddings HTTP/1.1\r\nContent-Length: " + length + b"\r\n\r\n" + payload

        cases = {
            "envoi en morceaux": (b"POST /v1/embeddings HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                                  b"25\r\n" + payload + b"\r\n0\r\n\r\n",
                                  "Content-Length obligatoire : le service ne lit pas les envois en morceaux"),
            "Content-Length +…": (sized(b"+37"), "Content-Length invalide"),
            "Content-Length avec _": (sized(b"3_7"), "Content-Length invalide"),
        }
        for case, (request, message) in cases.items():
            with self.subTest(case):
                status, _, body = self.raw(request)
                error = json.loads(body)["error"]
                self.assertEqual((status, error["code"]), (400, "invalid_request"))
                self.assertIn(message, error["message"])
        self.assertEqual(self.raw(sized(b"0" * 30 + b"37 "))[0], 200)   # zéros devant, espace après

    def test_leading_zeros_are_read_in_linear_time(self):
        """Une ligne d'en-tête peut faire 64 Kio. Lue en temps quadratique, « 000…0x » occupait un fil une
        quinzaine de secondes : la réponse arrive ici avant le délai de 5 s de la connexion de test."""
        zeros = b"0" * 65_000
        payload = b'{"model": "hashing", "inputs": ["a"]}'   # 37 octets
        cases = {"illisible": (b"POST /v1/embeddings", zeros + b"x", b"", 400),
                 "zéros seuls": (b"GET /health", zeros, b"", 200),
                 "zéros puis la taille": (b"POST /v1/embeddings", zeros + b"37", payload, 200)}
        for case, (line, length, content, expected) in cases.items():
            with self.subTest(case):
                status, _, body = self.raw(line + b" HTTP/1.0\r\nContent-Length: " + length + b"\r\n\r\n" + content)
                self.assertEqual(status, expected)
                if expected == 400:
                    self.assertEqual(json.loads(body)["error"], {"code": "invalid_request",
                                                                 "message": "Content-Length invalide"})

    def test_what_the_standard_library_refuses_is_a_json_client_error(self):
        """Rien n'est laissé non lu (sinon la connexion pourrait être réinitialisée) : la ligne de requête ou
        d'en-tête fait juste un octet de trop, et la version arrive sans en-têtes. La réponse a sa ligne de
        statut, même pour une version illisible, puis la connexion se ferme."""
        cases = {
            "ligne de requête trop longue": (b"GET /" + b"a" * (65537 - 5), 414),
            "en-tête trop long": (b"GET /health HTTP/1.1\r\nX: " + b"a" * (65537 - 3), 431),
            "version HTTP/2.0": (b"GET /health HTTP/2.0\r\n", 505),
            "version illisible": (b"GET /health HTTP/1.x\r\n", 400),
        }
        for case, (request, expected) in cases.items():
            with self.subTest(case), self.connect() as sock, sock.makefile("rb") as stream:
                sock.sendall(request)
                line, _, body = read_response(stream)
                self.assertEqual((line.split()[:2], json.loads(body)["error"]["code"]),
                                 (["HTTP/1.0", str(expected)], "invalid_request"))
                self.assertEqual(stream.read(), b"")   # puis fermée

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

    def test_a_repeated_content_length_is_merged_if_identical_refused_otherwise(self):
        """Comme le permet la RFC 9110 (§ 8.6) : répété à l'identique (espaces autour mis à part), il est
        fondu en un ; deux valeurs différentes, même d'un même nombre (« 037 »), sont refusées plutôt que de
        garder la première."""
        payload = b'{"model": "hashing", "inputs": ["a"]}'   # 37 octets
        cases = {"deux valeurs identiques": ((b"37", b"37"), 200),
                 "trois, dont une suivie d'espaces": ((b"37", b"37 \t", b"37"), 200),
                 "deux valeurs différentes": ((b"37", b"3"), 400),
                 "un même nombre, écrit autrement": ((b"37", b"037"), 400)}
        for case, (lengths, expected) in cases.items():
            with self.subTest(case):
                headers = b"".join(b"Content-Length: %b\r\n" % length for length in lengths)
                status, _, body = self.raw(b"POST /v1/embeddings HTTP/1.0\r\n" + headers + b"\r\n" + payload)
                self.assertEqual(status, expected)
                if expected == 400:
                    self.assertEqual(json.loads(body)["error"],
                                     {"code": "invalid_request", "message": "Content-Length en double"})

    def test_a_body_of_more_than_16_mib_is_413_without_being_read(self):
        """Annoncé plus grand, il est refusé sans être lu : la réponse part sans l'attendre (jamais envoyé
        ici), même au-delà de 19 chiffres, ou de 4300 (que int() refuse de convertir). 16 Mio passent."""
        too_large = {"code": "payload_too_large",
                     "message": "corps de requête trop volumineux : 16777216 octets au plus (16 Mio)"}
        for length in (b"%d" % (MAX_BODY + 1), b"9" * 19, b"1" + b"0" * 19, b"1" + b"0" * 30, b"9" * 5000):
            with self.subTest(chiffres=len(length)):
                status, _, body = self.raw(b"POST /v1/embeddings HTTP/1.0\r\nContent-Length: " + length + b"\r\n\r\n")
                self.assertEqual((status, json.loads(body)["error"]), (413, too_large))
        start = b'{"model": "hashing", "inputs": ["a"], "x": "'   # « x » sert de remplissage
        for size, expected in ((MAX_BODY, 200), (MAX_BODY + 1, 413)):
            body = start + b"a" * (size - len(start) - 2) + b'"}'
            with self.subTest(taille=size):
                request = b"POST /v1/embeddings HTTP/1.0\r\nContent-Length: %d\r\n\r\n%b" % (size, body)
                self.assertEqual(self.raw(request)[0], expected)

    def test_an_idle_client_is_dropped_without_an_answer(self):
        """Délai d'inactivité de 30 s, raccourci ici : un client qui n'envoie plus rien ne retient pas un
        fil sans fin, ni l'arrêt du service, qui attend ses fils. La connexion est abandonnée sans réponse :
        ni 500, ni corps lu à moitié."""
        server = create_server(ModelRegistry.from_dict(CONFIG), "127.0.0.1", 0, quiet=True)
        self.addCleanup(server.server_close)
        self.assertEqual(server.RequestHandlerClass.timeout, 30)
        server.RequestHandlerClass.timeout = 0.2
        serve(server)
        self.addCleanup(server.shutdown)
        idle = {
            "rien d'envoyé": b"",
            "corps annoncé, envoyé à moitié": b'POST /v1/embeddings HTTP/1.0\r\nContent-Length: 100\r\n\r\n{"model": ',
            "corps jeté, jamais envoyé": b"PUT /v1/embeddings HTTP/1.0\r\nContent-Length: 1000\r\n\r\n",
        }
        clients = [socket.create_connection(server.server_address, timeout=5) for _ in idle]
        try:
            for sock, request in zip(clients, idle.values()):
                sock.sendall(request)
            answers = dict(zip(idle, map(read_all, clients)))
            self.assertEqual(answers["rien d'envoyé"], b"")
            self.assertEqual(answers["corps annoncé, envoyé à moitié"], b"")
            self.assertTrue(answers["corps jeté, jamais envoyé"].startswith(b"HTTP/1.0 405 "))
            server.shutdown()
            closing = threading.Thread(target=server.server_close, daemon=True)
            closing.start()
            closing.join(5)
            self.assertFalse(closing.is_alive())   # aucun fil n'est resté pendu à un client muet
        finally:
            for sock in clients:
                sock.close()

    def test_a_timeout_that_is_not_the_client_s_gets_an_answer(self):
        """Seul un client muet est abandonné sans réponse. Un délai dépassé ailleurs, ici un TimeoutError qu'un
        moteur laisse passer (request_json, lui, le traduit :
        test_an_engine_silent_or_cut_in_its_error_is_a_retryable_502), a sa réponse, comme toute erreur
        imprévue : jamais de connexion coupée sans réponse."""
        with mock.patch.object(HashingEmbeddingBackend, "embed", side_effect=TimeoutError("timed out")), \
                redirect_stderr(io.StringIO()) as stderr:
            status, body = self.post("/v1/embeddings", {"model": "hashing", "inputs": ["a"]})
        self.assertEqual((status, body["error"]),
                         (500, {"code": "internal_error", "message": "TimeoutError: timed out"}))
        self.assertIn("TimeoutError: timed out", stderr.getvalue())   # l'imprévu laisse sa pile dans le journal

    def test_a_port_already_taken_is_refused_even_under_windows(self):
        """Sous Windows, SO_REUSEADDR (que http.server active) laissait un second service écouter le même
        port sans erreur, les requêtes allant à l'un ou à l'autre."""
        registry = ModelRegistry.from_dict(CONFIG)
        first = create_server(registry, "127.0.0.1", 0, quiet=True)
        self.addCleanup(first.server_close)
        with self.assertRaises(OSError):
            create_server(registry, "127.0.0.1", first.server_address[1], quiet=True).server_close()

    def test_a_json_body_must_be_strict(self):
        """Ce que json.loads accepterait en silence : clé en double (la dernière gagnerait), « \\ud800 »
        isolé (la réponse qui le reprend ne pourrait pas être encodée), NaN, imbrication sans fin ; et
        un entier trop long pour int(), qui donnerait une erreur 500."""
        def with_x(value: bytes) -> bytes:
            return b'{"model": "hashing", "inputs": ["a"], "x": ' + value + b"}"

        def nested(levels: int, objects: bool = False) -> bytes:   # levels - 1 listes (ou objets) dans « x »
            return with_x(b'{"a": ' * (levels - 2) + b"{}" + b"}" * (levels - 2) if objects
                          else b"[" * (levels - 1) + b"]" * (levels - 1))

        cases = {
            "clé en double": (b'{"model": "autre", "model": "hashing", "inputs": ["a"]}', "clé « model » en double"),
            "clé en double, dans un objet imbriqué": (with_x(b'{"a": 1, "a": 2}'), "clé « a » en double"),
            "surrogate isolé": (b'{"model": "\\ud800", "inputs": ["a"]}', "surrogate UTF-16 isolé"),
            "moitié basse seule, dans une clé": (b'{"\\udc00": 1, "model": "hashing", "inputs": ["a"]}',
                                                 "surrogate UTF-16 isolé"),
            "emoji coupé, dans une liste": (b'{"model": "hashing", "inputs": ["\\ud83d"]}', "surrogate UTF-16 isolé"),
            "NaN": (with_x(b"NaN"), "« NaN » n'est pas du JSON"),
            "Infinity": (with_x(b"Infinity"), "« Infinity » n'est pas du JSON"),
            "-Infinity": (with_x(b"-Infinity"), "« -Infinity » n'est pas du JSON"),
            "entier de 4301 chiffres": (with_x(b"1" * 4301), "nombre entier de plus de 4300 chiffres"),
            "entier négatif de 4301 chiffres": (with_x(b"-" + b"1" * 4301), "nombre entier de plus de 4300 chiffres"),
            "65 niveaux de listes": (nested(65), "JSON trop imbriqué : plus de 64 niveaux"),
            "65 niveaux d'objets": (nested(65, objects=True), "JSON trop imbriqué : plus de 64 niveaux"),
            "100 000 niveaux": (b"[" * 100_000 + b"]" * 100_000, "JSON trop imbriqué : plus de 64 niveaux"),
        }
        for case, (body, message) in cases.items():
            with self.subTest(case):
                status, answer = self.post("/v1/embeddings", body)
                self.assertEqual((status, answer["error"]["code"]), (400, "invalid_request"))
                self.assertIn(message, answer["error"]["message"])
        accepted = {
            "64 niveaux de listes": nested(64),
            "64 niveaux d'objets": nested(64, objects=True),
            "une même clé dans deux objets": with_x(b'{"model": 1}'),
            "entier de 4300 chiffres": with_x(b"1" * 4300),
            "entier négatif de 4300 chiffres": with_x(b"-" + b"1" * 4300),
            "paire de surrogates complète": json.dumps({"model": "hashing", "inputs": ["\U0001F600 a"]}).encode(),
        }
        for case, body in accepted.items():
            with self.subTest(case):
                self.assertEqual(self.post("/v1/embeddings", body)[0], 200)

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
        # Un vrai refus de connexion (Ollama absent de 127.0.0.1:1), le seul de ces tests : il coûte 2 s sous Windows.
        status, body = self.post("/v1/embeddings", {"model": "ollama-absent", "inputs": ["a"]})
        self.assertEqual((status, body["error"]["code"]), (502, "backend_error"))
        self.assertTrue(body["error"]["retryable"])   # Ollama injoignable : peut revenir

    def test_an_engine_answer_that_is_not_json_is_502_not_400(self):
        """La faute est chez le moteur, pas chez le client : jamais « votre requête est invalide »."""
        with service_behind(b"<html>proxy</html>") as url:
            status, body = post(url + "/v1/embeddings", {"model": "o", "inputs": ["a"]})
        self.assertEqual((status, body["error"]["code"], body["error"]["retryable"]), (502, "backend_error", False))

    def test_an_engine_silent_or_cut_in_its_error_is_a_retryable_502(self):
        """Le moteur envoie ses en-têtes d'erreur, puis se tait (délai dépassé) ou coupe : 502 backend_error,
        réessayable, comme une coupure au milieu d'une réponse 200 ; jamais 500, ni pile dans le journal."""
        start = b"HTTP/1.1 500 Erreur\r\nContent-Length: 100\r\n\r\n{"
        for case, cut in (("silence", False), ("coupure", True)):
            with self.subTest(case), silent_after(start, cut) as engine:
                alias = {"o": {"backend": "openai-compatible", "model": "m", "base_url": engine, "timeout": 0.2}}
                service = serve(create_server(ModelRegistry.from_dict({"embedding": alias, "generation": alias}),
                                              "127.0.0.1", 0, quiet=True))
                try:
                    with redirect_stderr(io.StringIO()) as stderr:
                        status, body = post(f"http://127.0.0.1:{service.server_address[1]}/v1/embeddings",
                                            {"model": "o", "inputs": ["a"]})
                finally:
                    service.shutdown()
                    service.server_close()
                self.assertEqual((status, body["error"]["code"], body["error"].get("retryable")),
                                 (502, "backend_error", True))
                self.assertEqual(stderr.getvalue(), "")

    def test_a_lone_surrogate_from_the_engine_is_replaced_not_a_cut_connection(self):
        """Une moitié de paire ne s'écrit pas en UTF-8 : U+FFFD à la place, plutôt qu'aucune réponse."""
        answer = json.dumps({"choices": [{"message": {"content": "Deux jours\ud800 [1]"}}]}).encode()
        with service_behind(answer) as url:
            status, body = post(url + "/v1/generate", {"model": "o", "prompt": "p"})
        self.assertEqual((status, body["text"]), (200, "Deux jours\ufffd [1]"))

    def test_models_listing(self):
        with urllib.request.urlopen(self.url + "/v1/models", timeout=5) as response:
            body = json.loads(response.read())
        self.assertEqual([m["alias"] for m in body["generation"]], ["extractive"])


class Stopped:
    """Un serveur factice, déjà arrêté (comme après Ctrl+C), qui porte l'adresse qu'aurait le vrai : on n'écoute pas
    pour de bon, sur toutes les interfaces par exemple."""

    def __init__(self, address: tuple[str, int]) -> None:
        self.server_address = address

    def serve_forever(self) -> None:
        raise KeyboardInterrupt

    def server_close(self) -> None:
        pass


def launch(*args: str, bound: str = "127.0.0.1") -> tuple[int, str, str]:
    """python -m ai_service <args>, dans ce processus, sur un serveur factice (Stopped) au port 54321 : le code de
    retour, la sortie et la sortie d'erreur."""
    with mock.patch.object(sys, "argv", ["ai_service", *args]), \
            mock.patch.object(entry_point, "create_server", lambda registry, host, port: Stopped((bound, 54321))), \
            redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
        code = entry_point.main()
    return code, out.getvalue(), err.getvalue()


class EntryPointTest(unittest.TestCase):
    def test_port_0_takes_a_free_port_and_shows_the_one_listening(self):
        """`--port 0` l'emporte sur le port du fichier (ici occupé), et l'adresse affichée est celle
        qui écoute vraiment, pas « :0 ». Sortie redirigée, donc mise en tampon (ni « -u » ni PYTHONUNBUFFERED) :
        les lignes du lancement s'y lisent quand même pendant que le service tourne."""
        env = {name: value for name, value in os.environ.items() if name != "PYTHONUNBUFFERED"}
        env["PYTHONIOENCODING"] = "utf-8"   # « génération », lu ci-dessous en UTF-8
        with socket.socket() as taken, tempfile.TemporaryDirectory() as tmp:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            config = Path(tmp) / "ai_service.toml"
            config.write_text(f'[server]\nport = {taken.getsockname()[1]}\n\n[embedding.hashing]\nbackend = "hashing"\n',
                              encoding="utf-8")
            process = subprocess.Popen([sys.executable, "-m", "ai_service", "--config", str(config), "--port", "0"],
                                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
            try:
                lines = queue.Queue()   # l'adresse, les embeddings, la génération
                threading.Thread(target=lambda: lines.put([process.stdout.readline() for _ in range(3)]),
                                 daemon=True).start()
                address, _, generation = lines.get(timeout=20)
                self.assertTrue(generation.decode("utf-8").startswith("  génération :"), generation)
                shown = re.search(rb"http://127\.0\.0\.1:(\d+)", address)
                self.assertIsNotNone(shown)
                port = int(shown.group(1))
                self.assertNotIn(port, (0, taken.getsockname()[1]))
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                    self.assertEqual(json.loads(response.read())["status"], "ok")
            finally:
                process.kill()
                process.wait(10)
                process.stdout.close()

    def test_a_port_already_taken_is_said_like_by_assistant_serve(self):
        """Même message clair que « assistant serve », code 1, sans pile."""
        with socket.socket() as taken, tempfile.TemporaryDirectory() as tmp:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):   # Windows : sans lui, un second bind partagerait le port
                taken.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            config = Path(tmp) / "ai_service.toml"
            config.write_text('[embedding.hashing]\nbackend = "hashing"\n', encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "ai_service", "--config", str(config),
                                     "--port", str(taken.getsockname()[1])], cwd=ROOT, capture_output=True,
                                    timeout=60, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        errors = result.stderr.decode("utf-8")
        self.assertEqual((result.returncode, result.stdout), (1, b""), errors)
        self.assertIn("Erreur : impossible d'écouter (port déjà pris, adresse inconnue ou non autorisée) — ", errors)
        self.assertNotIn("Traceback", errors)

    def test_a_bad_port_or_configuration_is_said_like_by_assistant_serve(self):
        """Port hors de 0 à 65535 ou qui n'est pas un entier (option ou fichier), fichier absent ou qui n'est pas
        du TOML, moteur inconnu ou configuration que le registre refuse : « Erreur : … » sur une ligne, code 1,
        sans pile, comme le port déjà pris. Les bornes, 0 et 65535, sont acceptées."""
        hashing = '[embedding.hashing]\nbackend = "hashing"\n'
        with tempfile.TemporaryDirectory() as tmp:
            def config(name: str, content: str | bytes) -> str:
                path = Path(tmp) / name
                path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
                return str(path)

            valid = config("valide.toml", hashing)
            far_port = config("port.toml", "[server]\nport = 70000\n\n" + hashing)
            text_port = config("port-texte.toml", '[server]\nport = "8100"\n\n' + hashing)
            true_port = config("port-booleen.toml", "[server]\nport = true\n\n" + hashing)
            not_toml = config("pas-toml.toml", "[embedding.hashing\n")
            not_utf8 = config("pas-utf8.toml", hashing.encode("utf-8") + b'description = "Cong\xe9s"\n')
            unknown = config("inconnu.toml", '[embedding.x]\nbackend = "inconnu"\n')
            no_engine = config("sans-moteur.toml", '[generation.x]\ndescription = "sans moteur"\n')
            absent = str(Path(tmp) / "absent.toml")
            cases = {   # arguments, début du message
                "--port 70000": (["--config", valid, "--port", "70000"],
                                 "l'option --port attend un port entre 0 et 65535, pas « 70000 »"),
                "--port -1": (["--config", valid, "--port", "-1"],
                              "l'option --port attend un port entre 0 et 65535, pas « -1 »"),
                "--port 65536": (["--config", valid, "--port", "65536"],
                                 "l'option --port attend un port entre 0 et 65535, pas « 65536 »"),
                "port du fichier": (["--config", far_port],
                                    f"{far_port} [server] : « port » = 70000, doit être un entier compris entre 0 et "
                                    "65535"),
                "port du fichier en texte": (["--config", text_port], f'{text_port} [server] : « port » = "8100", '
                                             "doit être un entier compris entre 0 et 65535"),
                "port du fichier booléen": (["--config", true_port], f"{true_port} [server] : « port » = true, "
                                            "doit être un entier compris entre 0 et 65535"),
                "fichier absent": (["--config", absent], "fichier de configuration inaccessible — [Errno 2] "),
                "pas du TOML": (["--config", not_toml], f"{not_toml} : TOML invalide — "),
                "pas de l'UTF-8": (["--config", not_utf8], f"{not_utf8} : TOML invalide — "),
                "moteur inconnu": (["--config", unknown], f"{unknown} : backend d'embeddings inconnu : inconnu"),
                "moteur absent": (["--config", no_engine], f"{no_engine} : generation.x : champ 'backend' obligatoire"),
            }
            for case, (args, message) in cases.items():
                with self.subTest(case):
                    code, out, err = launch(*args)
                    self.assertEqual((code, out), (1, ""), err)
                    self.assertTrue(err.startswith(f"Erreur : {message}"), err)
                    self.assertEqual(err.count("\n"), 1, err)
            for port in ("0", "65535"):
                with self.subTest(f"--port {port}"):
                    code, _, err = launch("--config", valid, "--port", port)
                    self.assertEqual((code, err), (0, ""))

    def test_all_interfaces_are_announced_with_an_address_to_connect_to(self):
        """Comme « assistant serve » : pour 0.0.0.0 (toutes les interfaces, mais pas une adresse où se connecter),
        127.0.0.1, en le disant ; tout autre hôte tel quel (« localhost », pas l'adresse résolue)."""
        hashing = '[embedding.hashing]\nbackend = "hashing"\n'
        everywhere = "http://127.0.0.1:54321, à l'écoute sur toutes les interfaces"
        cases = {   # arguments, contenu du fichier, adresse du serveur, adresse annoncée
            "--host localhost": (["--host", "localhost"], hashing, "127.0.0.1", "http://localhost:54321"),
            "--host 0.0.0.0": (["--host", "0.0.0.0"], hashing, "0.0.0.0", everywhere),
            "host du fichier": ([], '[server]\nhost = "0.0.0.0"\n\n' + hashing, "0.0.0.0", everywhere),
        }
        for case, (args, content, bound, announced) in cases.items():
            with self.subTest(case), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / "ai_service.toml"
                config.write_text(content, encoding="utf-8")
                code, out, err = launch("--config", str(config), *args, "--port", "0", bound=bound)
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(out.splitlines()[0], f"Service IA sur {announced}")


if __name__ == "__main__":
    unittest.main()
