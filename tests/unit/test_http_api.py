"""L'API HTTP de l'application, contre un conteneur monté sur les doubles : aucun service IA."""

import contextlib
import http.client
import io
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO
from unittest import mock

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.errors import IndexWriteError
from assistant.application.index_corpus import IndexCorpus
from assistant.application.search_passages import SearchPassages
from assistant.application.snapshots import RecordSnapshot
from assistant.application.status import CheckStatus
from assistant.composition import AppConfig, Container
from assistant.infrastructure.http_ai_client import HttpEmbedder
from assistant.infrastructure.vector_index import JsonVectorIndex
from assistant.interface.http_api import ReadWriteLock, create_server
from tests.contract.test_http_ai_client import stalling
from tests.fakes import (
    FakeIndex, FixedClock, KeywordEmbedder, ListSource, MemorySnapshotStore, ScriptedGenerator, StaticPrompts,
    WholeDocumentSplitter, make_document,
)
from tests.unit.test_status import BrokenEmbedder

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = [make_document("teletravail", "Deux jours de télétravail par semaine."),
             make_document("grille", "Salaire senior.", groups=["rh"])]
QUESTION = {"user": "alice", "question": "Combien de jours de télétravail ?"}
MAX_BODY = 16 * 1024 * 1024   # 16 Mio : au-delà, 413
TOO_LARGE = {"code": "payload_too_large", "message": "corps de requête trop volumineux : 16777216 octets au plus (16 Mio)"}


def make_container(tmp: Path, embedder=None, index=None, generator=None, documents=DOCUMENTS) -> Container:
    """Le conteneur de `serve`, monté sur les doubles (index JSON dans `tmp`, sauf autre index)."""
    config = replace(AppConfig.load(ROOT / "config/app.toml"), index_path=tmp / "index.json")
    embedder = embedder or KeywordEmbedder()
    index = index if index is not None else JsonVectorIndex(tmp / "index.json")
    source = ListSource(documents)
    settings = AskSettings(min_score=0.5)
    ask = AskQuestion(embedder, index, generator or ScriptedGenerator("Deux jours par semaine [1]."),
                      StaticPrompts(), settings)
    return Container(
        config, IndexCorpus(source, WholeDocumentSplitter(), embedder, index, FixedClock()), ask,
        SearchPassages(embedder, index), CheckStatus(source, WholeDocumentSplitter(), embedder, index, StaticPrompts()),
        RecordSnapshot(ask, MemorySnapshotStore(), FixedClock(), {}), MemorySnapshotStore(), index,
        StaticPrompts(), settings)


def serve(test: unittest.TestCase, container: Container) -> str:
    """Démarre l'API sur un port libre, arrêtée à la fin du test ; renvoie son adresse."""
    server = create_server(container, "127.0.0.1", 0, quiet=True)
    # Arrêt vérifié toutes les 50 ms (0,5 s par défaut) : sinon chaque test attend jusqu'à une demi-seconde.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_address[1]}"


def read_all(sock: socket.socket) -> bytes:
    """Tout ce qui arrive jusqu'à ce que le serveur ferme la connexion (requête HTTP/1.0, corps non lu…)."""
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


def padded(size: int) -> bytes:
    """Une question valide de `size` octets exactement : le champ « x » sert de remplissage."""
    start = b'{"user": "alice", "question": "jours", "x": "'
    return start + b"a" * (size - len(start) - 2) + b'"}'


def in_chunks(body: bytes) -> bytes:
    """Le corps envoyé en morceaux de 1 Mio (Transfer-Encoding: chunked)."""
    step = 1024 * 1024
    return b"".join(b"%x\r\n%b\r\n" % (len(body[i:i + step]), body[i:i + step])
                    for i in range(0, len(body), step)) + b"0\r\n\r\n"


class UnwritableIndex(FakeIndex):
    """Un index qui ne peut pas être écrit (disque plein…) : IndexWriteError, comme le prévoit le port."""

    def replace(self, manifest, chunks, vectors):
        raise IndexWriteError("data/index.json", "disque plein")


class SilentAIService(KeywordEmbedder):
    """Un adaptateur qui laisse passer le TimeoutError d'un service IA muet (le vrai client HTTP le traduit en
    AIServiceError : test_a_service_silent_in_the_middle_of_its_error_is_a_502) :
    un délai dépassé ailleurs que sur la connexion de l'appelant."""

    def embed_documents(self, texts):
        raise TimeoutError("timed out")


class RebuiltAtEverySearch(FakeIndex):
    """Un autre processus reconstruit l'index pendant chaque recherche : la question ne peut aboutir."""

    def search(self, vector, top_k, predicate, index_id=None):
        self.replace(replace(self.manifest(), index_id=self.manifest().index_id + "+"), self.chunks, self.vectors)
        return super().search(vector, top_k, predicate, index_id)


class FailingAsk:
    """Un cas d'usage qui lève l'erreur donnée : ce que l'API en fait est seul en jeu."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self, user, question):
        raise self.error


class HttpApiTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.url = serve(self, make_container(self.tmp))

    def call(self, method, path, body: bytes | None = None):
        request = urllib.request.Request(self.url + path, data=body, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:   # toujours une réponse JSON, jamais une connexion coupée
                return error.code, json.loads(error.read())

    def post(self, path, payload):
        return self.call("POST", path, json.dumps(payload).encode())

    def connect(self) -> socket.socket:
        return socket.create_connection(("127.0.0.1", int(self.url.rsplit(":", 1)[1])), timeout=5)

    def raw(self, request: bytes) -> tuple[int, dict[str, str], bytes]:
        """Requête en octets bruts (ce qu'urllib n'enverrait pas) : statut, en-têtes et corps de la réponse."""
        with self.connect() as sock, sock.makefile("rb") as stream:
            sock.sendall(request)
            status_line, headers, body = read_response(stream, head=request.startswith(b"HEAD "))
        return int(status_line.split()[1]), headers, body

    def test_health_then_index_then_ask(self):
        status, body = self.call("GET", "/health")
        self.assertEqual((status, body["index"]), (200, None))
        status, body = self.post("/v1/ask", {"user": "alice", "question": "télétravail"})
        self.assertEqual((status, body["error"]["code"]), (409, "index_unusable"))
        self.assertEqual(self.post("/v1/index", {})[0], 200)
        status, body = self.post("/v1/ask", QUESTION)
        self.assertEqual((status, body["status"]), (200, "answered"))
        self.assertEqual(self.call("GET", "/v1/status?detail=1")[0], 200)   # la requête ne compte pas

    def test_status_is_409_to_redo_and_503_unverified_with_the_report_as_body(self):
        """GET /v1/status rend le rapport de status, pas {"error": …} : 409 quand l'index est à refaire (ici, aucun
        index), même si le service IA est aussi en échec ; 503 quand seul le modèle servi n'a pas pu être vérifié."""
        failed = "modèle servi non vérifié, le service IA a échoué : Service IA injoignable"
        status, body = self.call("GET", "/v1/status")   # aucun index
        self.assertEqual((status, body["up_to_date"], body["unverified"], body["issues"]),
                         (409, False, False, ["aucun index : lancer l'indexation"]))
        self.assertNotIn("error", body)
        working, self.url = self.url, serve(self, make_container(self.tmp, embedder=BrokenEmbedder()))
        status, body = self.call("GET", "/v1/status")   # aucun index, service IA en échec : 409, pas 503
        self.assertEqual((status, body["up_to_date"], body["unverified"], body["issues"]),
                         (409, False, False, ["aucun index : lancer l'indexation", failed]))
        self.assertNotIn("error", body)
        self.url = working
        self.assertEqual(self.post("/v1/index", {})[0], 200)
        self.url = serve(self, make_container(self.tmp, embedder=BrokenEmbedder()))   # même index, service IA en échec
        status, body = self.call("GET", "/v1/status")
        self.assertEqual((status, body["up_to_date"], body["unverified"], body["issues"]), (503, False, True, [failed]))
        self.assertNotIn("error", body)

    def test_client_errors_are_4xx_with_a_json_body(self):
        self.post("/v1/index", {})
        cases = [
            (self.post("/v1/ask", {"user": "mallory", "question": "Q ?"}), 403, "unknown_user"),
            (self.post("/v1/ask", {"user": "alice", "question": "   "}), 400, "invalid_question"),
            # U+001C (séparateur) : un blanc pour str.strip(), comme l'espace. La question est vide.
            (self.post("/v1/ask", {"user": "alice", "question": "\x1c"}), 400, "invalid_question"),
            (self.post("/v1/ask", {"user": "alice", "question": None}), 400, "invalid_question"),
            (self.post("/v1/ask", {"user": "alice", "question": 42}), 400, "invalid_request"),
            (self.call("POST", "/v1/ask", b"{pas du json"), 400, "invalid_json"),
            (self.call("POST", "/v1/ask", b"[]"), 400, "invalid_json"),
            (self.call("POST", "/v1/ask", b"   "), 400, "invalid_json"),   # des espaces ne sont pas du JSON
            # JSON valide, mais encodé en latin-1 : c'est le décodage UTF-8 strict qui doit le refuser.
            (self.call("POST", "/v1/ask", '{"user": "alice", "question": "télétravail"}'.encode("latin-1")),
             400, "invalid_json"),
            (self.call("GET", "/v1/inconnu"), 404, "not_found"),
            (self.call("PUT", "/v1/ask", b"{}"), 405, "method_not_allowed"),
            (self.call("OPTIONS", "/v1/ask"), 405, "method_not_allowed"),
            (self.call("TRACE", "/v1/ask"), 405, "method_not_allowed"),   # inconnue de la bibliothèque standard
        ]
        for (status, body), expected_status, expected_code in cases:
            with self.subTest(expected_code):
                self.assertEqual((status, body["error"]["code"]), (expected_status, expected_code))

    def test_a_json_body_is_strict(self):
        """Ce que json.loads accepterait en silence (clé en double, « \\ud800 » isolé, NaN), ou refuserait par
        une trace ou en anglais (plus de 900 niveaux, entier de plus de 4300 chiffres) : 400 invalid_json."""
        self.post("/v1/index", {})

        def with_x(value: bytes) -> bytes:
            return b'{"user": "alice", "question": "jours", "x": ' + value + b"}"

        def nested(levels: int, objects: bool = False) -> bytes:   # levels - 1 listes (ou objets) dans « x »
            return with_x(b'{"a": ' * (levels - 2) + b"{}" + b"}" * (levels - 2) if objects
                          else b"[" * (levels - 1) + b"]" * (levels - 1))

        cases = {
            "clé en double": (b'{"user": "zoe", "user": "alice", "question": "jours"}', "clé « user » en double"),
            "clé en double, dans un objet imbriqué": (b'{"x": {"a": 1, "a": 2}, "user": "alice", "question": "jours"}',
                                                      "clé « a » en double"),
            "surrogate isolé": (b'{"user": "alice", "question": "\\ud800"}', "surrogate UTF-16 isolé"),
            "moitié basse seule, dans une clé": (b'{"\\udc00": 1, "user": "alice"}', "surrogate UTF-16 isolé"),
            "emoji coupé, dans une liste": (b'{"user": "alice", "x": ["\\ud83d"]}', "surrogate UTF-16 isolé"),
            "NaN": (with_x(b"NaN"), "« NaN » n'est pas du JSON"),
            "Infinity": (with_x(b"Infinity"), "« Infinity » n'est pas du JSON"),
            "-Infinity": (with_x(b"-Infinity"), "« -Infinity » n'est pas du JSON"),
            "entier de 4301 chiffres": (with_x(b"1" * 4301), "nombre entier de plus de 4300 chiffres"),
            "entier négatif de 4301 chiffres": (with_x(b"-" + b"1" * 4301), "nombre entier de plus de 4300 chiffres"),
            "901 niveaux de listes": (nested(901), "JSON trop imbriqué : plus de 900 niveaux"),
            "901 niveaux d'objets": (nested(901, objects=True), "JSON trop imbriqué : plus de 900 niveaux"),
            "100 000 niveaux": (b"[" * 100_000 + b"]" * 100_000, "JSON trop imbriqué : plus de 900 niveaux"),
        }
        for case, (body, message) in cases.items():
            with self.subTest(case):
                status, answer = self.call("POST", "/v1/ask", body)
                self.assertEqual((status, answer["error"]["code"]), (400, "invalid_json"))
                self.assertIn(message, answer["error"]["message"])
        accepted = {
            "900 niveaux de listes": nested(900),
            "900 niveaux d'objets": nested(900, objects=True),
            "une même clé dans deux objets": b'{"a": {"user": 1}, "user": "alice", "question": "jours"}',
            "entier de 4300 chiffres": with_x(b"1" * 4300),
            "entier négatif de 4300 chiffres": with_x(b"-" + b"1" * 4300),
            "réel de 5000 chiffres": with_x(b"1" * 5000 + b".5"),
            "paire de surrogates complète": b'{"user": "alice", "question": "\\ud83d\\ude00 jours"}',
        }
        for case, body in accepted.items():
            with self.subTest(case):
                self.assertEqual(self.call("POST", "/v1/ask", body)[0], 200)

    def test_the_depth_is_checked_before_any_other_defect(self):
        """Plus de 900 niveaux : le corps est refusé d'emblée, avant json.loads, même si un autre défaut vient
        avant. Au fond de 900 niveaux, le défaut est dit, même sous Python 3.11 derrière le serveur (qui y
        abandonne vers 982 niveaux) ; au fond de 901, le corps est refusé d'emblée."""
        self.post("/v1/index", {})
        cases = (
            (b'{"x": {"a": 1, "a": 2}, "y": ' + b"[" * 900 + b"]" * 900 + b', "user": "alice", "question": "jours"}',
             "JSON trop imbriqué : plus de 900 niveaux"),
            (b'{"user": "alice", "question": ' + b"[" * 899 + b"NaN" + b"]" * 899 + b"}", "« NaN » n'est pas du JSON"),
            (b'{"user": "alice", "question": ' + b"[" * 900 + b"NaN" + b"]" * 900 + b"}",
             "JSON trop imbriqué : plus de 900 niveaux"),
        )
        for body, message in cases:
            with self.subTest(body[:40]):
                self.assertEqual(self.call("POST", "/v1/ask", body),
                                 (400, {"error": {"code": "invalid_json", "message": message}}))

    def test_the_body_is_read_by_the_strict_reader_of_the_composition(self):
        """Un seul contrôleur de JSON strict, celui des fichiers (json_text), que l'API reçoit de la racine de
        composition (read_json_text) : elle n'en garde pas de copie."""
        with mock.patch("assistant.interface.http_api.read_json_text", side_effect=ValueError("témoin")) as reader:
            status, body = self.post("/v1/ask", QUESTION)
        reader.assert_called_once_with(json.dumps(QUESTION))
        self.assertEqual((status, body["error"]), (400, {"code": "invalid_json", "message": "témoin"}))

    def test_a_body_is_written_as_frozen(self):
        # json.dumps(corps, ensure_ascii=False) : une ligne, « , » et « : » suivis d'une espace, accents tels quels.
        body = b'{"user": "a", "user": "b"}'
        status, _, answer = self.raw(b"POST /v1/ask HTTP/1.1\r\nConnection: close\r\nContent-Length: %d\r\n\r\n"
                                     % len(body) + body)
        self.assertEqual((status, answer.decode("utf-8")),
                         (400, '{"error": {"code": "invalid_json", "message": "clé « user » en double"}}'))

    def test_an_empty_body_a_byte_order_mark_and_a_padded_length_are_accepted(self):
        self.assertEqual(self.call("POST", "/v1/index")[0], 200)                        # corps vide
        status, body = self.call("POST", "/v1/ask", b"\xef\xbb\xbf" + json.dumps(QUESTION).encode())   # marque d'ordre des octets
        self.assertEqual((status, body["status"]), (200, "answered"))
        payload = json.dumps(QUESTION).encode()
        status, _, body = self.raw(b"POST /v1/ask HTTP/1.1\r\nContent-Length: %s%d \r\n\r\n" % (b"0" * 30, len(payload))
                                   + payload)
        self.assertEqual((status, json.loads(body)["status"]), (200, "answered"))   # zéros devant, espace après

    def test_leading_zeros_are_read_in_linear_time(self):
        """Une ligne d'en-tête peut faire 64 Kio. Lue en temps quadratique, « 000…0x » occupait un fil une
        quinzaine de secondes : la réponse arrive ici avant le délai de 5 s de la connexion de test."""
        zeros = b"0" * 65_000
        cases = {"illisible": (zeros + b"x", b"", 400), "zéros seuls": (zeros, b"", 200),
                 "zéros puis la taille": (zeros + b"2", b"{}", 200)}
        for case, (length, content, expected) in cases.items():
            with self.subTest(case):
                status, _, body = self.raw(b"POST /v1/index HTTP/1.0\r\nContent-Length: " + length + b"\r\n\r\n"
                                           + content)
                self.assertEqual(status, expected)
                if expected == 400:
                    self.assertEqual(json.loads(body)["error"], {"code": "invalid_request",
                                                                 "message": "Content-Length invalide"})

    def test_a_content_length_with_a_sign_or_an_underscore_is_refused(self):
        """Un Content-Length fait d'autre chose que des chiffres : 400 en JSON. int() ne suffit pas : il
        accepterait « +… » ou « _ »."""
        payload = json.dumps(QUESTION).encode()

        def sized(length: str) -> bytes:
            return b"POST /v1/ask HTTP/1.1\r\nContent-Length: " + length.encode() + b"\r\n\r\n" + payload

        length = str(len(payload))
        cases = {
            "Content-Length +…": sized("+" + length),
            "Content-Length avec _": sized(length[:1] + "_" + length[1:]),
        }
        for case, request in cases.items():
            with self.subTest(case):
                status, _, body = self.raw(request)
                self.assertEqual((status, json.loads(body)["error"]),
                                 (400, {"code": "invalid_request", "message": "Content-Length invalide"}))

    def test_a_chunked_body_is_refused_saying_what_to_send_instead(self):
        """Un envoi en morceaux (Transfer-Encoding) n'est pas lu : 400, et le message dit quoi envoyer à la place
        (Content-Length). Jamais lu comme un corps vide, quel que soit le codage."""
        chunks = in_chunks(json.dumps(QUESTION).encode())
        for coding in ("chunked", "gzip, chunked"):
            with self.subTest(coding):
                status, _, body = self.raw(f"POST /v1/ask HTTP/1.1\r\nTransfer-Encoding: {coding}\r\n\r\n".encode()
                                           + chunks)
                self.assertEqual((status, json.loads(body)["error"]), (400, {
                    "code": "invalid_request",
                    "message": "Content-Length obligatoire : l'application ne lit pas les envois en morceaux "
                               f"(Transfer-Encoding: {coding})"}))

    def test_the_path_is_compared_as_sent(self):
        """Ni décodage des %XX ni résolution des « .. » : « /h%65alth » n'est pas « /health ». Seuls sont retirés la
        requête (?…), le fragment (#…), la forme absolue (http://hôte) et les « / » de tête en trop."""
        cases = {b"/h%65alth": 404, b"/v1/../health": 404, b"http://127.0.0.1/health?x=1": 200, b"//health": 200}
        for target, expected in cases.items():
            with self.subTest(target):
                self.assertEqual(self.raw(b"GET " + target + b" HTTP/1.0\r\n\r\n")[0], expected)

    def test_a_known_route_with_another_method_is_405_with_allow(self):
        """RFC 9110 : 405 et l'en-tête Allow sur une route connue ; 404 sur une route inconnue."""
        cases = [("GET", "/v1/ask", "POST"), ("POST", "/health", "GET, HEAD"), ("PUT", "/v1/index", "POST"),
                 ("DELETE", "/v1/status", "GET, HEAD"), ("TRACE", "/health", "GET, HEAD")]
        for method, path, allowed in cases:
            with self.subTest(f"{method} {path}"):
                status, headers, body = self.raw(f"{method} {path} HTTP/1.0\r\nContent-Length: 2\r\n\r\n{{}}".encode())
                self.assertEqual((status, headers.get("Allow"), json.loads(body)["error"]["code"]),
                                 (405, allowed, "method_not_allowed"))
        for method in ("GET", "POST", "PUT", "TRACE"):
            with self.subTest(f"{method} /v1/inconnu"):
                status, headers, body = self.raw(f"{method} /v1/inconnu HTTP/1.0\r\n\r\n".encode())
                self.assertEqual((status, json.loads(body)["error"]["code"]), (404, "not_found"))
                self.assertNotIn("Allow", headers)

    def test_head_is_accepted_wherever_get_is_without_the_body(self):
        """RFC 9110 : mêmes statut et en-têtes que GET (Date mise à part), sans corps ;
        ailleurs, 405 ou 404, sans corps non plus."""
        self.post("/v1/index", {})
        for path in ("/health", "/v1/status"):
            with self.subTest(path):
                status, headers, body = self.raw(f"HEAD {path} HTTP/1.0\r\n\r\n".encode())
                get_status, get_headers, get_body = self.raw(f"GET {path} HTTP/1.0\r\n\r\n".encode())
                del headers["Date"], get_headers["Date"]
                self.assertEqual((status, headers, body), (get_status, get_headers, b""))
                self.assertEqual((status, int(headers["Content-Length"])), (200, len(get_body)))
        for path, expected, allowed in (("/v1/ask", 405, "POST"), ("/v1/inconnu", 404, None)):
            with self.subTest(path):
                status, headers, body = self.raw(f"HEAD {path} HTTP/1.0\r\n\r\n".encode())
                self.assertEqual((status, headers.get("Allow"), body), (expected, allowed, b""))
                self.assertGreater(int(headers["Content-Length"]), 0)

    def test_a_thrown_away_body_does_not_delay_the_answer(self):
        """La réponse et la fin de l'envoi partent sans attendre un corps que personne ne lira (route
        inconnue, autre méthode, GET), même annoncé et jamais envoyé."""
        for line, length, expected in ((b"PUT /v1/ask", 4, 405), (b"POST /v1/inconnu", 4, 404),
                                       (b"GET /health", 4, 200), (b"PUT /v1/ask", 10**12, 405)):
            with self.subTest(line=line, length=length), self.connect() as sock:
                sock.sendall(line + b" HTTP/1.0\r\nContent-Length: %d\r\n\r\n{}" % length)
                self.assertTrue(read_all(sock).startswith(b"HTTP/1.0 %d " % expected))

    def test_a_big_thrown_away_body_still_gets_its_answer(self):
        """Le corps jeté est lu jusqu'au bout après la réponse : fermée sur des octets non lus, la
        connexion serait réinitialisée avant que le client, qui envoie tout son corps avant de lire,
        ne voie la réponse (sans cette lecture, sous Windows comme sous Linux, chacun de ces cas perdait sa
        réponse). Si sa fin est inconnue (envoi en morceaux, refusé), il est lu jusqu'à ce que le client ferme."""
        body = b"x" * 20_000_000   # bien plus que ce que retiennent les tampons du système
        cases = [("PUT", "/v1/ask", body, {}, 405), ("GET", "/health", body, {}, 200),
                 ("POST", "/v1/inconnu", iter([body]), {}, 400),   # un itérable part en morceaux (chunked)
                 ("POST", "/v1/ask", body, {"Transfer-Encoding": "gzip, chunked"}, 400)]
        for method, path, content, headers, expected in cases:
            with self.subTest(f"{method} {path} {headers}"):
                connection = http.client.HTTPConnection("127.0.0.1", int(self.url.rsplit(":", 1)[1]), timeout=30)
                try:
                    connection.request(method, path, body=content, headers=headers)
                    self.assertEqual(connection.getresponse().status, expected)
                finally:
                    connection.close()

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
        connection = http.client.HTTPConnection("127.0.0.1", int(self.url.rsplit(":", 1)[1]), timeout=5)
        try:
            connection.putrequest("POST", "/v1/ask")
            connection.putheader("Content-Length", "-1")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual((response.status, json.loads(response.read())["error"]["code"]),
                             (400, "invalid_request"))
        finally:
            connection.close()

    def test_a_repeated_content_length_is_merged_if_identical_refused_otherwise(self):
        """Répété à l'identique (espaces autour mis à part), il est fondu en un ; deux valeurs différentes,
        même d'un même nombre (« 037 »), sont refusées plutôt que de garder la première."""
        self.post("/v1/index", {})
        payload = json.dumps(QUESTION).encode()
        size = b"%d" % len(payload)
        accepted = {
            "deux valeurs identiques": b"Content-Length: %b\r\nContent-Length: %b\r\n" % (size, size),
            "trois, dont une suivie d'espaces": b"Content-Length: %b\r\nContent-Length: %b \t\r\nContent-Length: %b\r\n"
                                                % (size, size, size),
        }
        for case, headers in accepted.items():
            with self.subTest(case):
                status, _, body = self.raw(b"POST /v1/ask HTTP/1.0\r\n" + headers + b"\r\n" + payload)
                self.assertEqual((status, json.loads(body).get("status")), (200, "answered"))
        refused = {
            "deux valeurs différentes": b"Content-Length: %b\r\nContent-Length: 3\r\n" % size,
            "un même nombre, écrit autrement": b"Content-Length: %b\r\nContent-Length: 0%b\r\n" % (size, size),
        }
        for case, headers in refused.items():
            with self.subTest(case):
                status, _, body = self.raw(b"POST /v1/ask HTTP/1.0\r\n" + headers + b"\r\n" + payload)
                self.assertEqual((status, json.loads(body)["error"]),
                                 (400, {"code": "invalid_request", "message": "Content-Length en double"}))

    def test_a_body_of_more_than_16_mib_is_413(self):
        """Annoncé plus grand, il est refusé sans être lu : la réponse part sans l'attendre (jamais envoyé
        ici), même au-delà de 19 chiffres, ou de 4300 (que int() refuse de convertir). 16 Mio passent."""
        self.post("/v1/index", {})
        for length in (b"%d" % (MAX_BODY + 1), b"9" * 19, b"1" + b"0" * 19, b"1" + b"0" * 30, b"9" * 5000):
            with self.subTest(chiffres=len(length)):
                status, _, body = self.raw(b"POST /v1/ask HTTP/1.0\r\nContent-Length: " + length + b"\r\n\r\n")
                self.assertEqual((status, json.loads(body)["error"]), (413, TOO_LARGE))
        for size, expected in ((MAX_BODY, 200), (MAX_BODY + 1, 413)):
            with self.subTest(taille=size):
                request = b"POST /v1/ask HTTP/1.0\r\nContent-Length: %d\r\n\r\n%b" % (size, padded(size))
                self.assertEqual(self.raw(request)[0], expected)

    def test_an_idle_client_is_dropped_without_an_answer(self):
        """Délai d'inactivité de 30 s, raccourci ici : un client qui n'envoie plus rien ne retient pas un
        fil sans fin, ni l'arrêt du serveur, qui attend ses fils. La connexion est abandonnée sans réponse :
        ni 500, ni corps lu à moitié."""
        server = create_server(make_container(self.tmp), "127.0.0.1", 0, quiet=True)
        self.addCleanup(server.server_close)
        self.assertEqual(server.RequestHandlerClass.timeout, 30)
        server.RequestHandlerClass.timeout = 0.2
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.shutdown)
        idle = {
            "rien d'envoyé": b"",
            "corps annoncé, envoyé à moitié": b'POST /v1/ask HTTP/1.0\r\nContent-Length: 100\r\n\r\n{"user": ',
            "corps jeté, jamais envoyé": b"PUT /v1/ask HTTP/1.0\r\nContent-Length: 1000\r\n\r\n",
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

    def test_a_body_cut_short_by_the_client_is_400(self):
        """Le client ferme son envoi avant la fin que son Content-Length annonce, aussitôt ou un peu plus tard :
        400, plutôt que de traiter le début du corps (« {} » : 403, utilisateur vide)."""
        for content, delay in ((b"{}", 0), (b"{}", 0.2), (b"", 0)):
            with self.subTest(content=content, delay=delay), self.connect() as sock, sock.makefile("rb") as stream:
                sock.sendall(b"POST /v1/ask HTTP/1.1\r\nContent-Length: 100\r\n\r\n" + content)
                time.sleep(delay)
                sock.shutdown(socket.SHUT_WR)
                _, _, body = read_response(stream)
                message = f"corps de requête incomplet : {len(content)} octets reçus sur 100 annoncés (Content-Length)"
                self.assertEqual(json.loads(body).get("error"), {"code": "invalid_request", "message": message})

    def test_a_body_already_cut_short_is_400_even_where_it_would_be_thrown_away(self):
        """Un corps déjà coupé (le client a fermé son envoi avant la longueur annoncée) fait une requête incomplète
        (RFC 9112, section 6.3) : 400, comme pour un corps lu, même là où il serait jeté (route inconnue, autre
        méthode, GET, plus de 16 Mio annoncés). Ici, tout est envoyé avant que le serveur n'accepte la connexion.
        Coupé après la réponse, il ne change rien à celle-ci (test_a_thrown_away_body_does_not_delay_the_answer)."""
        server = create_server(make_container(self.tmp), "127.0.0.1", 0, quiet=True)
        self.addCleanup(server.server_close)
        cases = {"route inconnue": (b"POST /v1/inconnu", 100), "autre méthode": (b"PUT /v1/ask", 100),
                 "GET": (b"GET /health", 100), "plus de 16 Mio annoncés": (b"POST /v1/ask", 20_000_000)}
        for case, (line, length) in cases.items():
            with self.subTest(case), socket.create_connection(server.server_address, timeout=5) as sock, \
                    sock.makefile("rb") as stream:
                sock.sendall(line + b" HTTP/1.1\r\nContent-Length: %d\r\n\r\n{}" % length)
                sock.shutdown(socket.SHUT_WR)
                server.handle_request()   # la connexion n'est acceptée qu'une fois tout envoyé
                _, _, body = read_response(stream)
                self.assertEqual(json.loads(body).get("error"), {
                    "code": "invalid_request",
                    "message": f"corps de requête incomplet : 2 octets reçus sur {length} annoncés (Content-Length)"})

    def test_a_thrown_away_body_cut_short_is_seen_in_what_has_arrived(self):
        """L'envoi d'un corps jeté qui s'arrête avant la fin annoncée est vu dans ce qui en est déjà arrivé, lu sans
        attendre, 64 Kio au plus : 400. Au-delà, il ne l'est pas : la réponse est celle du corps jeté (404, 405).
        Arrivé en entier, le corps jeté n'est pas coupé, même si le client ferme ensuite son envoi : 404."""
        server = create_server(make_container(self.tmp), "127.0.0.1", 0, quiet=True)
        self.addCleanup(server.server_close)
        cases = ((b"POST /v1/inconnu", 16384, 1000, 400), (b"POST /v1/inconnu", 16385, 1000, 400),
                 (b"PUT /v1/ask", 60_000, 1000, 400), (b"POST /v1/inconnu", 70_000, 1000, 404),
                 (b"PUT /v1/ask", 70_000, 1000, 405), (b"POST /v1/inconnu", 60_000, 0, 404))
        for line, sent, missing, expected in cases:
            with self.subTest(line=line, sent=sent, missing=missing), \
                    socket.create_connection(server.server_address, timeout=5) as sock, sock.makefile("rb") as stream:
                sock.sendall(line + b" HTTP/1.1\r\nContent-Length: %d\r\n\r\n" % (sent + missing) + b"x" * sent)
                sock.shutdown(socket.SHUT_WR)
                server.handle_request()   # la connexion n'est acceptée qu'une fois tout envoyé
                status, _, body = read_response(stream)
                self.assertEqual(int(status.split()[1]), expected)
                if expected == 400:
                    message = f"corps de requête incomplet : {sent} octets reçus sur {sent + missing} annoncés"
                    self.assertEqual(json.loads(body)["error"],
                                     {"code": "invalid_request", "message": f"{message} (Content-Length)"})

    def test_a_timeout_that_is_not_the_client_s_gets_an_answer(self):
        """Seul un client muet est abandonné sans réponse. Un délai dépassé ailleurs, ici chez le service IA,
        a sa réponse, comme toute erreur : jamais de connexion coupée sans réponse."""
        self.url = serve(self, make_container(self.tmp, embedder=SilentAIService()))
        status, body = self.post("/v1/index", {})
        self.assertEqual((status, body["error"]), (500, {"code": "unreadable_state", "message": "timed out"}))

    def test_a_service_silent_in_the_middle_of_its_error_is_a_502(self):
        """Le vrai client HTTP, contre un service IA qui envoie ses en-têtes d'erreur puis se tait : son délai à
        lui expire, l'erreur est passagère, et l'appelant reçoit 502 ai_service_error, pas 500."""
        with stalling(b"HTTP/1.1 500 Erreur\r\nContent-Length: 100\r\n\r\n{") as url:
            self.url = serve(self, make_container(self.tmp, embedder=HttpEmbedder(url, "m", timeout=0.2)))
            status, body = self.post("/v1/index", {})
        self.assertEqual((status, body["error"]), (502, {
            "code": "ai_service_error", "message": f"Service IA injoignable ({url}/v1/embeddings) : timed out"}))

    def test_a_port_already_taken_is_refused_even_under_windows(self):
        """Sous Windows, SO_REUSEADDR (que http.server active) laissait un second serveur écouter le même
        port sans erreur, les requêtes allant à l'un ou à l'autre. C'est une erreur."""
        container = make_container(self.tmp)
        first = create_server(container, "127.0.0.1", 0, quiet=True)
        self.addCleanup(first.server_close)
        with self.assertRaises(OSError):
            create_server(container, "127.0.0.1", first.server_address[1], quiet=True).server_close()

    def test_an_index_rebuilt_during_every_search_is_409(self):
        container = make_container(self.tmp, index=RebuiltAtEverySearch())
        container.index_corpus.execute()
        self.url = serve(self, container)
        status, body = self.post("/v1/ask", QUESTION)
        self.assertEqual((status, body["error"]["code"]), (409, "index_unusable"))
        self.assertIn("reconstruit pendant la recherche", body["error"]["message"])

    def test_an_unreadable_index_is_a_server_error_not_a_client_error(self):
        (self.tmp / "index.json").write_text("{ pas du json", encoding="utf-8")
        status, body = self.post("/v1/ask", {"user": "alice", "question": "télétravail"})
        self.assertEqual((status, body["error"]["code"]), (500, "unreadable_state"))

    def test_an_index_that_cannot_be_written_has_its_own_code(self):
        """Pas « unreadable_state » : rien n'est illisible, l'écriture a échoué (l'index précédent reste)."""
        self.url = serve(self, make_container(self.tmp, index=UnwritableIndex()))
        status, body = self.post("/v1/index", {})
        self.assertEqual((status, body["error"]), (500, {
            "code": "index_write_failed", "message": "écriture impossible de l'index (data/index.json) : disque plein"}))

    def test_server_errors_are_500_with_a_json_body(self):
        cases = {
            "corpus vide (erreur applicative)": (make_container(self.tmp, documents=[]), "/v1/index", {},
                                                 "unreadable_state"),
            "panne imprévue": (replace(make_container(self.tmp), ask_question=FailingAsk(RuntimeError("panne"))),
                               "/v1/ask", QUESTION, "internal_error"),
        }
        for case, (container, path, payload, code) in cases.items():
            with self.subTest(case), contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.url = serve(self, container)
                status, body = self.post(path, payload)
                self.assertEqual((status, body["error"]["code"]), (500, code))
                if code == "internal_error":   # l'imprévu laisse sa pile dans le journal
                    self.assertIn("RuntimeError: panne", stderr.getvalue())

    def test_a_lone_surrogate_in_the_answer_is_replaced_not_a_cut_connection(self):
        """U+FFFD, le caractère de remplacement d'Unicode, à la place : une réponse plutôt qu'aucune."""
        self.url = serve(self, make_container(self.tmp, generator=ScriptedGenerator("Deux jours\ud800 par semaine [1].")))
        self.post("/v1/index", {})
        status, body = self.post("/v1/ask", QUESTION)
        self.assertEqual((status, body["text"]), (200, "Deux jours\ufffd par semaine [1]."))


class BlockingEmbedder(KeywordEmbedder):
    """Une question qui reste dans l'embedding tant qu'on ne la libère pas."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def embed_query(self, text):
        self.entered.set()
        self.release.wait(5)
        return super().embed_query(text)


class ConcurrencyTest(unittest.TestCase):
    """Même processus : une réindexation n'entre pas pendant une question (b), et ne se fait pas
    doubler indéfiniment par les questions suivantes."""

    def test_reindexing_waits_for_the_question_in_progress(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        embedder = BlockingEmbedder()
        url = serve(self, make_container(Path(tmp.name), embedder=embedder, documents=DOCUMENTS[:1]))

        def post(path, payload):
            request = urllib.request.Request(url + path, data=json.dumps(payload).encode(), method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status

        post("/v1/index", {})
        asking = threading.Thread(target=post, args=("/v1/ask", {"user": "alice", "question": "télétravail"}))
        asking.start()
        self.assertTrue(embedder.entered.wait(5))
        reindexed = threading.Event()
        threading.Thread(target=lambda: (post("/v1/index", {}), reindexed.set()), daemon=True).start()
        self.assertFalse(reindexed.wait(0.3))   # la question est en cours : la réindexation attend
        embedder.release.set()
        asking.join(5)
        self.assertTrue(reindexed.wait(5))

    def test_a_waiting_reindexing_goes_before_new_questions(self):
        lock, order = ReadWriteLock(), []
        first_in, release_first = threading.Event(), threading.Event()

        def first_question():
            with lock.reading():
                first_in.set()
                release_first.wait(5)

        def reindexing():
            with lock.writing():
                order.append("réindexation")

        def next_question():
            with lock.reading():
                order.append("question suivante")

        # Fils démons, attente bornée : si le verrou régresse, le test échoue au lieu de bloquer la suite.
        threads = [threading.Thread(target=first_question, daemon=True)]
        threads[0].start()
        self.assertTrue(first_in.wait(5))
        threads.append(threading.Thread(target=reindexing, daemon=True))
        threads[1].start()
        deadline = time.monotonic() + 5
        while not lock._writers_waiting and not order and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(order, [])        # la réindexation attend la question en cours
        threads.append(threading.Thread(target=next_question, daemon=True))
        threads[2].start()
        time.sleep(0.2)
        self.assertEqual(order, [])        # la question suivante ne double pas la réindexation
        release_first.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual([thread.is_alive() for thread in threads], [False] * 3)
        self.assertEqual(order, ["réindexation", "question suivante"])


if __name__ == "__main__":
    unittest.main()
