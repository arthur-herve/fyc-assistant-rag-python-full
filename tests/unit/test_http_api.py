"""L'API HTTP de l'application, contre un conteneur monté sur les doubles : aucun service IA."""

import http.client
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from assistant.application.ask_question import AskQuestion, AskSettings
from assistant.application.index_corpus import IndexCorpus
from assistant.application.search_passages import SearchPassages
from assistant.application.snapshots import RecordSnapshot
from assistant.application.status import CheckStatus
from assistant.composition import AppConfig, Container
from assistant.infrastructure.vector_index import JsonVectorIndex
from assistant.interface.http_api import ReadWriteLock, create_server
from tests.fakes import (
    FixedClock, KeywordEmbedder, ListSource, MemorySnapshotStore, ScriptedGenerator, StaticPrompts,
    WholeDocumentSplitter, make_document,
)

ROOT = Path(__file__).resolve().parents[2]


class HttpApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index_path = Path(self.tmp.name) / "index.json"
        config = replace(AppConfig.load(ROOT / "config/app.toml"), index_path=self.index_path)
        embedder, index = KeywordEmbedder(), JsonVectorIndex(self.index_path)
        source = ListSource([make_document("teletravail", "Deux jours de télétravail par semaine."),
                             make_document("grille", "Salaire senior.", groups=["rh"])])
        settings = AskSettings(min_score=0.5)
        ask = AskQuestion(embedder, index, ScriptedGenerator("Deux jours par semaine [1]."), StaticPrompts(), settings)
        container = Container(
            config, IndexCorpus(source, WholeDocumentSplitter(), embedder, index, FixedClock()), ask,
            SearchPassages(embedder, index), CheckStatus(source, WholeDocumentSplitter(), embedder, index, StaticPrompts()),
            RecordSnapshot(ask, MemorySnapshotStore(), FixedClock(), {}), MemorySnapshotStore(), index,
            StaticPrompts(), settings)
        self.server = create_server(container, "127.0.0.1", 0, quiet=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

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

    def test_health_then_index_then_ask(self):
        status, body = self.call("GET", "/health")
        self.assertEqual((status, body["index"]), (200, None))
        status, body = self.post("/v1/ask", {"user": "alice", "question": "télétravail"})
        self.assertEqual((status, body["error"]["code"]), (409, "index_unusable"))
        self.assertEqual(self.post("/v1/index", {})[0], 200)
        status, body = self.post("/v1/ask", {"user": "alice", "question": "Combien de jours de télétravail ?"})
        self.assertEqual((status, body["status"]), (200, "answered"))
        self.assertEqual(self.call("GET", "/v1/status?detail=1")[0], 200)   # la requête ne compte pas

    def test_client_errors_are_4xx_with_a_json_body(self):
        self.post("/v1/index", {})
        cases = [
            (self.post("/v1/ask", {"user": "mallory", "question": "Q ?"}), 403, "unknown_user"),
            (self.post("/v1/ask", {"user": "alice", "question": "   "}), 400, "invalid_question"),
            (self.post("/v1/ask", {"user": "alice", "question": None}), 400, "invalid_question"),
            (self.post("/v1/ask", {"user": "alice", "question": 42}), 400, "invalid_request"),
            (self.call("POST", "/v1/ask", b"{pas du json"), 400, "invalid_json"),
            (self.call("POST", "/v1/ask", b"[]"), 400, "invalid_json"),
            # JSON valide, mais encodé en latin-1 : c'est le décodage UTF-8 strict qui doit le refuser.
            (self.call("POST", "/v1/ask", '{"user": "alice", "question": "télétravail"}'.encode("latin-1")),
             400, "invalid_json"),
            (self.call("GET", "/v1/inconnu"), 404, "not_found"),
            (self.call("PUT", "/v1/ask", b"{}"), 405, "method_not_allowed"),
            (self.call("OPTIONS", "/v1/ask"), 405, "method_not_allowed"),
            (self.call("TRACE", "/v1/ask"), 405, "method_not_allowed"),   # refusée par la bibliothèque standard
        ]
        for (status, body), expected_status, expected_code in cases:
            with self.subTest(expected_code):
                self.assertEqual((status, body["error"]["code"]), (expected_status, expected_code))

    def test_what_the_csharp_version_accepts_is_accepted(self):
        self.assertEqual(self.call("POST", "/v1/index")[0], 200)                        # corps vide
        status, body = self.call("POST", "/v1/ask", b"\xef\xbb\xbf" + json.dumps(
            {"user": "alice", "question": "Combien de jours de télétravail ?"}).encode())   # marque d'ordre des octets
        self.assertEqual((status, body["status"]), (200, "answered"))
        request = urllib.request.Request(self.url + "/health", method="HEAD")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 405)

    def test_a_negative_content_length_gets_an_answer(self):
        """read(-1) attendrait la fin de la connexion : le fil resterait bloqué, sans réponse."""
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        try:
            connection.putrequest("POST", "/v1/ask")
            connection.putheader("Content-Length", "-1")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual((response.status, json.loads(response.read())["error"]["code"]),
                             (400, "invalid_request"))
        finally:
            connection.close()

    def test_an_unreadable_index_is_a_server_error_not_a_client_error(self):
        self.index_path.write_text("{ pas du json", encoding="utf-8")
        status, body = self.post("/v1/ask", {"user": "alice", "question": "télétravail"})
        self.assertEqual((status, body["error"]["code"]), (500, "unreadable_state"))



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
        config = replace(AppConfig.load(ROOT / "config/app.toml"), index_path=Path(tmp.name) / "index.json")
        embedder, index = BlockingEmbedder(), JsonVectorIndex(Path(tmp.name) / "index.json")
        source = ListSource([make_document("teletravail", "Deux jours de télétravail par semaine.")])
        settings = AskSettings(min_score=0.5)
        ask = AskQuestion(embedder, index, ScriptedGenerator("Deux jours [1]."), StaticPrompts(), settings)
        container = Container(
            config, IndexCorpus(source, WholeDocumentSplitter(), embedder, index, FixedClock()), ask,
            SearchPassages(embedder, index), CheckStatus(source, WholeDocumentSplitter(), embedder, index, StaticPrompts()),
            RecordSnapshot(ask, MemorySnapshotStore(), FixedClock(), {}), MemorySnapshotStore(), index,
            StaticPrompts(), settings)
        server = create_server(container, "127.0.0.1", 0, quiet=True)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}"

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

        threads = [threading.Thread(target=first_question)]
        threads[0].start()
        self.assertTrue(first_in.wait(5))
        threads.append(threading.Thread(target=reindexing))
        threads[1].start()
        deadline = time.monotonic() + 5
        while not lock._writers_waiting and not order and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(order, [])        # la réindexation attend la question en cours
        threads.append(threading.Thread(target=next_question))
        threads[2].start()
        time.sleep(0.2)
        self.assertEqual(order, [])        # la question suivante ne double pas la réindexation
        release_first.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual(order, ["réindexation", "question suivante"])


if __name__ == "__main__":
    unittest.main()
