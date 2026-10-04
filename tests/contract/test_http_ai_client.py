"""Contrat entre l'application et le service IA, vérifié sans réseau."""

import contextlib
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from assistant.application.errors import AIServiceError
from assistant.application.ports import GenerationRequest
from assistant.infrastructure.http_ai_client import HttpEmbedder, HttpGenerator

REQUEST = GenerationRequest(system="s", prompt="p", temperature=0.2, max_tokens=100)
MISSING = "Réponse du service IA incomplète, champs manquants : "
INCOHERENT = "Réponse du service IA incohérente : "

# (réponse JSON du service, erreur attendue) : un service qui ne respecte pas le contrat donne une
# AIServiceError non passagère, jamais une trace d'erreur.
MALFORMED_EMBEDDINGS = [
    ("null", MISSING + "[model, dimension, vectors]"),
    ("5", MISSING + "[model, dimension, vectors]"),
    ('"model dimension vectors"', MISSING + "[model, dimension, vectors]"),
    ("[]", MISSING + "[model, dimension, vectors]"),
    ('{"vectors": [[1.0]]}', MISSING + "[model, dimension]"),
    ('{"model": 5, "dimension": 2, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« model » doit être un texte"),
    ('{"model": null, "dimension": 2, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« model » doit être un texte"),
    # Demi-paire de substitution seule : pas un texte Unicode (elle ne s'écrit pas en UTF-8).
    ('{"model": "\\ud800", "dimension": 2, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« model » doit être un texte"),
    ('{"model": "m", "dimension": "abc", "vectors": [[0.1, 0.2]]}', INCOHERENT + "« dimension » doit être un entier"),
    ('{"model": "m", "dimension": true, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« dimension » doit être un entier"),
    ('{"model": "m", "dimension": 2.0, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« dimension » doit être un entier"),
    # Un entier de 32 bits, comme le veut le contrat : de -2**31 à 2**31 - 1.
    ('{"model": "m", "dimension": 2147483648, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« dimension » doit être un entier"),
    ('{"model": "m", "dimension": -2147483649, "vectors": [[0.1, 0.2]]}', INCOHERENT + "« dimension » doit être un entier"),
    ('{"model": "m", "dimension": 2147483647, "vectors": [[0.1, 0.2]]}',
     INCOHERENT + "1 vecteurs de 2147483647 dimensions attendus, reçu 1 de [2] dimensions"),
    ('{"model": "m", "dimension": 2, "vectors": [1]}', INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    ('{"model": "m", "dimension": 2, "vectors": "ab"}', INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, "x"]]}', INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, true]]}', INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, null]]}', INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, 1e400]]}', INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    # Un entier de 401 chiffres : trop grand pour un réel (OverflowError en Python).
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, 1' + "0" * 400 + ']]}',
     INCOHERENT + "« vectors » doit être une liste de listes de nombres"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1]]}',
     INCOHERENT + "1 vecteurs de 2 dimensions attendus, reçu 1 de [1] dimensions"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, 0.2], [0.3, 0.4]]}',
     INCOHERENT + "1 vecteurs de 2 dimensions attendus, reçu 2 de [2] dimensions"),
]

MALFORMED_GENERATIONS = [
    ("null", MISSING + "[model, text]"),
    ('{"text": "Deux jours [1]."}', MISSING + "[model]"),
    ('{"model": "m", "text": null}', INCOHERENT + "« text » doit être un texte"),
    ('{"model": "m", "text": 5}', INCOHERENT + "« text » doit être un texte"),
    ('{"model": "m", "text": "\\udc00 ok"}', INCOHERENT + "« text » doit être un texte"),
    ('{"model": ["m"], "text": "Deux jours [1]."}', INCOHERENT + "« model » doit être un texte"),
]

# (corps d'une réponse 502, détail attendu dans le message, passagère ?) : seule une erreur au format
# du contrat, avec un message textuel, est lue ; tout autre corps est cité tel quel, et un 5xx reste
# alors passager.
PROBLEM = '{"error": {"code": "backend_error", "message": "modele absent", "retryable": false}}'
DEEP = "[" * 100_000 + "]" * 100_000
ERROR_BODIES = [
    (PROBLEM.encode(), "modele absent", False),
    (b"\xef\xbb\xbf" + PROBLEM.encode(), "modele absent", False),   # marque d'ordre des octets
    (b'{"error": {"message": "surcharge"}}', "surcharge", True),
    (b'{"error": {"message": null, "retryable": false}}', '{"error": {"message": null, "retryable": false}}', True),
    (b'{"error": {"message": 5, "retryable": false}}', '{"error": {"message": 5, "retryable": false}}', True),
    (b'{"error": {"message": "\\ud800", "retryable": false}}',
     '{"error": {"message": "\\ud800", "retryable": false}}', True),
    (b'{"error": {"message": "x", "retryable": NaN}}', '{"error": {"message": "x", "retryable": NaN}}', True),
    (b'{"error": "x"}', '{"error": "x"}', True),
    (b"<html>proxy</html>", "<html>proxy</html>", True),
    (DEEP.encode(), DEEP, True),
    # JSON qui n'est pas strict (clé en double, 901 niveaux, entier de plus de 4300 chiffres) : pas une erreur au
    # format du contrat.
    (b'{"error": {"message": "a", "message": "b"}}', '{"error": {"message": "a", "message": "b"}}', True),
    (b'{"error": {"message": "x"}, "x": ' + b"[" * 900 + b"]" * 900 + b"}",
     '{"error": {"message": "x"}, "x": ' + "[" * 900 + "]" * 900 + "}", True),
    (b'{"error": {"message": "x"}, "n": 1' + b"0" * 5000 + b"}", '{"error": {"message": "x"}, "n": 1' + "0" * 5000 + "}",
     True),
]

# Clé en double (la dernière gagnait sans rien dire), plus de 900 niveaux, NaN, entier de plus de 4300 chiffres (int()
# refuse de le convertir) : illisible, avec le message de json_text.
HUGE = "1" + "0" * 5000
NOT_STRICT = [
    ('{"model": "m", "model": "autre", "dimension": 1, "vectors": [[1.0]]}', "clé « model » en double"),
    ('{"model": "m", "dimension": 1, "vectors": [[1.0]], "x": ' + "[" * 900 + "]" * 900 + "}",
     "JSON trop imbriqué : plus de 900 niveaux"),
    ('{"model": "m", "dimension": 1, "vectors": [[-Infinity]]}', "« -Infinity » n'est pas du JSON"),
    ('{"model": "m", "dimension": 2, "vectors": [[0.1, ' + HUGE + ']]}', "nombre entier de plus de 4300 chiffres"),
    ('{"model": "m", "dimension": ' + HUGE + ', "vectors": [[0.1, 0.2]]}', "nombre entier de plus de 4300 chiffres"),
]


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.response


@contextlib.contextmanager
def answering(status, body):
    """Un faux service IA qui répond toujours `status` et `body`, sur un port libre."""
    class Fixed(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixed)
    # Attente courte entre deux vérifications de l'arrêt : shutdown() rend la main en 0,05 s, pas 0,5 s.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def stalling(start: bytes, cut: bool = False):
    """Un faux service IA qui lit la requête, envoie `start` (ses en-têtes et un début de corps, ou autre chose que du
    HTTP), puis se tait, connexion ouverte, jusqu'à ce que le client abandonne ; ou coupe la connexion (cut). Sur un
    port libre."""
    class Stalling(BaseHTTPRequestHandler):
        timeout = 10   # au pire, le faux service abandonne lui-même

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.wfile.write(start)
            if not cut:
                self.rfile.read()   # se tait, jusqu'à ce que le client ferme
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stalling)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class HttpEmbedderContractTest(unittest.TestCase):
    def test_query_request_and_response(self):
        transport = RecordingTransport({"model": "ollama:nomic@abc", "dimension": 2, "vectors": [[0.1, 0.2]]})
        batch = HttpEmbedder("http://ia:8100/", "nomic", transport=transport).embed_query("bonjour")

        url, payload = transport.calls[0]
        self.assertEqual(url, "http://ia:8100/v1/embeddings")
        self.assertEqual(payload, {"model": "nomic", "input_type": "query", "inputs": ["bonjour"]})
        self.assertEqual((batch.model, batch.dimension), ("ollama:nomic@abc", 2))

    def test_documents_are_sent_as_documents(self):
        transport = RecordingTransport({"model": "m", "dimension": 1, "vectors": [[1], [2]]})
        HttpEmbedder("http://ia", "m", transport=transport).embed_documents(["a", "b"])
        self.assertEqual(transport.calls[0][1]["input_type"], "document")

    def test_incomplete_response_is_an_error(self):
        transport = RecordingTransport({"vectors": [[1.0]]})
        with self.assertRaises(AIServiceError):
            HttpEmbedder("http://ia", "m", transport=transport).embed_query("x")

    def test_vectors_that_do_not_match_the_announced_dimension_are_an_error(self):
        for vectors in ([[0.1]], [[0.1, 0.2], [0.3, 0.4]]):   # trop courts ; trop nombreux
            with self.subTest(vectors=vectors):
                transport = RecordingTransport({"model": "m", "dimension": 2, "vectors": vectors})
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder("http://ia", "m", transport=transport).embed_query("x")
                self.assertFalse(caught.exception.transient)

    def test_a_response_of_another_shape_is_an_error_not_a_crash(self):
        for response, message in MALFORMED_EMBEDDINGS:
            with self.subTest(response=response):
                transport = RecordingTransport(json.loads(response))
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder("http://ia", "m", transport=transport).embed_query("x")
                self.assertEqual(str(caught.exception), message)
                self.assertFalse(caught.exception.transient)   # réessayer ne changerait rien

    def test_a_502_marked_not_retryable_is_not_transient(self):
        """Le service dit « réessayer ne changera rien » (modèle absent) : pas de nouvelle tentative."""
        for retryable in (True, False):
            body = json.dumps({"error": {"code": "backend_error", "message": "x", "retryable": retryable}})
            with self.subTest(retryable=retryable), answering(502, body.encode()) as url:
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder(url, "m", timeout=5).embed_query("x")
                self.assertEqual(caught.exception.transient, retryable)

    def test_an_error_body_of_another_shape_is_quoted_as_is(self):
        for body, detail, transient in ERROR_BODIES:
            with self.subTest(body=body[:60]), answering(502, body) as url:
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder(url, "m", timeout=5).embed_query("x")
                self.assertEqual(str(caught.exception), f"Service IA : HTTP 502 — {detail}")
                self.assertEqual(caught.exception.transient, transient)

    def test_a_body_that_stops_or_is_cut_is_transient(self):
        """Le service envoie ses en-têtes, puis se tait (délai dépassé) ou coupe, au milieu du corps d'une
        erreur comme d'une réponse 200 : passager, comme avant les en-têtes (l'appelant de l'API HTTP reçoit
        502 ai_service_error), jamais une exception de la bibliothèque standard (500)."""
        error = b"HTTP/1.1 500 Erreur\r\nContent-Length: 100\r\n\r\n{"
        ok = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"
        cases = {
            "erreur, puis silence": (error, False),
            "erreur coupée": (error, True),
            "erreur en morceaux, coupée": (b"HTTP/1.1 500 Erreur\r\nTransfer-Encoding: chunked\r\n\r\n10\r\n{", True),
            "réponse 200, puis silence": (ok, False),
            "réponse 200 coupée": (ok, True),
        }
        for case, (start, cut) in cases.items():
            with self.subTest(case), stalling(start, cut) as url:
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder(url, "m", timeout=0.2).embed_query("x")
                self.assertTrue(str(caught.exception).startswith(f"Service IA injoignable ({url}/v1/embeddings) : "))
                self.assertTrue(caught.exception.transient)
                if not cut:
                    self.assertTrue(str(caught.exception).endswith(" : timed out"))

    def test_a_response_that_is_not_http_is_not_transient(self):
        """AI_SERVICE_URL vers un mauvais port, où répond un autre programme : autre chose que du HTTP, qui reviendra
        à chaque essai. Non passager, comme le contrat le dit pour un moteur (l'appelant de l'API HTTP reçoit 502
        ai_service_error), jamais une exception de la bibliothèque standard (500). La coupure en plein corps du test
        précédent, une exception de la même famille (http.client.HTTPException), reste passagère."""
        with stalling(b"SSH-2.0-OpenSSH_9.6\r\n", cut=True) as url:
            with self.assertRaises(AIServiceError) as caught:
                HttpEmbedder(url, "m", timeout=5).embed_query("x")
        self.assertEqual(str(caught.exception), f"Service IA ({url}/v1/embeddings) n'a pas répondu en HTTP : "
                                                "BadStatusLine('SSH-2.0-OpenSSH_9.6\\r\\n')")
        self.assertFalse(caught.exception.transient)

    def test_an_address_that_is_not_valid_is_said_and_not_transient(self):
        """AI_SERVICE_URL mal écrite (un port qui n'est pas un nombre) : rien n'est contacté. Ni « injoignable », ni
        « n'a pas répondu en HTTP » : l'adresse est invalide, et le restera à chaque essai."""
        with self.assertRaises(AIServiceError) as caught:
            HttpEmbedder("http://127.0.0.1:abc", "m", timeout=5).embed_query("x")
        self.assertTrue(str(caught.exception).startswith(
            "Adresse du service IA invalide (http://127.0.0.1:abc/v1/embeddings) : "), str(caught.exception))
        self.assertFalse(caught.exception.transient)

    def test_a_response_that_is_not_json_is_an_error(self):
        """Ni JSON (page d'un proxy, NaN, listes imbriquées sur 100 000 niveaux), ni UTF-8 : illisible,
        et non passager (le décorateur de nouvelles tentatives ne réessaie pas pour rien)."""
        for body in (b"<html>proxy</html>", b'{"model": NaN}', DEEP.encode(), b'{"model": "Cong\xe9s"}'):
            with self.subTest(body=body[:60]), answering(200, body) as url:
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder(url, "m", timeout=5).embed_query("x")
                self.assertIn(f"Réponse du service IA illisible ({url}/v1/embeddings) : ", str(caught.exception))
                self.assertFalse(caught.exception.transient)

    def test_a_response_that_is_not_strict_json_is_unreadable(self):
        for body, problem in NOT_STRICT:
            with self.subTest(body=body[:60]), answering(200, body.encode()) as url:
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder(url, "m", timeout=5).embed_query("x")
                self.assertEqual(str(caught.exception), f"Réponse du service IA illisible ({url}/v1/embeddings) : {problem}")
                self.assertFalse(caught.exception.transient)

    def test_a_response_that_is_not_utf_8_says_where(self):
        """La position se compte après la marque d'ordre des octets."""
        body = b'{"model": "Cong\xe9s"}'
        position = body.index(b"\xe9")
        for prefix in (b"", b"\xef\xbb\xbf"):
            with self.subTest(prefix=prefix), answering(200, prefix + body) as url:
                with self.assertRaises(AIServiceError) as caught:
                    HttpEmbedder(url, "m", timeout=5).embed_query("x")
                self.assertEqual(str(caught.exception), f"Réponse du service IA illisible ({url}/v1/embeddings) : "
                                                        f"pas en UTF-8 (octet 0xe9 à la position {position})")

    def test_a_response_with_a_byte_order_mark_is_read(self):
        body = b"\xef\xbb\xbf" + json.dumps({"model": "m", "dimension": 1, "vectors": [[1.0]]}).encode()
        with answering(200, body) as url:
            self.assertEqual(HttpEmbedder(url, "m", timeout=5).embed_query("x").model, "m")


class HttpGeneratorContractTest(unittest.TestCase):
    def test_request_and_response(self):
        transport = RecordingTransport({"model": "ollama:qwen3:4b", "text": "Réponse [1]"})
        generation = HttpGenerator("http://ia", "qwen3-4b", transport=transport).generate(
            GenerationRequest(system="s", prompt="p", temperature=0.2, max_tokens=100, seed=7))

        url, payload = transport.calls[0]
        self.assertEqual(url, "http://ia/v1/generate")
        self.assertEqual(payload, {"model": "qwen3-4b", "system": "s", "prompt": "p",
                                   "temperature": 0.2, "max_tokens": 100, "seed": 7})
        self.assertEqual(generation.text, "Réponse [1]")

    def test_a_character_outside_the_basic_plane_is_text(self):
        """« \\ud83d\\ude00 » est une paire complète (un emoji) : un texte."""
        transport = RecordingTransport(json.loads('{"model": "m", "text": "Bravo \\ud83d\\ude00"}'))
        generation = HttpGenerator("http://ia", "m", transport=transport).generate(REQUEST)
        self.assertEqual(generation.text, "Bravo \U0001F600")

    def test_a_response_of_another_shape_is_an_error_not_a_crash(self):
        for response, message in MALFORMED_GENERATIONS:
            with self.subTest(response=response):
                transport = RecordingTransport(json.loads(response))
                with self.assertRaises(AIServiceError) as caught:
                    HttpGenerator("http://ia", "m", transport=transport).generate(REQUEST)
                self.assertEqual(str(caught.exception), message)
                self.assertFalse(caught.exception.transient)


# Les octets du corps des requêtes, ceux de json.dumps(payload) : ASCII seul (é en minuscules, un emoji en deux
# demi-paires, DEL et U+2028 échappés), « , » et « : » suivis d'une espace, un réel écrit comme son repr (« 1.0 »).
TEXTS = ["Congés « payés »\xa0\U0001F600", 'a"b\\c/\n\t\x7f\x01\u2028', "~ ascii"]
EMBEDDINGS_REQUEST = (b'{"model": "m", "input_type": "document", "inputs": ["Cong\\u00e9s \\u00ab pay\\u00e9s \\u00bb'
                      b'\\u00a0\\ud83d\\ude00", "a\\"b\\\\c/\\n\\t\\u007f\\u0001\\u2028", "~ ascii"]}')
GENERATION_REQUESTS = (
    (GenerationRequest(system="Réponds.", prompt="Q ?", temperature=1.0, max_tokens=400),
     b'{"model": "m", "system": "R\\u00e9ponds.", "prompt": "Q ?", "temperature": 1.0, "max_tokens": 400, "seed": null}'),
    (GenerationRequest(system="s", prompt="p", temperature=0.2, max_tokens=100, seed=7),
     b'{"model": "m", "system": "s", "prompt": "p", "temperature": 0.2, "max_tokens": 100, "seed": 7}'),
)


@contextlib.contextmanager
def recording(body: bytes):
    """Un faux service IA qui répond 200 et `body`, et garde les corps reçus : (adresse, corps reçus)."""
    received = []

    class Recording(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", received
    finally:
        server.shutdown()
        server.server_close()


class RequestBytesTest(unittest.TestCase):
    def test_the_request_bodies_are_frozen(self):
        with recording(b'{"model": "m", "dimension": 1, "vectors": [[1.0], [1.0], [1.0]]}') as (url, received):
            HttpEmbedder(url, "m", timeout=5).embed_documents(TEXTS)
        self.assertEqual(received, [EMBEDDINGS_REQUEST])
        for request, expected in GENERATION_REQUESTS:
            with self.subTest(seed=request.seed), recording(b'{"model": "m", "text": "ok"}') as (url, received):
                HttpGenerator(url, "m", timeout=5).generate(request)
            self.assertEqual(received, [expected])


if __name__ == "__main__":
    unittest.main()
