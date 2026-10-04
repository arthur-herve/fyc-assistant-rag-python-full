import importlib.util
import math
import unittest
from unittest import mock

from ai_service.backends import ollama
from ai_service.backends.base import BackendError
from ai_service.backends.extractive import ExtractiveGenerationBackend
from ai_service.backends.hashing import HashingEmbeddingBackend
from ai_service.backends.http_json import request_json
from ai_service.backends.ollama import OllamaEmbeddingBackend, OllamaGenerationBackend
from ai_service.backends.openai_compatible import (
    OpenAICompatibleEmbeddingBackend, OpenAICompatibleGenerationBackend,
)
from ai_service.backends.sentence_transformers_backend import SentenceTransformersEmbeddingBackend
from ai_service.registry import ConfigError, ModelRegistry, UnknownModelError
from tests.ai_service.stub_servers import StubServer, ollama_routes, openai_routes, silent_after

PROMPT = """Passages :

[1] Télétravail
Télétravail
Chaque salarié peut télétravailler deux jours par semaine.

[2] Congés
Congés
Chaque salarié acquiert vingt-cinq jours de congés payés.

Question : Combien de jours de télétravail ?"""


class HashingBackendTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        backend = HashingEmbeddingBackend(dimension=64)
        first, second = backend.embed(["télétravail"]), backend.embed(["télétravail"])
        self.assertEqual(first.vectors, second.vectors)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in first.vectors[0])), 1.0)
        self.assertEqual(first.model_id, "hashing-64-stem6")

    def test_accents_do_not_matter(self):
        backend = HashingEmbeddingBackend()
        self.assertEqual(backend.embed(["Congés"]).vectors, backend.embed(["conges"]).vectors)


class ExtractiveBackendTest(unittest.TestCase):
    def test_picks_the_closest_sentence_and_cites_it(self):
        _, text = ExtractiveGenerationBackend().generate("", PROMPT, 0.2, 100, None)
        self.assertEqual(text, "Chaque salarié peut télétravailler deux jours par semaine. [1]")

    def test_noisy_mode_sometimes_forgets_citations(self):
        backend = ExtractiveGenerationBackend(citation_failure_rate=0.5, randomize=True)
        outputs = {backend.generate("", PROMPT, 0.2, 100, seed)[1] for seed in range(40)}
        self.assertTrue(any("[" not in o for o in outputs))
        self.assertTrue(any("[" in o for o in outputs))

    def test_same_seed_same_output(self):
        backend = ExtractiveGenerationBackend(citation_failure_rate=0.5, randomize=True)
        self.assertEqual(backend.generate("", PROMPT, 0.2, 100, 3), backend.generate("", PROMPT, 0.2, 100, 3))


class OllamaBackendTest(unittest.TestCase):
    def test_embeddings_request_format_and_model_digest(self):
        with StubServer(ollama_routes()) as stub:
            vectors = OllamaEmbeddingBackend("nomic-embed-text", base_url=stub.url).embed(["a", "b"])
            _, path, body = next(r for r in stub.requests if r[1] == "/api/embed")
        self.assertEqual(body, {"model": "nomic-embed-text", "input": ["a", "b"]})
        self.assertEqual(vectors.model_id, "ollama:nomic-embed-text@0a109f422b47")
        self.assertEqual(vectors.dimension, 3)

    def test_chat_request_format(self):
        with StubServer(ollama_routes()) as stub:
            model_id, text = OllamaGenerationBackend("qwen3:1.7b", base_url=stub.url, think=False).generate(
                "système", "prompt", 0.2, 150, 42)
            _, _, body = next(r for r in stub.requests if r[1] == "/api/chat")
        self.assertEqual(text, "<think>je réfléchis</think>\nDeux jours [1].")   # retiré par le registre
        self.assertEqual(model_id, "ollama:qwen3:1.7b@8f68893c685c")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "système"})
        self.assertEqual(body["options"], {"temperature": 0.2, "num_predict": 150, "seed": 42})
        self.assertIs(body["think"], False)
        self.assertIs(body["stream"], False)

    def test_thinking_models_get_a_separate_token_budget_and_thinking_is_discarded(self):
        with StubServer(ollama_routes(thinking=True)) as stub:
            _, text = OllamaGenerationBackend(
                "qwen3:4b", base_url=stub.url, think=True, thinking_tokens=1000,
            ).generate("système", "prompt", 0.2, 150, None)
            _, _, body = next(r for r in stub.requests if r[1] == "/api/chat")
        self.assertEqual(text, "Deux jours [1].")
        self.assertIs(body["think"], True)
        self.assertEqual(body["options"]["num_predict"], 1150)

    def test_thinking_budget_is_ignored_when_thinking_is_off(self):
        backend = OllamaGenerationBackend("x", think=False, thinking_tokens=1000)
        self.assertEqual(backend.thinking_tokens, 0)

    def test_the_digest_is_read_with_a_short_timeout(self):
        """/api/tags répond vite : 10 s y suffisent, et un Ollama figé se voit sans attendre le délai du
        modèle (120 s pour les embeddings, 300 s pour la génération), qui ne vaut que pour l'inférence."""
        calls = []

        def recorded(method, url, payload, timeout):
            calls.append((method, url.removeprefix(stub.url), timeout))
            return request_json(method, url, payload, timeout)

        with StubServer(ollama_routes()) as stub, mock.patch.object(ollama, "request_json", recorded):
            OllamaEmbeddingBackend("nomic-embed-text", base_url=stub.url).embed(["a"])
            OllamaGenerationBackend("qwen3:1.7b", base_url=stub.url).generate("s", "p", 0.2, 10, None)
        self.assertEqual(calls, [("GET", "/api/tags", 10), ("POST", "/api/embed", 120), ("GET", "/api/tags", 10),
                                 ("GET", "/api/tags", 10), ("POST", "/api/chat", 300), ("GET", "/api/tags", 10)])

    def test_unreachable_ollama_is_a_backend_error(self):
        # Le refus simulé là où il naît (socket.create_connection) : sous Windows, un vrai refus coûte 2 s. Le vrai
        # refus de ces tests : test_backend_failure_is_502 (test_server.py).
        refused = mock.patch("socket.create_connection", side_effect=ConnectionRefusedError("connexion refusée (simulée)"))
        with refused, self.assertRaises(BackendError) as caught:
            OllamaEmbeddingBackend("x", base_url="http://127.0.0.1:1", timeout=2).embed(["a"])
        self.assertTrue(caught.exception.retryable)   # Ollama peut revenir : réessayer a un sens

    def test_a_refusal_by_the_engine_is_not_worth_retrying(self):
        """4xx : le moteur refuse, réessayer n'y changera rien ; sauf 408 (délai dépassé) et 429 (trop
        de requêtes), qui disent justement de réessayer plus tard."""
        for status, retryable in ((400, False), (404, False), (408, True), (429, True), (500, True), (503, True)):
            routes = ollama_routes(tags=lambda: [{"name": "bge-m3:latest", "digest": "0123456789ab0000"}])
            routes[("POST", "/api/embed")] = lambda body, status=status: (status, {"error": "model not found"})
            with self.subTest(status), StubServer(routes) as stub:
                with self.assertRaises(BackendError) as caught:
                    OllamaEmbeddingBackend("bge-m3", base_url=stub.url).embed(["a"])
                self.assertEqual(caught.exception.retryable, retryable)

    def test_an_answer_not_shaped_like_ollama_is_a_definitive_error(self):
        """Proxy ou mauvaise base_url : une BackendError qui le dit, pas un AttributeError (500)."""
        installed = [{"name": "bge-m3:latest", "digest": "0123456789ab0000"}]
        cases = {
            "/api/tags en liste": {("GET", "/api/tags"): lambda body: (200, [])},
            "/api/tags sans liste": {("GET", "/api/tags"): lambda body: (200, {"models": None})},
            "modèle qui n'est pas un objet": {("GET", "/api/tags"): lambda body: (200, {"models": ["bge-m3"]})},
            "/api/embed en liste": {("POST", "/api/embed"): lambda body: (200, [[0.1], [0.2]])},
            "vecteurs à plat": {("POST", "/api/embed"): lambda body: (200, {"embeddings": [0.1, 0.2]})},
            "vecteurs en moins": {("POST", "/api/embed"): lambda body: (200, {"embeddings": [[0.1]]})},
        }
        for case, overrides in cases.items():
            with self.subTest(case), StubServer({**ollama_routes(tags=lambda: installed), **overrides}) as stub:
                with self.assertRaises(BackendError) as caught:
                    OllamaEmbeddingBackend("bge-m3", base_url=stub.url).embed(["a", "b"])
                self.assertFalse(caught.exception.retryable)   # le moteur répondra toujours pareil
        chat = {**ollama_routes(tags=lambda: installed),
                ("POST", "/api/chat"): lambda body: (200, {"message": {"content": 42}})}
        with StubServer(chat) as stub, self.assertRaises(BackendError) as caught:
            OllamaGenerationBackend("bge-m3", base_url=stub.url).generate("s", "p", 0.2, 10, None)
        self.assertFalse(caught.exception.retryable)

    def test_a_weights_update_is_seen_without_restarting_the_service(self):
        """ADR 0003 : un `ollama pull` derrière le même alias change l'identifiant."""
        installed = {"digest": "aaaaaaaaaaaa0000"}
        routes = ollama_routes(tags=lambda: [{"name": "bge-m3:latest", **installed},
                                             {"name": "qwen3:4b", **installed}])
        with StubServer(routes) as stub:
            embedder = OllamaEmbeddingBackend("bge-m3", base_url=stub.url)
            generator = OllamaGenerationBackend("qwen3:4b", base_url=stub.url)
            before = embedder.embed(["a"]).model_id, generator.generate("s", "p", 0.2, 10, None)[0]
            installed["digest"] = "bbbbbbbbbbbb0000"   # ollama pull : nouvelle version
            after = embedder.embed(["a"]).model_id, generator.generate("s", "p", 0.2, 10, None)[0]
        self.assertEqual(before, ("ollama:bge-m3@aaaaaaaaaaaa", "ollama:qwen3:4b@aaaaaaaaaaaa"))
        self.assertEqual(after, ("ollama:bge-m3@bbbbbbbbbbbb", "ollama:qwen3:4b@bbbbbbbbbbbb"))

    def test_a_pull_during_the_call_is_an_error(self):
        """On ne sait pas quelle version a servi : on échoue plutôt que d'étiqueter au hasard."""
        calls = {
            "/api/embed": lambda url: OllamaEmbeddingBackend("bge-m3", base_url=url).embed(["a"]),
            "/api/chat": lambda url: OllamaGenerationBackend("bge-m3", base_url=url).generate("s", "p", 0.2, 10, None),
        }
        for path, call in calls.items():
            with self.subTest(path=path):
                installed = {"digest": "aaaaaaaaaaaa0000"}
                routes = ollama_routes(tags=lambda: [{"name": "bge-m3:latest", **installed}])
                inference = routes[("POST", path)]

                def pull_then_infer(body, inference=inference, installed=installed):
                    installed["digest"] = "bbbbbbbbbbbb0000"   # ollama pull pendant l'inférence
                    return inference(body)

                routes[("POST", path)] = pull_then_infer
                with StubServer(routes) as stub, self.assertRaises(BackendError) as caught:
                    call(stub.url)
                self.assertIn("a changé pendant l'appel", str(caught.exception))
                self.assertTrue(caught.exception.retryable)   # le contrat le promet : il suffit de réessayer

    def test_no_digest_no_identifier(self):
        """Sans empreinte, on ne peut pas rattacher le résultat à un modèle : erreur, pas d'appel."""
        failing_tags = ollama_routes()
        failing_tags[("GET", "/api/tags")] = lambda body: (500, {"error": "chargement"})
        cases = {
            "modèle absent": ollama_routes(tags=lambda: []),
            "entrée sans empreinte": ollama_routes(tags=lambda: [{"name": "bge-m3:latest"}]),
            "empreinte nulle": ollama_routes(tags=lambda: [{"name": "bge-m3:latest", "digest": None}]),
            "/api/tags en erreur": failing_tags,
        }
        for case, routes in cases.items():
            with self.subTest(case), StubServer(routes) as stub:
                with self.assertRaises(BackendError) as caught:
                    OllamaEmbeddingBackend("bge-m3", base_url=stub.url).embed(["a"])
                self.assertNotIn("/api/embed", [path for _, path, _ in stub.requests])
                # Modèle absent : définitif ; Ollama en erreur passagère : réessayer a un sens.
                self.assertEqual(caught.exception.retryable, case == "/api/tags en erreur")

    def test_the_model_name_is_compared_the_way_ollama_does(self):
        """Casse, tag « latest » implicite, registre et espace « library » facultatifs."""
        tags = [{"name": "bge-m3:latest", "digest": "0123456789ab0000"}]
        with StubServer(ollama_routes(tags=lambda: tags)) as stub:
            for name in ("bge-m3", "BGE-M3", "bge-m3:LATEST", "registry.ollama.ai/library/bge-m3"):
                with self.subTest(name):
                    model_id = OllamaEmbeddingBackend(name, base_url=stub.url).embed(["a"]).model_id
                    self.assertEqual(model_id, f"ollama:{name}@0123456789ab")
        only_model = [{"model": "bge-m3:latest", "digest": "0123456789ab0000"}]   # sans « name » : « model » suffit
        with StubServer(ollama_routes(tags=lambda: only_model)) as stub:
            self.assertEqual(OllamaEmbeddingBackend("bge-m3", base_url=stub.url).embed(["a"]).model_id,
                             "ollama:bge-m3@0123456789ab")


class SentenceTransformersBackendTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("sentence_transformers"), "sentence-transformers est installé")
    def test_a_missing_library_is_not_worth_retrying(self):
        with self.assertRaises(BackendError) as caught:
            SentenceTransformersEmbeddingBackend("modele").embed(["a"])
        self.assertFalse(caught.exception.retryable)
        self.assertIn("pip install", str(caught.exception))


class OpenAICompatibleBackendTest(unittest.TestCase):
    def test_embeddings_are_reordered_by_index(self):
        with StubServer(openai_routes()) as stub:
            vectors = OpenAICompatibleEmbeddingBackend("m", base_url=stub.url + "/v1").embed(["a", "b"])
        self.assertEqual(vectors.vectors, [[0.0, 1.0], [1.0, 1.0]])

    def test_chat(self):
        with StubServer(openai_routes()) as stub:
            _, text = OpenAICompatibleGenerationBackend("m", base_url=stub.url + "/v1").generate(
                "s", "p", 0.1, 50, None)
            _, _, body = stub.requests[0]
        self.assertEqual(text, "Réponse [1]")
        self.assertNotIn("seed", body)

    def test_a_cut_or_non_http_answer_is_a_backend_error(self):
        """Un moteur coupé en pleine réponse peut réussir au prochain essai ; un service qui ne parle
        pas HTTP (base_url vers autre chose) répondra toujours pareil. Dans les deux cas : 502, pas 500."""
        cases = {
            "réponse coupée": (b"HTTP/1.0 200 OK\r\nContent-Length: 500\r\n\r\n" + b'{"data": [', True),
            "pas du HTTP": (b"garbage-not-http\r\n", False),
        }
        for case, (answer, retryable) in cases.items():
            with self.subTest(case), StubServer({("POST", "/v1/embeddings"): lambda body, answer=answer: answer}) as stub:
                with self.assertRaises(BackendError) as caught:
                    OpenAICompatibleEmbeddingBackend("m", base_url=stub.url + "/v1").embed(["a"])
                self.assertEqual(caught.exception.retryable, retryable)

    def test_an_error_that_stops_or_is_cut_in_its_body_is_retryable(self):
        """Le moteur envoie ses en-têtes d'erreur, puis se tait (délai dépassé) ou coupe : comme au milieu
        d'une réponse 200, une BackendError réessayable (502 backend_error), pas une exception de la
        bibliothèque standard (500, avec sa pile)."""
        error = b"HTTP/1.1 500 Erreur\r\nContent-Length: 100\r\n\r\n{"
        cases = {
            "silence": (error, False, "injoignable : timed out"),
            "coupure": (error, True, "a coupé sa réponse : IncompleteRead("),
            "coupure en morceaux": (b"HTTP/1.1 500 Erreur\r\nTransfer-Encoding: chunked\r\n\r\n10\r\n{", True,
                                    "a coupé sa réponse : IncompleteRead("),
        }
        for case, (start, cut, message) in cases.items():
            with self.subTest(case), silent_after(start, cut) as url:
                with self.assertRaises(BackendError) as caught:
                    OpenAICompatibleEmbeddingBackend("m", base_url=url + "/v1", timeout=0.5).embed(["a"])
                self.assertIn(f"{url}/v1/embeddings {message}", str(caught.exception))
                self.assertTrue(caught.exception.retryable)


class RegistryTest(unittest.TestCase):
    def test_prefixes_are_applied_by_the_service_not_the_application(self):
        registry = ModelRegistry.from_dict({"embedding": {"e5": {
            "backend": "hashing", "dimension": 16, "query_prefix": "query: ", "document_prefix": "passage: "}}})
        model = registry.embedding("e5")
        direct = HashingEmbeddingBackend(dimension=16)
        self.assertEqual(model.embed(["bonjour"], "query").vectors, direct.embed(["query: bonjour"]).vectors)
        self.assertEqual(model.embed(["bonjour"], "document").vectors, direct.embed(["passage: bonjour"]).vectors)

    def test_prefixes_are_part_of_the_model_identity(self):
        """Changer un préfixe change les vecteurs : l'identifiant doit changer aussi (ADR 0003)."""
        def model_id(**prefixes):
            registry = ModelRegistry.from_dict({"embedding": {"e5": {"backend": "hashing", "dimension": 16, **prefixes}}})
            return registry.embedding("e5").embed(["bonjour"], "query").model_id

        plain = model_id()
        variants = [
            model_id(query_prefix="query: ", document_prefix="passage: "),
            model_id(query_prefix="search_query: ", document_prefix="passage: "),
            model_id(query_prefix="query: ", document_prefix="document: "),   # celui des documents façonne l'index
            model_id(query_prefix="query: "),                                 # requête seule, comme mxbai
            model_id(document_prefix="passage: "),
        ]
        self.assertEqual(plain, "hashing-16-stem6")
        self.assertEqual(len({plain, *variants}), 6)
        self.assertTrue(all(v.startswith("hashing-16-stem6+prefixes-") for v in variants))

    def test_reasoning_tags_are_removed_whatever_the_engine(self):
        """Une particularité de modèle (qwen3…), pas de moteur : Ollama comme serveur compatible OpenAI."""
        leaked = "<think>je réfléchis</think>\nRéponse [1]"
        ollama = ollama_routes()
        ollama[("POST", "/api/chat")] = lambda body: (200, {"message": {"role": "assistant", "content": leaked}})
        openai = openai_routes()
        openai[("POST", "/v1/chat/completions")] = lambda body: (200, {"choices": [{"message": {"content": leaked}}]})
        for backend, routes, suffix in (("ollama", ollama, ""), ("openai-compatible", openai, "/v1")):
            with self.subTest(backend), StubServer(routes) as stub:
                registry = ModelRegistry.from_dict({"generation": {"m": {
                    "backend": backend, "model": "qwen3:1.7b", "base_url": stub.url + suffix}}})
                _, text = registry.generation("m").generate("s", "p", 0.2, 50, None)
                self.assertEqual(text, "Réponse [1]")

    def test_unknown_model(self):
        with self.assertRaises(UnknownModelError):
            ModelRegistry.from_dict({}).generation("gpt-9")

    def test_unknown_backend(self):
        with self.assertRaises(ConfigError):
            ModelRegistry.from_dict({"generation": {"x": {"backend": "magie"}}})

    def test_project_configuration_is_valid(self):
        from pathlib import Path
        registry = ModelRegistry.from_file(Path(__file__).parents[2] / "config/ai_service.toml")
        self.assertIn("hashing", [m["alias"] for m in registry.describe()["embedding"]])


if __name__ == "__main__":
    unittest.main()
