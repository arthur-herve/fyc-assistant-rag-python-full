"""Adaptateurs : le modèle est derrière une frontière réseau.

En entreprise, les machines équipées de GPU hébergent les modèles et les
serveurs applicatifs hébergent l'application. Ces adaptateurs traduisent
les ports `Embedder` et `Generator` en appels HTTP vers le service IA.

Le contrat est décrit dans docs/contrat-http.md.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence

from assistant.application.errors import AIServiceError
from assistant.application.ports import EmbeddingBatch, Generation, GenerationRequest

# Une fonction (url, payload) -> réponse JSON. Remplaçable dans les tests.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def urllib_transport(timeout: float) -> Transport:
    def post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise AIServiceError(f"Service IA : HTTP {error.code} — {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise AIServiceError(f"Service IA injoignable ({url}) : {error}") from error

    return post


def _require(payload: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise AIServiceError(f"Réponse du service IA incomplète, champs manquants : {missing}")


class HttpEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0,
                 transport: Transport | None = None) -> None:
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._model = model
        self._post = transport or urllib_transport(timeout)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._embed(list(texts), "document")

    def embed_query(self, text: str) -> EmbeddingBatch:
        return self._embed([text], "query")

    def _embed(self, texts: list[str], input_type: str) -> EmbeddingBatch:
        payload = self._post(
            self._url, {"model": self._model, "input_type": input_type, "inputs": texts}
        )
        _require(payload, "model", "dimension", "vectors")
        return EmbeddingBatch(
            model=payload["model"],
            dimension=int(payload["dimension"]),
            vectors=payload["vectors"],
        )


class HttpGenerator:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 transport: Transport | None = None) -> None:
        self._url = base_url.rstrip("/") + "/v1/generate"
        self._model = model
        self._post = transport or urllib_transport(timeout)

    def generate(self, request: GenerationRequest) -> Generation:
        payload = self._post(
            self._url,
            {
                "model": self._model,
                "system": request.system,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "seed": request.seed,
            },
        )
        _require(payload, "model", "text")
        return Generation(model=payload["model"], text=payload["text"])
