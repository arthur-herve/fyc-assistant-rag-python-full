"""Backends pour tout serveur exposant l'API « compatible OpenAI ».

C'est le cas de LM Studio, du serveur de llama.cpp et de vLLM (courant sur
les serveurs GPU d'entreprise). Aucun compte payant n'est nécessaire pour
les serveurs locaux.
"""

from __future__ import annotations

import os
from typing import Sequence

from .base import BackendError, Vectors
from .http_json import request_json


class _OpenAICompatible:
    def __init__(self, model: str, base_url: str, timeout: float,
                 api_key_env: str | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.model}"


class OpenAICompatibleEmbeddingBackend(_OpenAICompatible):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:1234/v1",
                 timeout: float = 120.0, api_key_env: str | None = None) -> None:
        super().__init__(model, base_url, timeout, api_key_env)

    def embed(self, texts: Sequence[str]) -> Vectors:
        data = request_json(
            "POST", f"{self.base_url}/embeddings",
            {"model": self.model, "input": list(texts)}, self.timeout, self.headers,
        )
        try:
            items = sorted(data["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError):
            raise BackendError(f"Réponse embeddings inattendue : {str(data)[:300]}") from None
        if len(vectors) != len(texts):
            raise BackendError(f"{len(texts)} embeddings attendus, {len(vectors)} reçus")
        return Vectors(model_id=self.model_id, dimension=len(vectors[0]), vectors=vectors)


class OpenAICompatibleGenerationBackend(_OpenAICompatible):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:1234/v1",
                 timeout: float = 300.0, api_key_env: str | None = None) -> None:
        super().__init__(model, base_url, timeout, api_key_env)

    def generate(self, system, prompt, temperature, max_tokens, seed):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        data = request_json(
            "POST", f"{self.base_url}/chat/completions", payload, self.timeout, self.headers
        )
        try:
            return self.model_id, data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            raise BackendError(f"Réponse chat inattendue : {str(data)[:300]}") from None
