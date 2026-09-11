"""Backends Ollama (https://ollama.com) : modèles ouverts exécutés en local.

API utilisée : POST /api/embed, POST /api/chat, GET /api/tags.
"""

from __future__ import annotations

import re
import threading
from typing import Sequence

from .base import BackendError, Vectors
from .http_json import request_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class _OllamaModel:
    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_id: str | None = None
        self._lock = threading.Lock()

    def model_id(self) -> str:
        """Identifiant avec l'empreinte du modèle : un `ollama pull` qui met à jour
        les poids change l'identifiant, et l'application le détecte."""
        with self._lock:
            if self._model_id is None:
                digest = ""
                try:
                    tags = request_json("GET", f"{self.base_url}/api/tags", None, 10)
                    wanted = {self.model, f"{self.model}:latest"}
                    for entry in tags.get("models", []):
                        if entry.get("name") in wanted or entry.get("model") in wanted:
                            digest = entry.get("digest", "")[:12]
                            break
                except BackendError:
                    pass
                if not digest:
                    # Ollama n'a pas répondu (chargement en cours ?) : on réessaiera au prochain
                    # appel plutôt que de figer un identifiant sans empreinte pour tout le processus.
                    return f"ollama:{self.model}"
                self._model_id = f"ollama:{self.model}@{digest}"
            return self._model_id


def _call(model: _OllamaModel, path: str, payload: dict) -> dict:
    try:
        return request_json("POST", f"{model.base_url}{path}", payload, model.timeout)
    except BackendError as error:
        raise BackendError(
            f"{error} — Ollama est-il lancé sur {model.base_url} ? "
            f"Le modèle est-il téléchargé (ollama pull {model.model}) ?"
        ) from error


class OllamaEmbeddingBackend(_OllamaModel):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 120.0) -> None:
        super().__init__(model, base_url, timeout)

    def embed(self, texts: Sequence[str]) -> Vectors:
        data = _call(self, "/api/embed", {"model": self.model, "input": list(texts)})
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise BackendError(f"Ollama n'a pas renvoyé {len(texts)} embeddings")
        return Vectors(model_id=self.model_id(), dimension=len(vectors[0]), vectors=vectors)


class OllamaGenerationBackend(_OllamaModel):
    """Génération via /api/chat.

    Modèles à « réflexion » (qwen3…) : mesuré avec Ollama 0.34 et qwen3:4b,
    `think = false` ne supprime pas le raisonnement, il le déverse dans la
    réponse elle-même (« Okay, let's see… »), sans balise. Avec `think = true`,
    Ollama renvoie le raisonnement à part (`message.thinking`) et la réponse
    dans `message.content`. On garde donc la réflexion activée, on l'ignore, et
    on lui accorde un budget de jetons séparé (`thinking_tokens`) : sinon elle
    consomme tout `max_tokens` et la réponse est vide.
    """

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 300.0, think: bool | None = None,
                 thinking_tokens: int = 0, keep_alive: str | None = None) -> None:
        super().__init__(model, base_url, timeout)
        self.think = think
        self.thinking_tokens = thinking_tokens if think else 0
        self.keep_alive = keep_alive

    def generate(self, system, prompt, temperature, max_tokens, seed):
        # Le budget de réflexion s'ajoute à max_tokens : la réponse peut donc dépasser
        # max_tokens si le modèle réfléchit peu (assumé, documenté dans docs/contrat-http.md).
        options = {"temperature": temperature, "num_predict": max_tokens + self.thinking_tokens}
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": options,
        }
        if self.think is not None:
            payload["think"] = self.think  # modèles à « réflexion » (qwen3…)
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        data = _call(self, "/api/chat", payload)
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError):
            raise BackendError(f"Réponse Ollama inattendue : {str(data)[:300]}") from None
        # Particularité de certains modèles qui fuit jusqu'ici : on la neutralise.
        # (`message.thinking`, renvoyé à part par Ollama, est simplement ignoré.)
        return self.model_id(), _THINK.sub("", content).strip()
