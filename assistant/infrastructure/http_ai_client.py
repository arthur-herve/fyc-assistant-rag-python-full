"""Adaptateurs : le modèle est derrière une frontière réseau.

En entreprise, les machines équipées de GPU hébergent les modèles et les
serveurs applicatifs hébergent l'application. Ces adaptateurs traduisent
les ports `Embedder` et `Generator` en appels HTTP vers le service IA.

Le contrat est décrit dans docs/contrat-http.md.
"""

from __future__ import annotations

import http.client
import json
import math
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence

from assistant.application.errors import AIServiceError
from assistant.application.ports import EmbeddingBatch, Generation, GenerationRequest

from .json_text import parse
from .text_files import decode_utf8

# Une fonction (url, payload) -> réponse JSON. Remplaçable dans les tests.
Transport = Callable[[str, dict[str, Any]], Any]


def _is_text(value: Any) -> bool:
    """Un texte Unicode. json.loads garde une demi-paire de substitution seule (« \\ud800 »), qui
    ne s'écrit pas en UTF-8, pas même dans un message d'erreur."""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _problem(body: str) -> tuple[str, bool] | None:
    """Le message et « retryable » d'une erreur au format du contrat ({"error": {"message": …}}),
    ou None si le corps a une autre forme : il est alors cité tel quel. Le message n'est retenu que
    si c'est un texte, comme le dit le contrat (docs/contrat-http.md)."""
    try:
        problem = parse(body)["error"]
        message, retryable = problem["message"], problem.get("retryable", True) is not False
    except (ValueError, KeyError, TypeError):
        return None
    return (message, retryable) if _is_text(message) else None


def urllib_transport(timeout: float) -> Transport:
    def post(url: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            try:
                response = urllib.request.urlopen(request, timeout=timeout)
            except urllib.error.HTTPError as error:   # une erreur : son corps se lit comme celui d'une réponse 200
                response = error
            with response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            # Injoignable, ou délai dépassé et coupure jusque dans le corps de la réponse (d'erreur ou non) :
            # passager, comme le dit le contrat.
            raise AIServiceError(f"Service IA injoignable ({url}) : {error}") from error
        except http.client.InvalidURL as error:
            # Adresse mal écrite (AI_SERVICE_URL avec un port qui n'est pas un nombre…) : rien n'a été contacté.
            raise AIServiceError(f"Adresse du service IA invalide ({url}) : {error}", transient=False) from error
        except http.client.HTTPException as error:
            # Une réponse qui n'est pas du HTTP (AI_SERVICE_URL vers un mauvais port, un autre programme) : elle
            # reviendra à chaque essai. Non passagère, comme le dit le contrat pour un moteur. Cette clause vient après
            # la précédente, qui garde passagères IncompleteRead et RemoteDisconnected, des HTTPException elles aussi.
            raise AIServiceError(f"Service IA ({url}) n'a pas répondu en HTTP : {error!r}", transient=False) from error
        if isinstance(response, urllib.error.HTTPError):
            # Octets invalides remplacés ; marque d'ordre des octets retirée, comme pour une réponse 200.
            detail = body.decode("utf-8-sig", errors="replace")
            detail, retryable = _problem(detail) or (detail, True)
            # 5xx : passager, sauf si le service dit que réessayer ne changera rien (modèle absent…).
            raise AIServiceError(f"Service IA : HTTP {response.code} — {detail}",
                                 transient=response.code >= 500 and retryable) from response
        try:
            # Marque d'ordre des octets acceptée, comme le permet le contrat. Puis JSON strict (json_text).
            return parse(decode_utf8(body))
        except ValueError as error:
            # JSON qui n'est pas strict (syntaxe, clé en double, NaN, entier de plus de 4300 chiffres, plus de
            # 900 niveaux), octets non UTF-8 : ce n'est pas le contrat.
            raise AIServiceError(f"Réponse du service IA illisible ({url}) : {error}",
                                 transient=False) from error

    return post


# Une réponse d'une autre forme que celle du contrat (null, nombre, champ mal typé…) est une
# AIServiceError non passagère, jamais une trace d'erreur : le client ne réessaie pas (docs/contrat-http.md).

def _require(payload: Any, *keys: str) -> None:
    missing = [k for k in keys if not isinstance(payload, dict) or k not in payload]
    if missing:
        raise AIServiceError(f"Réponse du service IA incomplète, champs manquants : [{', '.join(missing)}]",
                             transient=False)


def _incoherent(problem: str) -> AIServiceError:
    return AIServiceError(f"Réponse du service IA incohérente : {problem}", transient=False)


def _text(payload: dict[str, Any], key: str) -> str:
    if not _is_text(payload[key]):
        raise _incoherent(f"« {key} » doit être un texte")
    return payload[key]


def _is_int32(value: Any) -> bool:
    """Un entier de 32 bits, comme le veut le contrat pour `dimension` : ni booléen (True est un int en Python),
    ni 2.0."""
    return isinstance(value, int) and not isinstance(value, bool) and -2**31 <= value < 2**31


def _is_number(value: Any) -> bool:
    """Un nombre fini : ni booléen (True est un int en Python), ni l'infini que donne « 1e400 »."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:   # un entier de 400 chiffres
        return False


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
        model, dimension, vectors = _text(payload, "model"), payload["dimension"], payload["vectors"]
        if not _is_int32(dimension):
            raise _incoherent("« dimension » doit être un entier")
        if not isinstance(vectors, list) or not all(
            isinstance(v, list) and all(map(_is_number, v)) for v in vectors
        ):
            raise _incoherent("« vectors » doit être une liste de listes de nombres")
        if len(vectors) != len(texts) or any(len(v) != dimension for v in vectors):
            raise AIServiceError(
                f"Réponse du service IA incohérente : {len(texts)} vecteurs de {dimension} dimensions "
                f"attendus, reçu {len(vectors)} de {sorted({len(v) for v in vectors})} dimensions",
                transient=False)
        return EmbeddingBatch(model=model, dimension=dimension, vectors=vectors)


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
        return Generation(model=_text(payload, "model"), text=_text(payload, "text"))
