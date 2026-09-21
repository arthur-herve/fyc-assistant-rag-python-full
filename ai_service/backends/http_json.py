from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import BackendError


def request_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float,
                 headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
        # 4xx : le moteur refuse (modèle absent…), réessayer n'y changera rien.
        raise BackendError(f"{url} a répondu HTTP {error.code} : {detail}",
                           retryable=error.code >= 500 or error.code == 429) from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise BackendError(f"{url} injoignable : {error}") from error
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as error:   # JSON invalide ou octets non UTF-8
        raise BackendError(f"{url} a répondu autre chose que du JSON : {error}", retryable=False) from error
