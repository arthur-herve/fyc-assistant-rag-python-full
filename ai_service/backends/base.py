"""Contrats internes du service IA : un backend sait produire des vecteurs ou du texte."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class BackendError(RuntimeError):
    """Le moteur sous-jacent (Ollama, bibliothèque locale…) a échoué.

    `retryable` : réessayer peut-il réussir ? Oui pour un moteur injoignable ou surchargé,
    non pour un modèle absent ou une réponse que le moteur renverra toujours pareille.
    """

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Vectors:
    model_id: str
    dimension: int
    vectors: list[list[float]]


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> Vectors: ...


class GenerationBackend(Protocol):
    def generate(self, system: str, prompt: str, temperature: float,
                 max_tokens: int, seed: int | None) -> tuple[str, str]:
        """Renvoie (identifiant du modèle, texte)."""
        ...
