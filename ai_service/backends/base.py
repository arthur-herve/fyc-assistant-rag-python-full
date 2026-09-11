"""Contrats internes du service IA : un backend sait produire des vecteurs ou du texte."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class BackendError(RuntimeError):
    """Le moteur sous-jacent (Ollama, bibliothèque locale…) a échoué."""


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
