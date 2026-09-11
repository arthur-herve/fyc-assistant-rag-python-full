"""Registre des modèles servis, construit à partir de config/ai_service.toml.

L'application demande un *alias* (« nomic », « qwen3-4b ») ; le registre
choisit le backend et applique les particularités du modèle, comme les
préfixes « query: » / « passage: » exigés par certains modèles d'embeddings.
"""

from __future__ import annotations

import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .backends.base import EmbeddingBackend, GenerationBackend, Vectors


class UnknownModelError(KeyError):
    pass


class ConfigError(ValueError):
    pass


def _embedding_factory(backend: str, params: dict[str, Any], defaults: dict[str, Any]):
    if backend == "hashing":
        from .backends.hashing import HashingEmbeddingBackend
        return lambda: HashingEmbeddingBackend(**params)
    if backend == "ollama":
        from .backends.ollama import OllamaEmbeddingBackend
        return lambda: OllamaEmbeddingBackend(**{**defaults.get("ollama", {}), **params})
    if backend == "openai-compatible":
        from .backends.openai_compatible import OpenAICompatibleEmbeddingBackend
        return lambda: OpenAICompatibleEmbeddingBackend(
            **{**defaults.get("openai-compatible", {}), **params})
    if backend == "sentence-transformers":
        from .backends.sentence_transformers_backend import SentenceTransformersEmbeddingBackend
        return lambda: SentenceTransformersEmbeddingBackend(**params)
    raise ConfigError(f"backend d'embeddings inconnu : {backend}")


def _generation_factory(backend: str, params: dict[str, Any], defaults: dict[str, Any]):
    if backend == "extractive":
        from .backends.extractive import ExtractiveGenerationBackend
        return lambda: ExtractiveGenerationBackend(**params)
    if backend == "ollama":
        from .backends.ollama import OllamaGenerationBackend
        return lambda: OllamaGenerationBackend(**{**defaults.get("ollama", {}), **params})
    if backend == "openai-compatible":
        from .backends.openai_compatible import OpenAICompatibleGenerationBackend
        return lambda: OpenAICompatibleGenerationBackend(
            **{**defaults.get("openai-compatible", {}), **params})
    raise ConfigError(f"backend de génération inconnu : {backend}")


class _Lazy:
    """Instancie un backend au premier usage (charger un modèle coûte cher)."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._instance = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._instance is None:
                self._instance = self._factory()
            return self._instance


@dataclass
class EmbeddingModel:
    alias: str
    backend_name: str
    description: str
    query_prefix: str
    document_prefix: str
    _backend: _Lazy = field(repr=False)

    def embed(self, texts: Sequence[str], input_type: str) -> Vectors:
        prefix = self.query_prefix if input_type == "query" else self.document_prefix
        backend: EmbeddingBackend = self._backend.get()
        return backend.embed([prefix + t for t in texts])


@dataclass
class GenerationModel:
    alias: str
    backend_name: str
    description: str
    _backend: _Lazy = field(repr=False)

    def generate(self, system, prompt, temperature, max_tokens, seed) -> tuple[str, str]:
        backend: GenerationBackend = self._backend.get()
        return backend.generate(system, prompt, temperature, max_tokens, seed)


class ModelRegistry:
    def __init__(self, embedding: dict[str, EmbeddingModel],
                 generation: dict[str, GenerationModel]) -> None:
        self._embedding = embedding
        self._generation = generation

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelRegistry":
        defaults = config.get("defaults", {})
        embedding: dict[str, EmbeddingModel] = {}
        for alias, raw in config.get("embedding", {}).items():
            params = dict(raw)
            backend = params.pop("backend", None)
            if not backend:
                raise ConfigError(f"embedding.{alias} : champ 'backend' obligatoire")
            query_prefix = params.pop("query_prefix", "")
            document_prefix = params.pop("document_prefix", "")
            description = params.pop("description", "")
            embedding[alias] = EmbeddingModel(
                alias, backend, description, query_prefix, document_prefix,
                _Lazy(_embedding_factory(backend, params, defaults)),
            )
        generation: dict[str, GenerationModel] = {}
        for alias, raw in config.get("generation", {}).items():
            params = dict(raw)
            backend = params.pop("backend", None)
            if not backend:
                raise ConfigError(f"generation.{alias} : champ 'backend' obligatoire")
            description = params.pop("description", "")
            generation[alias] = GenerationModel(
                alias, backend, description,
                _Lazy(_generation_factory(backend, params, defaults)),
            )
        return cls(embedding, generation)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRegistry":
        with open(path, "rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    def embedding(self, alias: str) -> EmbeddingModel:
        try:
            return self._embedding[alias]
        except KeyError:
            raise UnknownModelError(
                f"modèle d'embeddings inconnu : {alias} (disponibles : {sorted(self._embedding)})"
            ) from None

    def generation(self, alias: str) -> GenerationModel:
        try:
            return self._generation[alias]
        except KeyError:
            raise UnknownModelError(
                f"modèle de génération inconnu : {alias} (disponibles : {sorted(self._generation)})"
            ) from None

    def describe(self) -> dict[str, list[dict[str, str]]]:
        return {
            "embedding": [
                {"alias": m.alias, "backend": m.backend_name, "description": m.description}
                for m in self._embedding.values()
            ],
            "generation": [
                {"alias": m.alias, "backend": m.backend_name, "description": m.description}
                for m in self._generation.values()
            ],
        }
