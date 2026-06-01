from __future__ import annotations

from hashlib import blake2b
import math
import re
from typing import Protocol

from app.config import Settings
from app.services.gemini import GeminiClient


class Embedder(Protocol):
    async def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        ...

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        ...


class GeminiEmbedder:
    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini

    async def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        return await self.gemini.embed_texts(texts, task_type=task_type)

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        return await self.gemini.embed_text(text, task_type=task_type)


class LocalHashEmbedder:
    """Deterministic local embeddings for quota-free retrieval demos."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [
            self._stem(token)
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in self.STOPWORDS
        ]
        if not tokens:
            return vector

        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend(
            (f"{left}_{right}", 1.35)
            for left, right in zip(tokens, tokens[1:], strict=False)
        )

        for feature, weight in features:
            digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _stem(token: str) -> str:
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 5 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("s"):
            return token[:-1]
        return token


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, lazy_load=True)

    async def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        prepared = [self._prepare(text, task_type) for text in texts]
        embeddings = self.model.embed(prepared, batch_size=32)
        return [embedding.tolist() for embedding in embeddings]

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        return (await self.embed_texts([text], task_type=task_type))[0]

    @staticmethod
    def _prepare(text: str, task_type: str) -> str:
        if "QUERY" in task_type.upper():
            return f"query: {text}"
        return f"passage: {text}"


def create_embedder(settings: Settings, gemini: GeminiClient) -> Embedder:
    provider = settings.embedding_provider.lower().strip()
    if provider == "gemini":
        return GeminiEmbedder(gemini)
    if provider == "fastembed":
        return FastEmbedder(settings.fastembed_model)
    if provider == "local":
        return LocalHashEmbedder(settings.local_embedding_dimensions)
    raise ValueError("EMBEDDING_PROVIDER must be 'fastembed', 'local', or 'gemini'.")
