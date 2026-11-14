"""Embedding back-ends.

* ``tfidf``  - local, deterministic, no network: hashed uni+bi-gram TF-IDF (default)
* ``openai`` - OpenAI embeddings API (needs OPENAI_API_KEY and `pip install openai`)
"""
from __future__ import annotations

import math
import os
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from .textutils import tokenize


class EmbeddingError(RuntimeError):
    pass


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingTfidfEmbedder:
    """Hashed TF-IDF vectors (fixed dimensionality, so memory stays bounded)."""

    backend = "tfidf"

    def __init__(self, dim: int = 4096, idf: Optional[np.ndarray] = None):
        self.dim = dim
        self.idf = idf
        self.model = f"hashing-tfidf-{dim}"

    @staticmethod
    def _features(text: str) -> List[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode("utf-8")) % self.dim

    def _counts(self, text: str) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for feature in self._features(text):
            b = self._bucket(feature)
            counts[b] = counts.get(b, 0) + 1
        return counts

    def fit(self, texts: Sequence[str]) -> "HashingTfidfEmbedder":
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for bucket in self._counts(text):
                df[bucket] += 1
        n = len(texts)
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        return self

    def _vectorize(self, text: str) -> np.ndarray:
        if self.idf is None:
            raise EmbeddingError("TF-IDF embedder has not been fitted")
        vec = np.zeros(self.dim, dtype=np.float32)
        for bucket, count in self._counts(text).items():
            vec[bucket] = (1.0 + math.log(count)) * self.idf[bucket]
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize_rows(np.vstack([self._vectorize(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize_rows(self._vectorize(text)[None, :])[0]

    def arrays(self) -> Dict[str, np.ndarray]:
        return {"idf": self.idf} if self.idf is not None else {}


class OpenAIEmbedder:
    backend = "openai"

    def __init__(self, model: str = "text-embedding-3-small", client=None):
        self.model = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError("pip install openai to use OpenAI embeddings") from exc
            if not os.getenv("OPENAI_API_KEY"):
                raise EmbeddingError("OPENAI_API_KEY is not set")
            client = OpenAI()
        self.client = client

    def fit(self, texts: Sequence[str]) -> "OpenAIEmbedder":
        return self

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), 64):
            batch = [t[:8000] for t in texts[start : start + 64]]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:
                raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc
            vectors.extend(item.embedding for item in response.data)
        return _normalize_rows(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def arrays(self) -> Dict[str, np.ndarray]:
        return {}


def make_embedder(backend: str, model: Optional[str] = None, idf: Optional[np.ndarray] = None, dim: int = 4096):
    backend = (backend or "tfidf").lower()
    if backend == "tfidf":
        return HashingTfidfEmbedder(dim=dim, idf=idf)
    if backend == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    raise EmbeddingError(f"Unknown embedding backend: {backend!r} (use 'tfidf' or 'openai')")
