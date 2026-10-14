"""Embeddings: turn text into unit-length vectors.

bge models are trained to embed queries with an instruction prefix and documents without one,
so there are two methods. Embedding is CPU work: call these through asyncio.to_thread from async
code so the event loop (and the worker's lease heartbeat) keeps running.
"""

import hashlib
from typing import Protocol

import numpy as np

from research_assistant.ranking.dense import normalize
from research_assistant.text.tokenize import tokenize


class Embedder(Protocol):
    dim: int

    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 (384 dimensions) through fastembed, running on ONNX Runtime on the CPU.
    The model loads on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None, batch_size: int = 64):
        self.model_name, self.cache_dir, self.batch_size = model_name, cache_dir, batch_size
        self.dim = 384
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
        return self._model

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.array(list(self._load().embed(texts, batch_size=self.batch_size))))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.array(list(self._load().query_embed(texts))))


class HashEmbedder:
    """Deterministic stand-in for tests: hashes each word into one of `dim` buckets with a random sign.
    Texts that share words get similar vectors, and nothing is downloaded."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        return v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return normalize(np.array([self._vector(t) for t in texts], dtype=np.float32).reshape(len(texts), self.dim))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_docs(texts)
