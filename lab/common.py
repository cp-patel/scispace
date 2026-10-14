"""Shared helpers for the vector search lab: exact search, recall, and synthetic test data."""

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first."""
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def exact_search(X: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """The answer key: score every vector (one matrix-vector product) and keep the top k."""
    return top_k_indices(X @ q, k)


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    """Share of the true top-k that the approximate method found."""
    return len(set(found.tolist()) & set(truth.tolist())) / len(truth)


def gaussian_mixture(n: int, dim: int, centers: int = 20, spread: float = 0.15, seed: int = 0) -> np.ndarray:
    """Unit vectors grouped around random centers. Real embeddings are clustered like this;
    uniform random vectors are not (in high dimensions all their distances look alike)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(centers, dim))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    X = c[rng.integers(centers, size=n)] + spread * rng.normal(size=(n, dim)) / np.sqrt(dim) * 4
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
