"""Vector helpers: unit-length normalization, cosine similarity, and Rocchio query feedback."""

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale each vector (last axis) to length 1, so a dot product equals cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity: a is (n, d), b is (m, d), result is (n, m)."""
    return normalize(a) @ normalize(b).T


def rocchio(
    query: np.ndarray,
    relevant: np.ndarray,
    nonrelevant: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.0,
) -> np.ndarray:
    """q' = alpha * q + beta * mean(relevant) - gamma * mean(nonrelevant), then back to unit length.

    Pseudo-relevance feedback: treating the top early hits as relevant moves a vague query
    toward what good results actually look like.
    """
    q = alpha * normalize(query)
    if len(relevant):
        q = q + beta * normalize(relevant).mean(axis=0)
    if gamma and nonrelevant is not None and len(nonrelevant):
        q = q - gamma * normalize(nonrelevant).mean(axis=0)
    return normalize(q)
