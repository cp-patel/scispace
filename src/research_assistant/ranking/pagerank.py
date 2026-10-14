"""Personalized PageRank on a citation graph, solved by power iteration."""

import numpy as np


def personalized_pagerank(
    n: int,
    edges: np.ndarray | list[tuple[int, int]],
    teleport: np.ndarray,
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, int]:
    """pi = (1 - alpha) * v + alpha * P^T pi.

    Picture a reader who starts at a strong paper (chosen by the teleport vector v) and keeps
    following references. With probability 1 - alpha they jump back to a strong paper.
    Papers the reader keeps landing on are what the topic builds on.

    edges: (src, dst) pairs meaning "src cites dst". Self-loops and duplicates are ignored.
    Papers that cite nothing in the graph (dangling) send their mass back to v, so pi sums to 1.
    The L1 error shrinks by a factor of alpha each step.
    Returns (pi, iterations used).
    """
    v = np.asarray(teleport, dtype=float)
    if v.shape != (n,) or (v < 0).any() or v.sum() <= 0:
        raise ValueError("teleport must be a non-negative vector of length n with a positive sum")
    v = v / v.sum()
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    if len(e):
        e = np.unique(e, axis=0)
    src, dst = e[:, 0], e[:, 1]
    out_degree = np.bincount(src, minlength=n).astype(float)
    dangling = out_degree == 0
    pi = v.copy()
    for iteration in range(1, max_iter + 1):
        flow = np.zeros(n)
        if len(e):
            np.add.at(flow, dst, pi[src] / out_degree[src])
        new = (1.0 - alpha) * v + alpha * (flow + pi[dangling].sum() * v)
        if np.abs(new - pi).sum() < tol:
            return new, iteration
        pi = new
    return pi, max_iter
