"""Determinantal Point Process (DPP) selection.

P(S) is proportional to det(L_S). With L = diag(q) K diag(q) and K a cosine Gram matrix,
det(L_S) = prod(q_i^2) * Vol^2(the vectors in S): large when items are high quality AND point
in different directions. Greedy MAP picks, one at a time, the item that grows log det the most,
using the incremental Cholesky update of Chen, Zhang, Zhou (2018) in O(N k^2).
"""

from collections.abc import Iterable

import numpy as np


def dpp_kernel(emb: np.ndarray, quality: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """L = diag(q) K diag(q) + eps I. K = E E^T is positive semidefinite; eps keeps L invertible."""
    e = np.asarray(emb, dtype=float)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    q = np.asarray(quality, dtype=float)
    return q[:, None] * (e @ e.T) * q[None, :] + eps * np.eye(len(q))


def dpp_greedy(L: np.ndarray, k: int, must: Iterable[int] = (), stop_eps: float = 1e-10) -> tuple[list[int], float]:
    """Greedy MAP. Returns (chosen indices, log det(L_S)).

    d2[i] is how much volume item i would still add (its squared distance from the span of the
    chosen items, in L's geometry). Each step takes the largest d2, then updates every d2 with
    one new Cholesky row: e_i = (L[j, i] - <c_j, c_i>) / d_j and d2[i] -= e_i^2.
    """
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    k = min(k, n)
    c = np.zeros((k, n))
    d2 = np.diag(L).copy()
    chosen: list[int] = []
    log_det = 0.0
    queue = [int(j) for j in dict.fromkeys(must)][:k]
    while len(chosen) < k:
        if queue:
            j = queue.pop(0)
        else:
            candidates = d2.copy()
            candidates[chosen] = -np.inf
            j = int(np.argmax(candidates))
        if d2[j] < stop_eps:
            break
        t = len(chosen)
        dj = np.sqrt(d2[j])
        e = (L[j, :] - c[:t, j] @ c[:t, :]) / dj
        c[t, :] = e
        log_det += np.log(d2[j])
        d2 = d2 - e**2
        chosen.append(j)
    return chosen, float(log_det)
