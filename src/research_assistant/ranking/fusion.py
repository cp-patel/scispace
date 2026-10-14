"""Turn scores into ranks, and merge several ranked lists with Reciprocal Rank Fusion (RRF)."""

import numpy as np


def ranks_from_scores(scores: np.ndarray, present: np.ndarray | None = None) -> np.ndarray:
    """Rank 1 is the highest score. Ties keep the original order.

    Items marked absent (present[i] is False) get rank inf, and they do not use up rank slots,
    so the present items are ranked 1..p among themselves.
    """
    s = np.asarray(scores, dtype=float)
    if present is not None:
        present = np.asarray(present, dtype=bool)
        s = np.where(present, s, -np.inf)
    order = np.argsort(-s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    if present is not None:
        ranks[~present] = np.inf
    return ranks


def rrf(rank_lists: list[np.ndarray], weights: list[float] | None = None, k: float = 60.0) -> np.ndarray:
    """RRF(d) = sum_i w_i / (k + rank_i(d)).

    Only ranks matter, never raw scores, so lists with incompatible scales (BM25 points,
    cosines, PageRank mass) can be merged without tuning. An absent item (rank inf) adds 0.
    """
    if not rank_lists:
        raise ValueError("rrf needs at least one ranked list")
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights and rank_lists must have the same length")
    total = np.zeros(len(rank_lists[0]), dtype=float)
    for w, ranks in zip(weights, rank_lists):
        total += w / (k + np.asarray(ranks, dtype=float))
    return total
