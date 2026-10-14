"""Evaluation metrics and statistics: recall, precision, nDCG, MRR, bootstrap CIs, a permutation test, Cohen's kappa."""

import numpy as np

from research_assistant.ranking.dense import normalize


def recall_at_k(ranked: list[str], relevant: set[str], k: int, capped: bool = True) -> float:
    """Share of relevant items found in the top k.

    capped=True divides by min(k, |relevant|), so a perfect top-k scores 1 even when the
    answer key is bigger than k (as with a survey that cites 300 papers and k = 40).
    """
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / (min(k, len(relevant)) if capped else len(relevant))


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG: DCG = sum over hits of 1 / log2(position + 1), divided by the best possible DCG."""
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(dcg / ideal) if ideal > 0 else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant item (0 if none)."""
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_resamples, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip test on paired differences. Returns a p-value.

    If A and B were equally good, each difference would be as likely positive as negative,
    so randomly flipping signs shows how often a mean difference this large happens by luck.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + (null >= observed - 1e-12).sum()) / (n_resamples + 1))


def cohen_kappa(pred: list[bool], actual: list[bool]) -> float:
    """Agreement beyond chance: (p_observed - p_expected) / (1 - p_expected)."""
    p = np.asarray(pred, dtype=bool)
    a = np.asarray(actual, dtype=bool)
    observed = float((p == a).mean())
    expected = p.mean() * a.mean() + (1 - p.mean()) * (1 - a.mean())
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def mean_pairwise_cosine(emb: np.ndarray) -> float:
    """Average cosine between different items. Lower means a more varied set."""
    e = normalize(emb).astype(float)
    n = len(e)
    if n < 2:
        return 0.0
    s = e @ e.T
    return float((s.sum() - np.trace(s)) / (n * (n - 1)))
