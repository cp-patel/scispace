"""BM25F: keyword relevance over several fields (title, abstract).

Each field's term counts are length-normalized and weighted, then combined into one
pseudo term frequency before saturation. Saturation means the 50th repeat of a word adds
almost nothing, and length normalization means long abstracts don't win just by being long.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    weight: float  # how much one occurrence in this field counts
    b: float  # 0 = ignore field length, 1 = fully normalize by length


def idf(n_docs: int, df: np.ndarray) -> np.ndarray:
    """IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1). Rare terms count more; never negative."""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def bm25f_scores(
    docs: list[dict[str, list[str]]],
    query: list[str],
    fields: dict[str, FieldSpec],
    k1: float = 1.2,
) -> np.ndarray:
    """Score every doc against the query.

    docs: one dict per doc mapping field name to its tokens, e.g. {"title": [...], "abstract": [...]}.
    Returns an array of n scores. Repeated query terms count once.
    """
    terms = list(dict.fromkeys(query))
    n = len(docs)
    if n == 0 or not terms:
        return np.zeros(n)
    column = {t: j for j, t in enumerate(terms)}
    pseudo_tf = np.zeros((n, len(terms)))
    contains = np.zeros((n, len(terms)), dtype=bool)
    for name, spec in fields.items():
        lengths = np.array([len(d.get(name, ())) for d in docs], dtype=float)
        avg = lengths.mean() if lengths.mean() > 0 else 1.0
        norm = np.maximum(1.0 - spec.b + spec.b * lengths / avg, 1e-9)
        tf = np.zeros((n, len(terms)))
        for i, d in enumerate(docs):
            for token in d.get(name, ()):
                j = column.get(token)
                if j is not None:
                    tf[i, j] += 1.0
        contains |= tf > 0
        pseudo_tf += spec.weight * tf / norm[:, None]
    saturated = pseudo_tf * (k1 + 1.0) / (pseudo_tf + k1)
    return saturated @ idf(n, contains.sum(axis=0))


def bm25_scores(docs: list[list[str]], query: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Plain single-field BM25 (used for chunks and sentences)."""
    return bm25f_scores([{"text": d} for d in docs], query, {"text": FieldSpec(1.0, b)}, k1)
