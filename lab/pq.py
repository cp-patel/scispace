"""Product Quantization (PQ): compress vectors to m bytes each.

Split every vector into m sub-vectors and replace each one with the id (0..255) of its nearest
centroid in that subspace, learned by k-means. A 384-number float32 vector (1,536 bytes) becomes
m bytes. At query time, precompute the inner product of each query piece with all 256 centroids
of its subspace (a lookup table); a vector's approximate score is then just m table lookups
summed (asymmetric distance computation: the query stays exact, the database is compressed).
"""

import numpy as np

from lab.common import top_k_indices
from research_assistant.ranking.cluster import kmeans, sq_dists


class PQ:
    def __init__(self, m: int = 48, nbits: int = 8, seed: int = 0, max_iter: int = 25):
        if nbits > 8:
            raise ValueError("this lab stores codes as uint8, so nbits must be at most 8")
        self.m, self.ksub, self.seed, self.max_iter = m, 1 << nbits, seed, max_iter
        self.codebooks: np.ndarray | None = None
        self.codes = np.zeros((0, m), dtype=np.uint8)
        self.X: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] % self.m:
            raise ValueError("the dimension must be divisible by m")
        self.dsub = X.shape[1] // self.m
        self.codebooks = np.stack([
            kmeans(self._piece(X, s), self.ksub, n_init=1, max_iter=self.max_iter, seed=self.seed + s)[1]
            for s in range(self.m)
        ]).astype(np.float32)

    def _piece(self, X: np.ndarray, s: int) -> np.ndarray:
        return X[:, s * self.dsub:(s + 1) * self.dsub]

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        codes = np.empty((len(X), self.m), dtype=np.uint8)
        for s in range(self.m):
            codes[:, s] = sq_dists(self._piece(X, s), self.codebooks[s]).argmin(axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.concatenate([self.codebooks[s][codes[:, s]] for s in range(self.m)], axis=1)

    def add(self, X: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.codes = self.encode(self.X)

    def search(self, q: np.ndarray, k: int, rerank: int = 0) -> np.ndarray:
        """Approximate top-k. With rerank > 0, re-score the best `rerank` candidates with the exact vectors."""
        table = np.einsum("skd,sd->sk", self.codebooks, q.reshape(self.m, self.dsub))
        scores = table[np.arange(self.m)[None, :], self.codes].sum(axis=1)
        if rerank:
            candidates = top_k_indices(scores, rerank)
            return candidates[top_k_indices(self.X[candidates] @ q, k)]
        return top_k_indices(scores, k)

    def bytes_per_vector(self) -> int:
        return self.m
