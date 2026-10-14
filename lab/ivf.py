"""IVF (inverted file): split the space into nlist cells with k-means; search only the nprobe nearest cells."""

import numpy as np

from lab.common import top_k_indices
from research_assistant.ranking.cluster import kmeans, sq_dists


class IVF:
    def __init__(self, nlist: int = 256, seed: int = 0, max_iter: int = 25):
        self.nlist, self.seed, self.max_iter = nlist, seed, max_iter
        self.centroids: np.ndarray | None = None
        self.lists: list[np.ndarray] = []
        self.X = np.zeros((0, 0), dtype=np.float32)

    def fit(self, X: np.ndarray) -> None:
        _, self.centroids, _ = kmeans(np.asarray(X, dtype=np.float32), self.nlist, n_init=1, max_iter=self.max_iter, seed=self.seed)

    def add(self, X: np.ndarray) -> None:
        if self.centroids is None:
            raise RuntimeError("call fit() before add()")
        self.X = np.asarray(X, dtype=np.float32)
        cell = sq_dists(self.X, self.centroids).argmin(axis=1)
        order = np.argsort(cell, kind="stable")
        bounds = np.searchsorted(cell[order], np.arange(1, self.nlist))
        self.lists = np.split(order, bounds)

    def search(self, q: np.ndarray, k: int, nprobe: int = 8) -> np.ndarray:
        nearest_cells = np.argsort(((self.centroids - q) ** 2).sum(axis=1))[:nprobe]
        ids = np.concatenate([self.lists[c] for c in nearest_cells])
        if len(ids) == 0:
            return ids
        return ids[top_k_indices(self.X[ids] @ q, k)]

    def bytes_per_vector(self) -> int:
        """The raw float32 vector plus an 8-byte id in its cell list."""
        return self.X.shape[1] * 4 + 8
