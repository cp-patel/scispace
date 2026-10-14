"""Random-hyperplane LSH (SimHash) for cosine similarity.

Each bit records which side of a random hyperplane a vector falls on. Two vectors at angle theta
land on the same side with probability 1 - theta / pi. A table key is `bits` such bits (AND:
near-duplicates only), and `tables` independent tables are unioned (OR: catch more true neighbors).
Candidates from matching buckets are then re-ranked exactly.
"""

from collections import defaultdict

import numpy as np

from lab.common import top_k_indices


class SimHashLSH:
    def __init__(self, dim: int, bits: int = 12, tables: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim, self.bits, self.n_tables = dim, bits, tables
        self.planes = rng.normal(size=(tables * bits, dim)).astype(np.float32)
        self.powers = (1 << np.arange(bits)).astype(np.int64)
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(tables)]
        self.X = np.zeros((0, dim), dtype=np.float32)

    def signs(self, X: np.ndarray) -> np.ndarray:
        """(n, tables, bits) booleans: which side of each hyperplane."""
        return ((np.asarray(X, dtype=np.float32) @ self.planes.T) > 0).reshape(len(X), self.n_tables, self.bits)

    def keys(self, X: np.ndarray) -> np.ndarray:
        """(n, tables) integer bucket keys."""
        return (self.signs(X).astype(np.int64) * self.powers).sum(axis=2)

    def add(self, X: np.ndarray) -> None:
        start = len(self.X)
        self.X = np.vstack([self.X, np.asarray(X, dtype=np.float32)])
        for offset, row in enumerate(self.keys(X)):
            for t, key in enumerate(row):
                self.tables[t][int(key)].append(start + offset)

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        row = self.keys(q[None, :])[0]
        candidates: set[int] = set()
        for t, key in enumerate(row):
            candidates.update(self.tables[t].get(int(key), ()))
        if not candidates:
            return np.array([], dtype=np.int64)
        ids = np.fromiter(candidates, dtype=np.int64)
        return ids[top_k_indices(self.X[ids] @ q, k)]

    def bytes_per_vector(self) -> int:
        """Index overhead: one 8-byte key per table (raw vectors are also kept for re-ranking)."""
        return 8 * self.n_tables
