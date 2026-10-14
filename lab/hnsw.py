"""HNSW (Hierarchical Navigable Small World) graph search, after Malkov and Yashunin (2018).

Every vector is a node. Each node gets a random top layer: level = floor(-ln(U) * mL) with
mL = 1 / ln(M), so each layer up holds about 1/M as many nodes, like the express lanes of a
skip list. A search starts at the single entry point on the top layer, walks greedily toward the
query on each layer, drops down a layer, and finishes with a beam search of width ef on layer 0.
Expected hops grow like log N. Similarity here is the inner product of unit vectors.
"""

import heapq

import numpy as np


class HNSW:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100, seed: int = 0):
        self.M, self.M0, self.ef_construction = M, 2 * M, ef_construction
        self.mL = 1.0 / np.log(M)
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((0, dim), dtype=np.float32)
        self.graph: list[dict[int, list[int]]] = []  # graph[layer][node] -> neighbor ids
        self.entry: int | None = None
        self.max_level = -1

    def add(self, X: np.ndarray) -> None:
        start = len(self.X)
        self.X = np.vstack([self.X, np.asarray(X, dtype=np.float32)])
        for i in range(start, len(self.X)):
            self._insert(i)

    def search(self, q: np.ndarray, k: int, ef: int = 64) -> np.ndarray:
        if self.entry is None:
            return np.array([], dtype=np.int64)
        entry = [self.entry]
        for layer in range(self.max_level, 0, -1):
            entry = [self._search_layer(q, entry, 1, layer)[0][1]]
        found = self._search_layer(q, entry, max(ef, k), 0)
        return np.array([node for _, node in found[:k]], dtype=np.int64)

    def bytes_per_vector(self) -> int:
        """Raw float32 vector plus about M0 four-byte neighbor ids on layer 0 (upper layers are small)."""
        return self.X.shape[1] * 4 + 4 * self.M0

    def _insert(self, i: int) -> None:
        level = int(-np.log(1.0 - self.rng.random()) * self.mL)
        while len(self.graph) <= level:
            self.graph.append({})
        for layer in range(level + 1):
            self.graph[layer][i] = []
        if self.entry is None:
            self.entry, self.max_level = i, level
            return
        q = self.X[i]
        entry = [self.entry]
        for layer in range(self.max_level, level, -1):  # greedy descent through the express lanes
            entry = [self._search_layer(q, entry, 1, layer)[0][1]]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, entry, self.ef_construction, layer)
            neighbors = self._select(found, self.M)
            self.graph[layer][i] = neighbors
            limit = self.M0 if layer == 0 else self.M
            for nb in neighbors:  # link back, and prune the neighbor's list if it grew too long
                links = self.graph[layer][nb]
                links.append(i)
                if len(links) > limit:
                    sims = self.X[links] @ self.X[nb]
                    self.graph[layer][nb] = self._select(list(zip(sims.tolist(), links)), limit)
            entry = [node for _, node in found]
        if level > self.max_level:
            self.entry, self.max_level = i, level

    def _search_layer(self, q: np.ndarray, entry: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns up to ef (similarity, node) pairs, best first."""
        visited = set(entry)
        sims = (self.X[entry] @ q).tolist()
        candidates = [(-s, e) for s, e in zip(sims, entry)]  # max-heap on similarity
        heapq.heapify(candidates)
        best = [(s, e) for s, e in zip(sims, entry)]  # min-heap: worst kept result on top
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)
        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            if -neg_sim < best[0][0]:
                break  # the closest unexplored node is worse than everything we keep
            fresh = [nb for nb in self.graph[layer].get(node, ()) if nb not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for s, nb in zip((self.X[fresh] @ q).tolist(), fresh):
                if len(best) < ef or s > best[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(best, (s, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(best, reverse=True)

    def _select(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Neighbor-selection heuristic: keep a candidate only if it is closer to the base node than
        to every neighbor kept so far, so links point in different directions. Then top up with the
        best leftovers so nodes keep m links."""
        kept: list[int] = []
        leftovers: list[int] = []
        for s, c in sorted(candidates, reverse=True):
            if len(kept) >= m:
                break
            if all(float(self.X[c] @ self.X[k]) < s for k in kept):
                kept.append(c)
            else:
                leftovers.append(c)
        for c in leftovers:
            if len(kept) >= m:
                break
            kept.append(c)
        return kept
