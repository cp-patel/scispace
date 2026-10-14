"""Choose a set of K items: top-k, MMR, and monotone submodular maximization with (lazy) greedy.

Submodular means diminishing returns: an item adds less when similar items are already chosen.
For a monotone submodular objective, greedy is proven to reach at least (1 - 1/e), about 63%,
of the best possible value (Nemhauser, Wolsey, Fisher 1978).
"""

import heapq
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np


class Objective(Protocol):
    n: int

    def initial(self) -> Any: ...
    def gains(self, state: Any) -> np.ndarray: ...
    def gain(self, state: Any, j: int) -> float: ...
    def add(self, state: Any, j: int) -> Any: ...
    def value(self, chosen: Iterable[int]) -> float: ...


def topk(scores: np.ndarray, k: int, must: Iterable[int] = ()) -> list[int]:
    """Baseline: must-include items first, then the highest scores."""
    chosen = list(dict.fromkeys(must))[:k]
    for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable"):
        if len(chosen) >= k:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen


def mmr(rel: np.ndarray, sim: np.ndarray, k: int, lam: float = 0.7, must: Iterable[int] = ()) -> list[int]:
    """Maximal Marginal Relevance: repeatedly add argmax [lam * rel - (1 - lam) * max similarity to the chosen set]."""
    rel = np.asarray(rel, dtype=float)
    n = len(rel)
    chosen = list(dict.fromkeys(must))[:k]
    while len(chosen) < min(k, n):
        redundancy = sim[:, chosen].max(axis=1) if chosen else np.zeros(n)
        score = lam * rel - (1.0 - lam) * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


class FacilityLocation:
    """F(S) = sum_i w_i * max_{j in S} sim[i, j]. Every item wants a close representative in S.

    Similarities must be non-negative (clip cosines at 0) so F stays monotone.
    """

    def __init__(self, weights: np.ndarray, sim: np.ndarray):
        self.w = np.asarray(weights, dtype=float)
        self.sim = np.asarray(sim, dtype=float)
        if (self.sim < 0).any():
            raise ValueError("similarities must be non-negative; clip them at 0")
        self.n = len(self.w)

    def initial(self) -> np.ndarray:
        return np.zeros(self.n)  # best similarity to the chosen set, for every item

    def gains(self, best: np.ndarray) -> np.ndarray:
        return self.w @ np.maximum(self.sim - best[:, None], 0.0)

    def gain(self, best: np.ndarray, j: int) -> float:
        return float(self.w @ np.maximum(self.sim[:, j] - best, 0.0))

    def add(self, best: np.ndarray, j: int) -> np.ndarray:
        return np.maximum(best, self.sim[:, j])

    def value(self, chosen: Iterable[int]) -> float:
        chosen = list(chosen)
        return float(self.w @ self.sim[:, chosen].max(axis=1)) if chosen else 0.0


class SelectionObjective:
    """F(S) = sum_f w_f * log(1 + sum_{d in S} p_d * r_{d,f})  +  (mu / n) * sum_i p_i * max_{j in S} sim[i, j].

    Part 1 gives diminishing returns per facet, which balances facets.
    Part 2 (facility location) spreads the picks over the whole candidate list.
    p: probability each candidate is relevant (n,). r: facet relevance in [0, 1] (n, F).
    """

    def __init__(self, p: np.ndarray, r: np.ndarray, sim: np.ndarray, facet_weights: np.ndarray | None = None, mu: float = 1.0):
        self.p = np.asarray(p, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.n, n_facets = self.r.shape
        self.fw = np.full(n_facets, 1.0 / n_facets) if facet_weights is None else np.asarray(facet_weights, dtype=float)
        self.contrib = self.p[:, None] * self.r
        self.fl = FacilityLocation(self.p * (mu / self.n), sim)

    def initial(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.r.shape[1]), self.fl.initial()

    def gains(self, state: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        mass, best = state
        facet = (np.log1p(mass[None, :] + self.contrib) - np.log1p(mass)[None, :]) @ self.fw
        return facet + self.fl.gains(best)

    def gain(self, state: tuple[np.ndarray, np.ndarray], j: int) -> float:
        mass, best = state
        facet = float((np.log1p(mass + self.contrib[j]) - np.log1p(mass)) @ self.fw)
        return facet + self.fl.gain(best, j)

    def add(self, state: tuple[np.ndarray, np.ndarray], j: int) -> tuple[np.ndarray, np.ndarray]:
        mass, best = state
        return mass + self.contrib[j], self.fl.add(best, j)

    def value(self, chosen: Iterable[int]) -> float:
        chosen = list(chosen)
        mass = self.contrib[chosen].sum(axis=0) if chosen else np.zeros(self.r.shape[1])
        return float(np.log1p(mass) @ self.fw) + self.fl.value(chosen)


def _start(obj: Objective, must: Iterable[int]) -> tuple[list[int], Any]:
    chosen, state = [], obj.initial()
    for j in dict.fromkeys(must):
        chosen.append(int(j))
        state = obj.add(state, int(j))
    return chosen, state


def plain_greedy(obj: Objective, k: int, must: Iterable[int] = ()) -> list[int]:
    """Recompute every gain each round and take the best. Simple reference for tests."""
    chosen, state = _start(obj, must)
    while len(chosen) < min(k, obj.n):
        g = obj.gains(state)
        g[chosen] = -np.inf
        j = int(np.argmax(g))
        chosen.append(j)
        state = obj.add(state, j)
    return chosen


def lazy_greedy(obj: Objective, k: int, must: Iterable[int] = ()) -> list[int]:
    """Same picks as plain_greedy, with far fewer gain evaluations (Minoux's accelerated greedy).

    Keep each item's last computed gain in a max-heap. Gains can only shrink as the set grows,
    so a stale gain is an upper bound: if the top item's fresh gain still beats every other
    upper bound, it is the true best and we take it without checking the rest.
    """
    chosen, state = _start(obj, must)
    taken = set(chosen)
    g0 = obj.gains(state)
    heap = [(-g0[j], j) for j in range(obj.n) if j not in taken]
    heapq.heapify(heap)
    while len(chosen) < min(k, obj.n) and heap:
        _, j = heapq.heappop(heap)
        fresh = obj.gain(state, j)
        if not heap or (-fresh, j) <= heap[0]:
            chosen.append(j)
            state = obj.add(state, j)
        else:
            heapq.heappush(heap, (-fresh, j))
    return chosen


def budgeted_greedy(obj: Objective, costs: np.ndarray, budget: float) -> list[int]:
    """Maximize F(S) subject to sum(costs[S]) <= budget.

    Cost-benefit lazy greedy (gain per unit cost), then compare with the best single item that
    fits and keep whichever is better. That pair is proven to reach 1/2 (1 - 1/e) of the optimum.
    """
    costs = np.asarray(costs, dtype=float)
    if (costs <= 0).any():
        raise ValueError("costs must be positive")
    chosen, state, spent = [], obj.initial(), 0.0
    g0 = obj.gains(state)
    heap = [(-(g0[j] / costs[j]), j) for j in range(obj.n) if costs[j] <= budget]
    heapq.heapify(heap)
    while heap:
        _, j = heapq.heappop(heap)
        if spent + costs[j] > budget:
            continue  # it will never fit again, because spending only grows
        fresh = obj.gain(state, j) / costs[j]
        if not heap or (-fresh, j) <= heap[0]:
            if fresh <= 0:
                break
            chosen.append(j)
            state = obj.add(state, j)
            spent += costs[j]
        else:
            heapq.heappush(heap, (-fresh, j))
    fits = [j for j in range(obj.n) if costs[j] <= budget]
    if fits:
        best_single = max(fits, key=lambda j: obj.value([j]))
        if obj.value([best_single]) > obj.value(chosen):
            return [best_single]
    return chosen
