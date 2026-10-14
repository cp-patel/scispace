import itertools
import math

import numpy as np
import pytest

from research_assistant.ranking.dense import normalize
from research_assistant.ranking.dpp import dpp_greedy, dpp_kernel
from research_assistant.ranking.select import (
    FacilityLocation,
    SelectionObjective,
    budgeted_greedy,
    lazy_greedy,
    mmr,
    plain_greedy,
    topk,
)


def instance(seed: int, n: int = 30, facets: int = 4, dim: int = 8):
    rng = np.random.default_rng(seed)
    emb = normalize(rng.normal(size=(n, dim))).astype(float)
    sim = np.maximum(emb @ emb.T, 0.0)
    p = rng.uniform(0.05, 0.95, size=n)
    r = rng.uniform(0.0, 1.0, size=(n, facets))
    return SelectionObjective(p, r, sim, mu=1.0), emb, p


@pytest.mark.parametrize("seed", range(25))
def test_lazy_greedy_picks_exactly_what_plain_greedy_picks(seed):
    obj, _, _ = instance(seed)
    assert lazy_greedy(obj, 10) == plain_greedy(obj, 10)


@pytest.mark.parametrize("seed", range(10))
def test_greedy_reaches_one_minus_one_over_e_of_brute_force(seed):
    obj, _, _ = instance(seed, n=11, facets=3)
    k = 4
    best = max(obj.value(s) for s in itertools.combinations(range(11), k))
    assert obj.value(lazy_greedy(obj, k)) >= (1 - 1 / math.e) * best - 1e-12


@pytest.mark.parametrize("seed", range(25))
def test_objective_has_diminishing_returns(seed):
    obj, _, _ = instance(seed, n=20)
    rng = np.random.default_rng(seed)
    order = rng.permutation(20)
    small, big, x = list(order[:3]), list(order[:8]), int(order[10])
    gain_small = obj.value(small + [x]) - obj.value(small)
    gain_big = obj.value(big + [x]) - obj.value(big)
    assert gain_small >= gain_big - 1e-12


def test_incremental_gain_matches_value_difference():
    obj, _, _ = instance(3)
    state = obj.initial()
    chosen = []
    for j in [4, 9, 1]:
        expected = obj.value(chosen + [j]) - obj.value(chosen)
        assert obj.gain(state, j) == pytest.approx(expected)
        assert obj.gains(state)[j] == pytest.approx(expected)
        state = obj.add(state, j)
        chosen.append(j)


def test_must_include_items_come_first():
    obj, _, _ = instance(5)
    picks = lazy_greedy(obj, 5, must=[7, 2])
    assert picks[:2] == [7, 2] and len(picks) == 5 and len(set(picks)) == 5


def test_k_larger_than_n_returns_everything_once():
    obj, _, _ = instance(1, n=6)
    assert sorted(lazy_greedy(obj, 50)) == list(range(6))


def test_facility_location_rejects_negative_similarity():
    with pytest.raises(ValueError):
        FacilityLocation(np.ones(2), np.array([[1.0, -0.5], [-0.5, 1.0]]))


@pytest.mark.parametrize("seed", range(10))
def test_budgeted_greedy_respects_budget_and_guarantee(seed):
    rng = np.random.default_rng(seed)
    n = 10
    emb = normalize(rng.normal(size=(n, 6))).astype(float)
    obj = FacilityLocation(rng.uniform(0.1, 1.0, n), np.maximum(emb @ emb.T, 0.0))
    costs = rng.integers(1, 6, size=n).astype(float)
    budget = 8.0
    picks = budgeted_greedy(obj, costs, budget)
    assert costs[picks].sum() <= budget
    feasible = (s for r in range(n + 1) for s in itertools.combinations(range(n), r) if costs[list(s)].sum() <= budget)
    best = max(obj.value(s) for s in feasible)
    assert obj.value(picks) >= 0.5 * (1 - 1 / math.e) * best - 1e-12


def test_topk_and_mmr():
    scores = np.array([0.1, 0.9, 0.5, 0.8])
    assert topk(scores, 2) == [1, 3]
    assert topk(scores, 2, must=[0]) == [0, 1]
    # items 1 and 3 are near-duplicates, so MMR swaps 3 for 2
    sim = np.eye(4)
    sim[1, 3] = sim[3, 1] = 0.99
    assert mmr(scores, sim, 2, lam=0.5) == [1, 2]


@pytest.mark.parametrize("seed", range(15))
def test_dpp_log_det_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(20, 6))
    L = dpp_kernel(emb, np.exp(rng.uniform(0, 1, 20)))
    chosen, log_det = dpp_greedy(L, 5)
    sign, expected = np.linalg.slogdet(L[np.ix_(chosen, chosen)])
    assert sign > 0
    assert log_det == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_dpp_greedy_matches_brute_force_greedy(seed):
    rng = np.random.default_rng(seed)
    L = dpp_kernel(rng.normal(size=(12, 5)), np.exp(rng.uniform(0, 1, 12)))
    chosen = []
    for _ in range(4):
        rest = [i for i in range(12) if i not in chosen]
        chosen.append(max(rest, key=lambda i: np.linalg.slogdet(L[np.ix_(chosen + [i], chosen + [i])])[1]))
    assert dpp_greedy(L, 4)[0] == chosen


def test_dpp_first_pick_is_highest_quality_and_duplicates_are_avoided():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    quality = np.array([1.0, 0.99, 0.5])
    chosen, _ = dpp_greedy(dpp_kernel(emb, quality), 2)
    assert chosen == [0, 2]  # item 1 duplicates item 0, so the orthogonal item wins


def test_dpp_must_include():
    rng = np.random.default_rng(0)
    L = dpp_kernel(rng.normal(size=(10, 4)), np.ones(10))
    chosen, _ = dpp_greedy(L, 3, must=[9])
    assert chosen[0] == 9 and len(set(chosen)) == 3
