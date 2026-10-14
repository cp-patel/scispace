import numpy as np
import pytest

from research_assistant.ranking.bayes import (
    JudgeRates,
    calibration_table,
    judge_rates,
    linear_prior,
    posterior,
    prior_from_table,
)
from research_assistant.ranking.pagerank import personalized_pagerank


def random_graph(seed: int, n: int = 25, m: int = 70) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(m, 2))


def direct_solve(n: int, edges: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """Solve (I - alpha * P^T - alpha * v d^T) pi = (1 - alpha) v exactly, for comparison."""
    e = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    P = np.zeros((n, n))
    for s, t in e:
        P[s, t] = 1.0
    out = P.sum(axis=1)
    dangling = (out == 0).astype(float)
    P = np.divide(P, out[:, None], out=np.zeros_like(P), where=out[:, None] > 0)
    v = v / v.sum()
    A = np.eye(n) - alpha * P.T - alpha * np.outer(v, dangling)
    return np.linalg.solve(A, (1 - alpha) * v)


@pytest.mark.parametrize("seed", range(15))
def test_pagerank_sums_to_one_and_matches_direct_solve(seed):
    n = 25
    edges = random_graph(seed, n)
    v = np.random.default_rng(seed + 100).random(n)
    pi, iters = personalized_pagerank(n, edges, v, alpha=0.85, tol=1e-12, max_iter=1000)
    assert pi.min() >= 0
    assert pi.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(pi, direct_solve(n, edges, v, 0.85), atol=1e-8)
    assert iters < 1000


def test_pagerank_rewards_papers_the_seeds_cite():
    # 0 and 1 are seeds; both cite 2. Paper 3 is cited by nobody.
    edges = [(0, 2), (1, 2), (3, 0)]
    v = np.array([1.0, 1.0, 0.0, 0.0])
    pi, _ = personalized_pagerank(4, edges, v)
    assert pi[2] > pi[3]


def test_pagerank_with_no_edges_returns_teleport():
    pi, _ = personalized_pagerank(3, [], np.array([2.0, 1.0, 1.0]))
    assert np.allclose(pi, [0.5, 0.25, 0.25])


def test_pagerank_rejects_bad_teleport():
    with pytest.raises(ValueError):
        personalized_pagerank(2, [], np.array([0.0, 0.0]))


def test_uninformative_judge_leaves_prior_unchanged():
    rates = JudgeRates(sensitivity=0.7, specificity=0.3)  # Se = 1 - Sp
    assert posterior(0.4, True, rates) == pytest.approx(0.4)
    assert posterior(0.4, False, rates) == pytest.approx(0.4)


def test_informative_judge_moves_posterior_the_right_way():
    rates = JudgeRates(sensitivity=0.9, specificity=0.8)
    assert posterior(0.3, True, rates) > 0.3
    assert posterior(0.3, False, rates) < 0.3
    # odds 3/7 * LR+ 4.5 = 27/14 -> p = 27/41
    assert posterior(0.3, True, rates) == pytest.approx(27 / 41)


def test_linear_prior_endpoints_and_table_lookup():
    assert linear_prior(1, 150) == pytest.approx(0.6)
    assert linear_prior(150, 150) == pytest.approx(0.1)
    assert linear_prior(999, 150) == pytest.approx(0.1)
    table = [(1, 10, 0.5), (11, None, 0.1)]
    assert prior_from_table(3, table) == 0.5
    assert prior_from_table(500, table) == 0.1
    with pytest.raises(ValueError):
        prior_from_table(0, table)


def test_calibration_table_and_judge_rates_use_add_one_smoothing():
    table = calibration_table([1, 2, 30, 200], [True, True, False, False])
    assert table[0] == (1, 10, (2 + 1) / (2 + 2))
    assert table[2] == (26, 50, (0 + 1) / (1 + 2))
    assert table[1] == (11, 25, 0.5)  # empty bucket
    rates = judge_rates([True, True, False, False], [True, False, False, True])
    assert rates.sensitivity == pytest.approx((1 + 1) / (2 + 2))
    assert rates.specificity == pytest.approx((1 + 1) / (2 + 2))
