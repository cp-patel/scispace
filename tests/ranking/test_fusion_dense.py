import numpy as np
import pytest

from research_assistant.ranking.dense import cosine_matrix, normalize, rocchio
from research_assistant.ranking.fusion import ranks_from_scores, rrf


def test_ranks_highest_score_is_rank_one_and_ties_keep_order():
    assert ranks_from_scores(np.array([0.2, 0.9, 0.2])).tolist() == [2.0, 1.0, 3.0]


def test_absent_items_get_inf_and_do_not_use_rank_slots():
    ranks = ranks_from_scores(np.array([5.0, 1.0, 3.0]), present=np.array([False, True, True]))
    assert ranks[0] == np.inf
    assert ranks[1:].tolist() == [2.0, 1.0]


def test_rrf_known_values():
    a = np.array([1.0, 2.0, np.inf])
    b = np.array([2.0, 1.0, 1.0])
    fused = rrf([a, b], k=60)
    assert fused[0] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 61)


@pytest.mark.parametrize("seed", range(20))
def test_rrf_only_depends_on_ranks_not_score_scale(seed):
    rng = np.random.default_rng(seed)
    s1, s2 = rng.normal(size=30), rng.normal(size=30)
    base = rrf([ranks_from_scores(s1), ranks_from_scores(s2)])
    rescaled = rrf([ranks_from_scores(10 * s1 + 3), ranks_from_scores(np.exp(s2))])
    assert np.allclose(base, rescaled)


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        rrf([np.array([1.0])], weights=[1.0, 2.0])


def test_normalize_and_cosine():
    x = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(x[0]), 1.0)
    assert np.allclose(x[1], 0.0)
    assert cosine_matrix(np.array([[1.0, 0.0]]), np.array([[0.0, 2.0], [5.0, 0.0]])).tolist() == [[0.0, 1.0]]


@pytest.mark.parametrize("seed", range(20))
def test_rocchio_moves_query_toward_relevant_centroid(seed):
    rng = np.random.default_rng(seed)
    q = normalize(rng.normal(size=16))
    relevant = rng.normal(size=(5, 16)) + 2.0
    centroid = normalize(normalize(relevant).mean(axis=0))
    q2 = rocchio(q, relevant, alpha=1.0, beta=0.75)
    assert np.isclose(np.linalg.norm(q2), 1.0, atol=1e-5)
    assert q2 @ centroid > q @ centroid


def test_rocchio_with_beta_zero_returns_query():
    q = normalize(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(rocchio(q, np.ones((2, 3)), beta=0.0), q, atol=1e-6)
