import numpy as np
import pytest

from research_assistant.ranking.cluster import (
    bibliographic_coupling,
    choose_k,
    k_range,
    kmeans,
    knn_sparsify,
    merge_small,
    normalized_laplacian,
    spectral_clusters,
)
from research_assistant.ranking.metrics import (
    bootstrap_ci,
    cohen_kappa,
    mean_pairwise_cosine,
    mrr,
    ndcg_at_k,
    paired_permutation_test,
    precision_at_k,
    recall_at_k,
)


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two labelings group items the same way (label names may differ)."""
    return all((a[i] == a[j]) == (b[i] == b[j]) for i in range(len(a)) for j in range(len(a)))


def planted(seed: int, groups: int = 3, size: int = 10, dim: int = 32):
    rng = np.random.default_rng(seed)
    centers = np.linalg.qr(rng.normal(size=(dim, groups)))[0].T  # orthonormal group directions
    emb, truth, refs = [], [], []
    for g in range(groups):
        for _ in range(size):
            emb.append(centers[g] + 0.15 * rng.normal(size=dim))
            truth.append(g)
            refs.append({f"G{g}R{r}" for r in rng.choice(12, size=6, replace=False)})
    return np.array(emb), np.array(truth), refs


@pytest.mark.parametrize("seed", range(5))
def test_spectral_recovers_planted_groups_and_their_count(seed):
    emb, truth, refs = planted(seed)
    labels, k, eigvals = spectral_clusters(emb, refs)
    assert k == 3
    assert same_partition(labels, truth)
    assert np.all(np.diff(np.sort(eigvals)) >= -1e-9)


def test_kmeans_separates_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=c, scale=0.1, size=(20, 2)) for c in ([0, 0], [5, 5], [0, 5])])
    labels, centers, inertia = kmeans(X, 3, n_init=5)
    assert same_partition(labels, np.repeat([0, 1, 2], 20))
    assert inertia < 3.0 and centers.shape == (3, 2)


def test_choose_k_picks_largest_gap_and_k_range():
    assert choose_k(np.array([0.0, 0.01, 0.02, 0.9, 0.95, 1.0]), 2, 5) == 3
    assert k_range(40) == (3, 8)
    assert k_range(20) == (3, 5)
    assert k_range(10) == (2, 2)


def test_coupling_knn_and_laplacian():
    m = bibliographic_coupling([{"a", "b"}, {"b", "c"}, set()])
    assert m[0, 1] == pytest.approx(0.5) and m[0, 2] == 0.0 and m[0, 0] == pytest.approx(1.0)
    W = knn_sparsify(np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]]), k=1)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
    L = normalized_laplacian(W)
    assert np.allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() > -1e-9


def test_merge_small_moves_singletons():
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05]])
    labels = merge_small(np.array([0, 0, 1, 1, 2]), emb, min_size=2)
    assert len(set(labels.tolist())) == 2
    assert labels[4] == labels[0]


def test_ranking_metrics_known_values():
    ranked = ["a", "x", "b", "y"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 2)
    assert recall_at_k(ranked, relevant, 4, capped=False) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b"], {"a", "b", "c", "d"}, 2) == 1.0  # capped by k
    assert precision_at_k(ranked, relevant, 4) == pytest.approx(0.5)
    assert mrr(["x", "a"], relevant) == 0.5
    assert ndcg_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    assert ndcg_at_k(ranked, relevant, 4) == pytest.approx(expected)


def test_kappa_and_diversity():
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)
    assert cohen_kappa([True, False], [True, False]) == pytest.approx(1.0)
    assert mean_pairwise_cosine(np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(0.0)
    assert mean_pairwise_cosine(np.array([[1.0, 0.0], [2.0, 0.0]])) == pytest.approx(1.0)


def test_bootstrap_ci_covers_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(0)
    covered = 0
    for trial in range(200):
        sample = rng.normal(loc=1.0, scale=1.0, size=30)
        low, high = bootstrap_ci(sample, n_resamples=2000, seed=trial)
        covered += low <= 1.0 <= high
    assert 0.88 <= covered / 200 <= 0.99


def test_permutation_test_detects_real_difference_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20)
    assert paired_permutation_test(a + 1.0, a) < 0.01
    assert paired_permutation_test(a + rng.normal(scale=0.01, size=20), a) > 0.05
