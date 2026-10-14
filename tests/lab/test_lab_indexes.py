import numpy as np
import pytest

from lab.common import exact_search, gaussian_mixture, recall, top_k_indices
from lab.hnsw import HNSW
from lab.ivf import IVF
from lab.lsh import SimHashLSH
from lab.pq import PQ


@pytest.fixture(scope="module")
def data():
    points = gaussian_mixture(2050, 32, centers=20, seed=0)
    X, Q = points[:2000], points[2000:]  # held-out queries from the same mixture
    truth = [exact_search(X, q, 10) for q in Q]
    return X, Q, truth


def mean_recall(search, Q, truth) -> float:
    return float(np.mean([recall(search(q), t) for q, t in zip(Q, truth)]))


def test_top_k_indices_orders_best_first():
    assert top_k_indices(np.array([0.1, 0.9, 0.5]), 2).tolist() == [1, 2]
    assert top_k_indices(np.array([0.3]), 5).tolist() == [0]


def test_lsh_bit_collision_rate_matches_one_minus_theta_over_pi():
    rng = np.random.default_rng(0)
    lsh = SimHashLSH(dim=64, bits=16, tables=64, seed=1)
    a = rng.normal(size=64)
    b = a + 0.8 * rng.normal(size=64)
    theta = np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    same = (lsh.signs(a[None, :]) == lsh.signs(b[None, :])).mean()
    assert same == pytest.approx(1 - theta / np.pi, abs=0.05)


def test_lsh_recall(data):
    X, Q, truth = data
    lsh = SimHashLSH(dim=32, bits=8, tables=16, seed=0)
    lsh.add(X)
    assert mean_recall(lambda q: lsh.search(q, 10), Q, truth) >= 0.8


def test_ivf_recall_grows_with_nprobe(data):
    X, Q, truth = data
    ivf = IVF(nlist=32, seed=0)
    ivf.fit(X)
    ivf.add(X)
    low = mean_recall(lambda q: ivf.search(q, 10, nprobe=1), Q, truth)
    high = mean_recall(lambda q: ivf.search(q, 10, nprobe=8), Q, truth)
    assert high >= low and high >= 0.9


def test_pq_compresses_and_keeps_good_recall_with_rerank(data):
    X, Q, truth = data
    pq = PQ(m=8, nbits=8, seed=0)
    pq.fit(X)
    pq.add(X)
    assert pq.codes.dtype == np.uint8 and pq.bytes_per_vector() == 8
    assert np.mean((pq.decode(pq.codes) - X) ** 2) < np.mean(X**2)  # decoding is closer than nothing
    assert mean_recall(lambda q: pq.search(q, 10, rerank=100), Q, truth) >= 0.9


def test_pq_rejects_bad_dimension():
    with pytest.raises(ValueError):
        PQ(m=5).fit(np.zeros((300, 32), dtype=np.float32))


def test_hnsw_recall_at_ef_64(data):
    X, Q, truth = data
    index = HNSW(dim=32, M=16, ef_construction=100, seed=0)
    index.add(X)
    assert mean_recall(lambda q: index.search(q, 10, ef=64), Q, truth) >= 0.9


def test_hnsw_levels_thin_out_upward():
    index = HNSW(dim=8, M=16, seed=0)
    index.add(gaussian_mixture(1500, 8, seed=2))
    sizes = [len(layer) for layer in index.graph]
    assert sizes[0] == 1500 and all(a >= b for a, b in zip(sizes, sizes[1:]))


@pytest.mark.slow
def test_hnsw_recall_on_larger_mixture():
    points = gaussian_mixture(5100, 64, seed=3)
    X, Q = points[:5000], points[5000:]
    index = HNSW(dim=64, M=16, ef_construction=100, seed=0)
    index.add(X)
    assert mean_recall(lambda q: index.search(q, 10, ef=64), Q, [exact_search(X, q, 10) for q in Q]) >= 0.9
