"""Spectral clustering of papers into sections, plus k-means (also used by the vector lab).

Build a similarity graph, take the eigenvectors of its normalized Laplacian, and run k-means on
them. The eigengap (the biggest jump between consecutive small eigenvalues) says how many
natural groups the graph has.
"""

import numpy as np

from research_assistant.ranking.dense import normalize


def sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row of X (n, d) to every row of C (k, d)."""
    return np.maximum((X**2).sum(axis=1)[:, None] - 2.0 * X @ C.T + (C**2).sum(axis=1)[None, :], 0.0)


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: each new center is drawn with probability proportional to squared distance from the nearest center."""
    n = len(X)
    first = int(rng.integers(n))
    centers = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d2 / total))
        centers.append(X[idx])
        d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(axis=1))
    return np.array(centers)


def kmeans(X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with k-means++ starts. Returns (labels, centers, inertia) of the best run."""
    X = np.asarray(X)
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of points")
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(n_init):
        centers = kmeans_pp_init(X, k, rng)
        for _ in range(max_iter):
            labels = sq_dists(X, centers).argmin(axis=1)
            sums = np.zeros_like(centers, dtype=float)
            np.add.at(sums, labels, X)
            counts = np.bincount(labels, minlength=k)
            new = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centers)
            if np.allclose(new, centers):
                centers = new
                break
            centers = new
        d = sq_dists(X, centers)
        labels = d.argmin(axis=1)
        inertia = float(d[np.arange(len(X)), labels].sum())
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    return best


def bibliographic_coupling(ref_sets: list[set[str]]) -> np.ndarray:
    """Cosine between reference sets: |A and B| / sqrt(|A| |B|). Papers citing the same work are related."""
    n = len(ref_sets)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = ref_sets[i], ref_sets[j]
            if a and b:
                m[i, j] = m[j, i] = len(a & b) / np.sqrt(len(a) * len(b))
    return m


def knn_sparsify(W: np.ndarray, k: int) -> np.ndarray:
    """Keep each node's k strongest edges, then symmetrize with max(W, W^T). The diagonal is zeroed."""
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    n = len(W)
    if k < n - 1:
        keep = np.zeros_like(W, dtype=bool)
        idx = np.argsort(-W, axis=1)[:, :k]
        keep[np.arange(n)[:, None], idx] = True
        W = np.where(keep, W, 0.0)
    return np.maximum(W, W.T)


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """L_sym = I - D^(-1/2) W D^(-1/2). Isolated nodes get a zero row in the second term."""
    degree = W.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    return np.eye(len(W)) - inv_sqrt[:, None] * W * inv_sqrt[None, :]


def k_range(n: int) -> tuple[int, int]:
    """Allowed section counts: 3 to min(8, n/4) for normal sets; 2 to max(2, n/4) for small ones (n < 12)."""
    if n >= 12:
        return 3, min(8, n // 4)
    return 2, max(2, n // 4)


def choose_k(eigvals: np.ndarray, k_min: int, k_max: int) -> int:
    """Eigengap: the k (1-based) where lambda_(k+1) - lambda_k is largest, with eigenvalues sorted ascending."""
    lam = np.sort(np.asarray(eigvals, dtype=float))
    k_max = min(k_max, len(lam) - 1)
    if k_max < k_min:
        return max(1, min(k_min, len(lam)))
    gaps = {k: lam[k] - lam[k - 1] for k in range(k_min, k_max + 1)}
    return max(gaps, key=gaps.get)


def spectral_clusters(
    emb: np.ndarray, ref_sets: list[set[str]], beta: float = 0.7, knn: int = 8, seed: int = 0
) -> tuple[np.ndarray, int, np.ndarray]:
    """Cluster papers using W = beta * max(0, cosine) + (1 - beta) * bibliographic coupling.

    Returns (labels, k, eigenvalues).
    """
    n = len(emb)
    if n < 2:
        return np.zeros(n, dtype=int), 1, np.zeros(n)
    e = normalize(emb).astype(float)
    W = beta * np.maximum(e @ e.T, 0.0) + (1.0 - beta) * bibliographic_coupling(ref_sets)
    W = knn_sparsify(W, knn)
    vals, vecs = np.linalg.eigh(normalized_laplacian(W))
    k = choose_k(vals, *k_range(n))
    U = vecs[:, :k]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    labels, _, _ = kmeans(U, k, n_init=10, seed=seed)
    return labels, k, vals


def merge_small(labels: np.ndarray, emb: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Move members of clusters smaller than min_size into the nearest other cluster (by centroid cosine). Relabel 0..k-1."""
    labels = np.array(labels)
    e = normalize(emb)
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < min_size]
        if len(small) == 0 or len(ids) == 1:
            break
        target = small[0]
        others = [c for c in ids if c != target]
        centroids = normalize(np.array([e[labels == c].mean(axis=0) for c in others]))
        for i in np.where(labels == target)[0]:
            labels[i] = others[int(np.argmax(centroids @ e[i]))]
    _, labels = np.unique(labels, return_inverse=True)
    return labels
