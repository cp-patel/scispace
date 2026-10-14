"""Vector search benchmark: recall@10, queries per second, build time, and bytes per vector.

Run: python -m lab.bench --source openalex --n 50000 --queries 1000 [--pgvector]
Quick check without network: python -m lab.bench --source synthetic --n 5000 --queries 200
Writes lab/results/results.csv, lab/results/results.md, and (if matplotlib is installed) lab/results/pareto.png.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from lab.common import exact_search, recall
from lab.data import load
from lab.hnsw import HNSW
from lab.ivf import IVF
from lab.lsh import SimHashLSH
from lab.pq import PQ

FIELDS = ["method", "params", "recall_at_10", "qps", "build_s", "bytes_per_vector"]


def measure(name: str, params: str, search, Q: np.ndarray, truth: list[np.ndarray], build_s: float, bytes_per_vector: int) -> dict:
    start = time.perf_counter()
    found = [search(q) for q in Q]
    elapsed = time.perf_counter() - start
    row = {"method": name, "params": params, "recall_at_10": round(float(np.mean([recall(f, t) for f, t in zip(found, truth)])), 4),
           "qps": round(len(Q) / elapsed, 1), "build_s": round(build_s, 2), "bytes_per_vector": bytes_per_vector}
    print(row)
    return row


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def run_numpy_methods(X: np.ndarray, Q: np.ndarray, truth: list[np.ndarray]) -> list[dict]:
    rows = [measure("exact", "-", lambda q: exact_search(X, q, 10), Q, truth, 0.0, X.shape[1] * 4)]
    for bits, tables in [(8, 8), (8, 16), (10, 32), (12, 32)]:
        lsh = SimHashLSH(X.shape[1], bits, tables, seed=0)
        build = timed(lambda lsh=lsh: lsh.add(X))
        size = lsh.bytes_per_vector() + X.shape[1] * 4
        rows.append(measure("lsh", f"bits={bits} tables={tables}", lambda q, lsh=lsh: lsh.search(q, 10), Q, truth, build, size))
    ivf = IVF(nlist=256 if len(X) >= 20_000 else 64, seed=0)
    build = timed(lambda: (ivf.fit(X), ivf.add(X)))
    for nprobe in (1, 2, 4, 8, 16, 32):
        rows.append(measure("ivf", f"nlist={ivf.nlist} nprobe={nprobe}", lambda q, nprobe=nprobe: ivf.search(q, 10, nprobe),
                            Q, truth, build, ivf.bytes_per_vector()))
    for m in (8, 16, 32, 48):
        pq = PQ(m=m, seed=0)
        build = timed(lambda pq=pq: (pq.fit(X), pq.add(X)))
        rows.append(measure("pq", f"m={m}", lambda q, pq=pq: pq.search(q, 10), Q, truth, build, pq.bytes_per_vector()))
        rows.append(measure("pq+rerank", f"m={m} rerank=100", lambda q, pq=pq: pq.search(q, 10, rerank=100), Q, truth, build, pq.bytes_per_vector()))
    hnsw = HNSW(X.shape[1], M=16, ef_construction=100, seed=0)
    build = timed(lambda: hnsw.add(X))
    for ef in (16, 32, 64, 128, 256):
        rows.append(measure("hnsw (ours)", f"M=16 ef={ef}", lambda q, ef=ef: hnsw.search(q, 10, ef), Q, truth, build, hnsw.bytes_per_vector()))
    return rows


def run_pgvector(X: np.ndarray, Q: np.ndarray, truth: list[np.ndarray], url: str) -> list[dict]:
    import psycopg
    from pgvector.psycopg import register_vector

    rows = []
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS lab_vectors")
        conn.execute(f"CREATE TABLE lab_vectors (id int PRIMARY KEY, v vector({X.shape[1]}))")
        with conn.cursor().copy("COPY lab_vectors (id, v) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, v in enumerate(X):
                copy.write_row((i, v))
        build = timed(lambda: conn.execute("CREATE INDEX ON lab_vectors USING hnsw (v vector_ip_ops) WITH (m = 16, ef_construction = 100)"))
        for ef in (16, 32, 64, 128, 256):
            conn.execute(f"SET hnsw.ef_search = {ef}")

            def search(q, conn=conn):
                return np.array([r[0] for r in conn.execute("SELECT id FROM lab_vectors ORDER BY v <#> %s LIMIT 10", (q,)).fetchall()])

            rows.append(measure("hnsw (pgvector)", f"m=16 ef_search={ef}", search, Q, truth, build, X.shape[1] * 4 + 8 * 16))
        conn.execute("DROP TABLE lab_vectors")
    return rows


def write_results(rows: list[dict], out_dir: str = "lab/results") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Method | Params | Recall@10 | Queries/s | Build (s) | Bytes/vector |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['params']} | {r['recall_at_10']} | {r['qps']} | {r['build_s']} | {r['bytes_per_vector']} |" for r in rows]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in dict.fromkeys(r["method"] for r in rows):
        pts = [(r["recall_at_10"], r["qps"]) for r in rows if r["method"] == method]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (log scale)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["openalex", "synthetic"], default="synthetic")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--pgvector", action="store_true", help="also benchmark pgvector's HNSW (needs DATABASE_URL)")
    parser.add_argument("--out", default="lab/results")
    args = parser.parse_args()
    X, Q = load(args.source, args.n, args.queries)
    truth = [exact_search(X, q, 10) for q in Q]
    rows = run_numpy_methods(X, Q, truth)
    if args.pgvector:
        rows += run_pgvector(X, Q, truth, os.environ.get("DATABASE_URL", "postgresql://ra:ra@localhost:55432/ra"))
    write_results(rows, args.out)


if __name__ == "__main__":
    main()
