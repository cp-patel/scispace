"""Benchmark data: real abstract embeddings from OpenAlex (cached to disk), or a synthetic mixture for quick runs."""

import asyncio
from pathlib import Path

import numpy as np

from lab.common import gaussian_mixture
from research_assistant.config import get_settings
from research_assistant.embed import FastEmbedder
from research_assistant.runtime import openalex_client


async def fetch_abstracts(n: int) -> list[str]:
    settings = get_settings()
    oa = openalex_client(settings)
    try:
        works = await oa.sample(n)
    finally:
        await oa.http.aclose()
    print(f"fetched {len(works)} abstracts, OpenAlex cost ${oa.http.cost_usd:.4f}")
    return [f"{w.title}. {w.abstract}" for w in works]


def load(source: str, n: int, queries: int, cache_dir: str = "lab/data") -> tuple[np.ndarray, np.ndarray]:
    """Returns (database vectors, held-out query vectors), all unit length."""
    if source == "synthetic":
        points = gaussian_mixture(n + queries, 384, centers=50, seed=0)
        return points[:n], points[n:]
    path = Path(cache_dir) / f"abstracts_{n + queries}.npy"
    if not path.exists():
        texts = asyncio.run(fetch_abstracts(n + queries))
        settings = get_settings()
        vectors = FastEmbedder(settings.embed_model, settings.embed_cache_dir).embed_docs(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    vectors = np.load(path)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(vectors))
    return vectors[order[queries:]], vectors[order[:queries]]
