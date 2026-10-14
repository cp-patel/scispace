"""Build the real dependencies (HTTP clients, embedder, brains) from settings. Used by the worker and the CLI tools."""

import functools
import json
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from research_assistant.agent.nodes import Deps, record_usage
from research_assistant.brain.claude import ClaudeBrain
from research_assistant.brain.offline import OfflineBrain
from research_assistant.config import Settings
from research_assistant.embed import FastEmbedder
from research_assistant.sources.arxiv import Arxiv
from research_assistant.sources.http import HttpClient
from research_assistant.sources.openalex import OpenAlex

USER_AGENT = "research-assistant/0.1 (literature review tool)"


def openalex_client(settings: Settings) -> OpenAlex:
    headers = {"User-Agent": USER_AGENT}
    if settings.openalex_api_key:
        headers["Authorization"] = f"Bearer {settings.openalex_api_key}"
    return OpenAlex(HttpClient("openalex", rate_per_second=8, headers=headers, cache_dir=settings.http_cache_dir))


def arxiv_client(settings: Settings) -> Arxiv:
    # arXiv: at most one request every 3 seconds, on one connection.
    http = HttpClient("arxiv", rate_per_second=1 / 3, max_connections=1, timeout_s=60,
                      headers={"User-Agent": USER_AGENT}, cache_dir=settings.http_cache_dir)
    return Arxiv(http)


def load_calibration(path: str | None) -> dict | None:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def build_deps(settings: Settings, pool: AsyncConnectionPool) -> Deps:
    algo = settings.algo
    brains = {"offline": OfflineBrain(algo.max_cards, algo.support_threshold, algo.partial_threshold)}
    if settings.anthropic_api_key:
        brains["claude"] = ClaudeBrain(settings, on_usage=functools.partial(record_usage, pool))
    return Deps(
        pool=pool,
        brains=brains,
        embedder=FastEmbedder(settings.embed_model, settings.embed_cache_dir),
        openalex=openalex_client(settings),
        arxiv=arxiv_client(settings),
        settings=settings,
        calibration=load_calibration(settings.calibration_path),
    )
