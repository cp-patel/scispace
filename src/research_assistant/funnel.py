"""Level 1 retrieval: build the candidate pool, score it, shortlist, screen, and select.

No database access here, so the agent, the tests, and the evals all call the same code.
Embedding and NumPy work is synchronous; async callers should wrap compute_signals in asyncio.to_thread.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from research_assistant.brain.base import Brain, PaperView, Plan, Verdict
from research_assistant.config import AlgoParams
from research_assistant.embed import Embedder
from research_assistant.ranking.bayes import JudgeRates, linear_prior, posterior, prior_from_table
from research_assistant.ranking.bm25 import FieldSpec, bm25f_scores
from research_assistant.ranking.dense import rocchio
from research_assistant.ranking.dpp import dpp_greedy, dpp_kernel
from research_assistant.ranking.fusion import ranks_from_scores, rrf
from research_assistant.ranking.pagerank import personalized_pagerank
from research_assistant.ranking.select import SelectionObjective, lazy_greedy, mmr, topk
from research_assistant.sources.openalex import OpenAlex, Work
from research_assistant.text.tokenize import tokenize

DEFAULT_RATES = {"offline": JudgeRates(0.8, 0.6), "claude": JudgeRates(0.9, 0.8)}


@dataclass
class Candidate:
    id: str
    title: str
    abstract: str
    embedding: np.ndarray
    year: int | None = None
    publication_date: str | None = None
    doi: str | None = None
    venue: str | None = None
    cited_by_count: int = 0
    authors: tuple[str, ...] = ()
    referenced_ids: tuple[str, ...] = ()
    arxiv_id: str | None = None
    relevance_score: float | None = None  # OpenAlex's own score, for search hits only

    @classmethod
    def from_work(cls, w: Work, embedding: np.ndarray) -> "Candidate":
        return cls(id=w.id, title=w.title, abstract=w.abstract, embedding=embedding, year=w.year,
                   publication_date=w.publication_date, doi=w.doi, venue=w.venue, cited_by_count=w.cited_by_count,
                   authors=w.authors, referenced_ids=w.referenced_ids, arxiv_id=w.arxiv_id,
                   relevance_score=w.relevance_score)

    def view(self, **extra) -> PaperView:
        return PaperView(id=self.id, title=self.title, abstract=self.abstract, year=self.year, **extra)


@dataclass
class Signals:
    bm25: np.ndarray  # (n, F) BM25F score per facet
    dense: np.ndarray  # (n, F) cosine per facet, after Rocchio when enabled
    ppr: np.ndarray  # (n,) Personalized PageRank
    facet_rel: np.ndarray  # (n, F) facet relevance in [0, 1]
    fused: np.ndarray  # (n,) weighted RRF over every enabled list
    ranks: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Selection:
    indices: list[int]  # into the shortlist passed to select()
    method: str
    objective: float | None


class LocalSearch(Protocol):
    async def search(self, facet_vec: np.ndarray, keywords: str, limit: int,
                     published_before: str | None, exclude: set[str]) -> list[Candidate]: ...


class NoPapersFound(Exception):
    """Nothing usable came back for the topic."""


def embed_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}"


def facet_terms(plan: Plan) -> list[list[str]]:
    return [list(dict.fromkeys(tokenize(f"{f.question} {' '.join(f.keywords)}"))) for f in plan.facets]


async def embed_works(works: list[Work], embedder: Embedder) -> list[Candidate]:
    if not works:
        return []
    vectors = await asyncio.to_thread(embedder.embed_docs, [embed_text(w.title, w.abstract) for w in works])
    return [Candidate.from_work(w, v) for w, v in zip(works, vectors)]


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    """Drop repeats by OpenAlex id, then DOI, then normalized title (titles of 20+ letters only, so
    short generic titles like 'Introduction' never merge different papers)."""
    ids, dois, titles, out = set(), set(), set(), []
    for c in cands:
        title = re.sub(r"[^a-z0-9]", "", c.title.lower())
        if c.id in ids or (c.doi and c.doi in dois) or (len(title) >= 20 and title in titles):
            continue
        ids.add(c.id)
        if c.doi:
            dois.add(c.doi)
        if len(title) >= 20:
            titles.add(title)
        out.append(c)
    return out


def in_years(c: Candidate, year_from: int | None, year_to: int | None) -> bool:
    if c.year is None:
        return True
    return (year_from is None or c.year >= year_from) and (year_to is None or c.year <= year_to)


def text_signals(pool: list[Candidate], plan: Plan, facet_vecs: np.ndarray, algo: AlgoParams) -> tuple[np.ndarray, np.ndarray]:
    docs = [{"title": tokenize(c.title), "abstract": tokenize(c.abstract)} for c in pool]
    fields = {"title": FieldSpec(algo.bm25_title_weight, algo.bm25_title_b),
              "abstract": FieldSpec(algo.bm25_abstract_weight, algo.bm25_abstract_b)}
    bm25 = np.stack([bm25f_scores(docs, terms, fields, algo.bm25_k1) for terms in facet_terms(plan)], axis=1)
    E = np.stack([c.embedding for c in pool]).astype(np.float32)
    return bm25, E @ np.asarray(facet_vecs, dtype=np.float32).T


def facet_rrf(bm25: np.ndarray, dense: np.ndarray, k: float) -> np.ndarray:
    """(n, F): each facet's two-list RRF. A zero BM25 score means 'no query word', so it counts as absent."""
    cols = []
    for f in range(bm25.shape[1]):
        cols.append(rrf([ranks_from_scores(bm25[:, f], present=bm25[:, f] > 0), ranks_from_scores(dense[:, f])], k=k))
    return np.stack(cols, axis=1)


def first_pass_scores(pool: list[Candidate], plan: Plan, facet_vecs: np.ndarray, algo: AlgoParams) -> np.ndarray:
    bm25, dense = text_signals(pool, plan, facet_vecs, algo)
    return facet_rrf(bm25, dense, algo.rrf_k).mean(axis=1)


async def gather(
    plan: Plan,
    *,
    openalex: OpenAlex,
    embedder: Embedder,
    algo: AlgoParams,
    local: LocalSearch | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    published_before: str | None = None,
    exclude_ids: list[str] | tuple[str, ...] = (),
) -> tuple[list[Candidate], list[Candidate]]:
    """Build the pool: local store, OpenAlex keyword search per facet, then citation snowballing.

    Returns (pool, new): `new` are candidates that came from OpenAlex and should be saved.
    """
    exclude = set(exclude_ids)
    facet_vecs = await asyncio.to_thread(embedder.embed_queries, [f.question for f in plan.facets])
    found: dict[str, Candidate] = {}
    if local is not None:
        for facet, vec in zip(plan.facets, facet_vecs):
            for c in await local.search(vec, " ".join(facet.keywords), algo.local_top, published_before, exclude):
                if in_years(c, year_from, year_to):
                    found.setdefault(c.id, c)
    searched: dict[str, Work] = {}
    for facet in plan.facets:
        for w in await openalex.search(" ".join(facet.keywords), year_from=year_from, year_to=year_to,
                                       published_before=published_before, pages=algo.openalex_pages):
            if w.id not in found and w.id not in exclude:
                searched.setdefault(w.id, w)
    pool = dedupe(list(found.values()) + await embed_works(list(searched.values()), embedder))
    if not pool:
        return [], []

    # Snowballing: follow citations out of (references) and into (citing works) the strongest papers.
    first = await asyncio.to_thread(first_pass_scores, pool, plan, facet_vecs, algo)
    seeds = [pool[i] for i in np.argsort(-first, kind="stable")[: algo.snowball_seeds]]
    have = {c.id for c in pool}
    backward = [r for s in seeds for r in s.referenced_ids if r not in have and r not in exclude]
    snow = await openalex.by_ids(backward, published_before=published_before)
    for s in seeds[: algo.snowball_forward_seeds]:
        snow += await openalex.citing(s.id, year_from=year_from, year_to=year_to, published_before=published_before)
    fresh = {w.id: w for w in snow if w.id not in have and w.id not in exclude}
    extra = [c for c in await embed_works(list(fresh.values()), embedder) if in_years(c, year_from, year_to)]
    pool = dedupe(pool + extra)

    if len(pool) > algo.pool_cap:
        first = await asyncio.to_thread(first_pass_scores, pool, plan, facet_vecs, algo)
        keep = sorted(np.argsort(-first, kind="stable")[: algo.pool_cap].tolist())
        pool = [pool[i] for i in keep]
    new_ids = set(searched) | set(fresh)
    return pool, [c for c in pool if c.id in new_ids]


def compute_signals(
    pool: list[Candidate],
    plan: Plan,
    facet_vecs: np.ndarray,
    algo: AlgoParams,
    *,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_ppr: bool = True,
    use_rocchio: bool = True,
) -> Signals:
    """Every ranking signal for the pool, plus the fused score. The use_* flags exist for ablations."""
    if not (use_bm25 or use_dense or use_ppr):
        raise ValueError("enable at least one signal")
    k = algo.rrf_k
    F = len(plan.facets)
    bm25, dense = text_signals(pool, plan, facet_vecs, algo)
    E = np.stack([c.embedding for c in pool]).astype(np.float32)
    if use_rocchio:
        first = facet_rrf(bm25, dense, k)
        refined = np.stack([
            rocchio(facet_vecs[f], E[np.argsort(-first[:, f], kind="stable")[: algo.rocchio_top]],
                    alpha=algo.rocchio_alpha, beta=algo.rocchio_beta, gamma=algo.rocchio_gamma)
            for f in range(F)
        ])
        dense = E @ refined.T
    bm25_ranks = [ranks_from_scores(bm25[:, f], present=bm25[:, f] > 0) for f in range(F)]
    dense_ranks = [ranks_from_scores(dense[:, f]) for f in range(F)]
    ranks: dict[str, np.ndarray] = {}
    lists, weights = [], []
    for f in range(F):
        ranks[f"bm25_f{f}"], ranks[f"dense_f{f}"] = bm25_ranks[f], dense_ranks[f]
        if use_bm25:
            lists.append(bm25_ranks[f])
            weights.append(1.0 / F)
        if use_dense:
            lists.append(dense_ranks[f])
            weights.append(1.0 / F)
    n = len(pool)
    ppr = np.zeros(n)
    if use_ppr:
        text = rrf(lists, weights, k) if lists else rrf(bm25_ranks + dense_ranks, k=k)
        index = {c.id: i for i, c in enumerate(pool)}
        edges = np.array([(i, index[r]) for i, c in enumerate(pool) for r in c.referenced_ids if r in index], dtype=np.int64).reshape(-1, 2)
        seeds = np.argsort(-text, kind="stable")[: algo.ppr_seeds]
        teleport = np.zeros(n)
        teleport[seeds] = text[seeds]
        ppr, _ = personalized_pagerank(n, edges, teleport, algo.ppr_alpha, algo.ppr_tol, algo.ppr_max_iter)
        ranks["ppr"] = ranks_from_scores(ppr, present=ppr > 0)
        lists.append(ranks["ppr"])
        weights.append(algo.rrf_graph_weight)
    fused = rrf(lists, weights, k)
    ranks["fused"] = ranks_from_scores(fused)
    facet_rel = np.stack([(1.0 / (k + bm25_ranks[f]) + 1.0 / (k + dense_ranks[f])) / (2.0 / (k + 1)) for f in range(F)], axis=1)
    return Signals(bm25=bm25, dense=dense, ppr=ppr, facet_rel=facet_rel, fused=fused, ranks=ranks)


def shortlist(signals: Signals, m: int) -> list[int]:
    return np.argsort(-signals.fused, kind="stable")[:m].tolist()


def judge_rates_for(brain_name: str, calibration: dict | None) -> JudgeRates:
    measured = (calibration or {}).get("judge_rates", {}).get(brain_name)
    if measured:
        return JudgeRates(measured["sensitivity"], measured["specificity"])
    return DEFAULT_RATES.get(brain_name, JudgeRates(0.8, 0.6))


async def screen(
    brain: Brain,
    plan: Plan,
    shortlisted: list[Candidate],
    facet_rel: np.ndarray,
    algo: AlgoParams,
    calibration: dict | None = None,
) -> tuple[dict[str, Verdict], np.ndarray]:
    """Ask the brain about each shortlisted paper (in batches, a few at a time), then turn each
    verdict into P(relevant) with Bayes' rule. A paper with no verdict keeps its prior."""
    m = len(shortlisted)
    views = [c.view(rank=r + 1, shortlist_size=m, facet_relevance=[round(float(x), 3) for x in facet_rel[r]])
             for r, c in enumerate(shortlisted)]
    gate = asyncio.Semaphore(algo.screen_concurrency)

    async def judge(batch: list[PaperView]) -> list[Verdict]:
        async with gate:
            return await brain.screen(plan, batch)

    batches = [views[s:s + algo.screen_batch] for s in range(0, m, algo.screen_batch)]
    try:
        # A task group cancels the batches still waiting once one fails (for example over budget);
        # asyncio.gather would let them run on and keep spending.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(judge(b)) for b in batches]
    except ExceptionGroup as group:
        raise group.exceptions[0] from None
    verdicts = {v.paper_id: v for t in tasks for v in t.result()}
    rates = judge_rates_for(brain.name, calibration)
    table = (calibration or {}).get("prior_table")
    post = []
    for r, c in enumerate(shortlisted):
        prior = prior_from_table(r + 1, table) if table else linear_prior(r + 1, m, algo.prior_top, algo.prior_bottom)
        verdict = verdicts.get(c.id)
        post.append(prior if verdict is None else posterior(prior, verdict.include, rates))
    return verdicts, np.array(post)


def select(
    method: str,
    shortlisted: list[Candidate],
    post: np.ndarray,
    facet_rel: np.ndarray,
    k: int,
    algo: AlgoParams,
    must: list[int] | tuple[int, ...] = (),
) -> Selection:
    """Pick k papers from the shortlist. `must` holds shortlist positions that are always kept."""
    if not shortlisted:
        return Selection([], method, None)
    E = np.stack([c.embedding for c in shortlisted]).astype(float)
    sim = E @ E.T
    k = min(k, len(shortlisted))
    if method == "topk":
        return Selection(topk(post, k, must), method, None)
    if method == "mmr":
        return Selection(mmr(post, sim, k, algo.mmr_lambda, must), method, None)
    if method == "submodular":
        objective = SelectionObjective(post, facet_rel, np.maximum(sim, 0.0), mu=algo.select_mu)
        chosen = lazy_greedy(objective, k, must)
        return Selection(chosen, method, objective.value(chosen))
    if method == "dpp":
        chosen, log_det = dpp_greedy(dpp_kernel(E, np.exp(algo.dpp_theta * post), algo.dpp_eps), k, must)
        return Selection(chosen, method, log_det)
    raise ValueError(f"unknown selection method: {method}")
