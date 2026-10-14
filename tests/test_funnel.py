import numpy as np
import pytest

from research_assistant import funnel
from research_assistant.brain.base import Plan, Verdict
from research_assistant.brain.offline import OfflineBrain
from research_assistant.config import AlgoParams
from research_assistant.embed import HashEmbedder
from research_assistant.sources.openalex import OpenAlex
from tests.fakes import fake_http, make_world, openalex_handler

ALGO = AlgoParams(openalex_pages=1, snowball_seeds=5, snowball_forward_seeds=2, ppr_seeds=10, rocchio_top=5)
TOPIC = "retrieval augmented generation grounding"


@pytest.fixture
def world():
    return make_world()


async def build_pool(world, **kwargs):
    plan = await OfflineBrain().plan(TOPIC)
    oa = OpenAlex(fake_http(openalex_handler(world)))
    pool, new = await funnel.gather(plan, openalex=oa, embedder=HashEmbedder(), algo=ALGO, **kwargs)
    return plan, pool, new, oa


async def test_gather_uses_search_and_snowballing(world):
    plan, pool, new, oa = await build_pool(world)
    ids = {c.id for c in pool}
    assert len(pool) == len(ids) and {c.id for c in new} == ids
    assert sum(i.startswith("W1") for i in ids) >= 25  # nearly all on-topic papers were found
    assert oa.http.cost_usd > 0


async def test_gather_respects_exclusions_and_dates(world):
    _, pool, _, _ = await build_pool(world, exclude_ids=["W1000", "W1002"], published_before="2019-12-31")
    ids = {c.id for c in pool}
    assert "W1000" not in ids and "W1002" not in ids
    assert all(c.publication_date <= "2019-12-31" for c in pool)


async def test_gather_with_no_hits_returns_empty(world):
    plan = Plan(facets=[{"id": 0, "question": "zzz", "keywords": ["zzzz"]}], criteria={"include": ["x"], "exclude": ["y"]})
    pool, new = await funnel.gather(plan, openalex=OpenAlex(fake_http(openalex_handler(world))), embedder=HashEmbedder(), algo=ALGO)
    assert pool == [] and new == []


async def test_signals_rank_on_topic_papers_first(world):
    plan, pool, _, _ = await build_pool(world)
    vecs = HashEmbedder().embed_queries([f.question for f in plan.facets])
    sig = funnel.compute_signals(pool, plan, vecs, ALGO)
    n, F = len(pool), len(plan.facets)
    assert sig.bm25.shape == (n, F) and sig.facet_rel.shape == (n, F) and sig.fused.shape == (n,)
    assert sig.ppr.sum() == pytest.approx(1.0)
    assert ((sig.facet_rel > 0) & (sig.facet_rel <= 1.0 + 1e-9)).all()
    top = [pool[i].id for i in funnel.shortlist(sig, 15)]
    assert sum(i.startswith("W1") for i in top) >= 13


async def test_ablation_flags(world):
    plan, pool, _, _ = await build_pool(world)
    vecs = HashEmbedder().embed_queries([f.question for f in plan.facets])
    only_bm25 = funnel.compute_signals(pool, plan, vecs, ALGO, use_dense=False, use_ppr=False, use_rocchio=False)
    assert "ppr" not in only_bm25.ranks and np.allclose(only_bm25.ppr, 0)
    with pytest.raises(ValueError):
        funnel.compute_signals(pool, plan, vecs, ALGO, use_bm25=False, use_dense=False, use_ppr=False)


async def test_screen_turns_verdicts_into_probabilities(world):
    plan, pool, _, _ = await build_pool(world)
    vecs = HashEmbedder().embed_queries([f.question for f in plan.facets])
    sig = funnel.compute_signals(pool, plan, vecs, ALGO)
    short_idx = funnel.shortlist(sig, 20)
    short = [pool[i] for i in short_idx]
    verdicts, post = await funnel.screen(OfflineBrain(), plan, short, sig.facet_rel[short_idx], ALGO)
    assert len(verdicts) == 20 and post.shape == (20,) and ((post > 0) & (post < 1)).all()
    included = [post[i] for i, c in enumerate(short) if verdicts[c.id].include]
    excluded = [post[i] for i, c in enumerate(short) if not verdicts[c.id].include]
    assert min(included) > max(excluded)


async def test_screen_missing_verdict_keeps_prior():
    class Silent:
        name = "offline"

        async def screen(self, plan, papers):
            return [Verdict(paper_id=papers[0].id, include=True, facets=[], reason="")]

    plan = await OfflineBrain().plan(TOPIC)
    cands = [funnel.Candidate(id=f"W{i}", title="t", abstract="a", embedding=np.ones(4) / 2) for i in range(3)]
    _, post = await funnel.screen(Silent(), plan, cands, np.zeros((3, 5)), AlgoParams(screen_batch=10),
                                  calibration={"prior_table": [[1, None, 0.3]]})
    assert post[0] > 0.3 and post[1] == pytest.approx(0.3) and post[2] == pytest.approx(0.3)


@pytest.mark.parametrize("method", ["topk", "mmr", "submodular", "dpp"])
async def test_select_methods_honor_k_and_must(world, method):
    plan, pool, _, _ = await build_pool(world)
    vecs = HashEmbedder().embed_queries([f.question for f in plan.facets])
    sig = funnel.compute_signals(pool, plan, vecs, ALGO)
    short_idx = funnel.shortlist(sig, 30)
    short = [pool[i] for i in short_idx]
    post = np.linspace(0.9, 0.1, 30)
    sel = funnel.select(method, short, post, sig.facet_rel[short_idx], 10, ALGO, must=[29])
    assert len(sel.indices) == 10 == len(set(sel.indices)) and sel.indices[0] == 29
    assert (sel.objective is not None) == (method in ("submodular", "dpp"))


def test_select_on_tiny_pool_and_bad_method():
    cands = [funnel.Candidate(id=f"W{i}", title="t", abstract="a", embedding=np.eye(3)[i]) for i in range(3)]
    sel = funnel.select("submodular", cands, np.array([0.5, 0.4, 0.3]), np.ones((3, 2)) * 0.5, 40, ALGO)
    assert sorted(sel.indices) == [0, 1, 2]
    assert funnel.select("dpp", [], np.array([]), np.zeros((0, 2)), 40, ALGO).indices == []
    with pytest.raises(ValueError):
        funnel.select("random", cands, np.ones(3), np.ones((3, 2)), 2, ALGO)


def test_dedupe_by_id_doi_and_long_title():
    e = np.ones(2)
    cands = [
        funnel.Candidate(id="W1", title="A long enough title about retrieval", abstract="", embedding=e, doi="10.1/a"),
        funnel.Candidate(id="W1", title="dup id", abstract="", embedding=e),
        funnel.Candidate(id="W2", title="other", abstract="", embedding=e, doi="10.1/a"),
        funnel.Candidate(id="W3", title="A long enough title about retrieval!", abstract="", embedding=e),
        funnel.Candidate(id="W4", title="Intro", abstract="", embedding=e),
        funnel.Candidate(id="W5", title="Intro", abstract="", embedding=e),
    ]
    assert [c.id for c in funnel.dedupe(cands)] == ["W1", "W4", "W5"]


async def test_screen_stops_other_batches_when_one_fails():
    import asyncio

    calls = []

    class Failing:
        name = "offline"

        async def screen(self, plan, papers):
            calls.append(len(papers))
            await asyncio.sleep(0.01)  # a real call waits on the network; the budget check runs after it returns
            raise RuntimeError("over budget")

    plan = await OfflineBrain().plan(TOPIC)
    cands = [funnel.Candidate(id=f"W{i}", title="t", abstract="a", embedding=np.ones(4) / 2) for i in range(10)]
    with pytest.raises(RuntimeError, match="over budget"):
        await funnel.screen(Failing(), plan, cands, np.zeros((10, 5)), AlgoParams(screen_batch=2, screen_concurrency=1))
    await asyncio.sleep(0.2)  # give any orphaned batches time to run
    assert len(calls) <= 2  # the batch already woken may start; the other three never do


async def test_local_store_hits_respect_the_year_range(world):
    vec = HashEmbedder().embed_docs(["retrieval augmented generation grounding"])[0]

    class Local:
        async def search(self, facet_vec, keywords, limit, published_before, exclude):
            return [
                funnel.Candidate(id="WOLD", title="Retrieval augmented generation grounding, old", abstract="retrieval", embedding=vec, year=2001),
                funnel.Candidate(id="WNEW", title="Retrieval augmented generation grounding, new", abstract="retrieval", embedding=vec, year=2022),
            ]

    plan = await OfflineBrain().plan(TOPIC)
    pool, _ = await funnel.gather(plan, openalex=OpenAlex(fake_http(openalex_handler(world))), embedder=HashEmbedder(), algo=ALGO,
                                  local=Local(), year_from=2020, year_to=2024)
    ids = {c.id for c in pool}
    assert "WNEW" in ids and "WOLD" not in ids
