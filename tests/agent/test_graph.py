"""The whole agent, end to end, against the fake world: offline brain, hash embeddings, real Postgres."""

import asyncio
import contextlib
import re

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from research_assistant.agent.graph import build_graph
from research_assistant.agent.nodes import Deps, JobNodes, record_usage
from research_assistant.brain.base import Usage
from research_assistant.brain.offline import OfflineBrain
from research_assistant.citations import quote_in_source
from research_assistant.config import AlgoParams, JobSettings, Settings
from research_assistant.db import jobs, papers, results
from research_assistant.embed import HashEmbedder
from research_assistant.sources.arxiv import Arxiv
from research_assistant.sources.openalex import OpenAlex
from research_assistant.worker import Worker
from tests.fakes import arxiv_handler, fake_http, make_world, openalex_handler

pytestmark = pytest.mark.db
TOPIC = "retrieval augmented generation grounding"
ALGO = AlgoParams(openalex_pages=1, snowball_seeds=5, snowball_forward_seeds=2, ppr_seeds=10, rocchio_top=5,
                  section_budget_tokens=2000, max_cards=6)


async def make_worker(db, brain=None, worker_id="w1") -> Worker:
    world = make_world()
    deps = Deps(pool=db, brains={"offline": brain or OfflineBrain(max_cards=6)}, embedder=HashEmbedder(),
                openalex=OpenAlex(fake_http(openalex_handler(world), "openalex")),
                arxiv=Arxiv(fake_http(arxiv_handler(world), "arxiv")),
                settings=Settings(_env_file=None, algo=ALGO))
    saver = AsyncPostgresSaver(db)
    await saver.setup()
    return Worker(deps, build_graph(JobNodes(deps), saver), worker_id)


async def new_job(db, **kwargs) -> str:
    return await jobs.create_job(db, JobSettings(topic=TOPIC, max_papers=12, shortlist_size=30, **kwargs))


async def test_full_job_produces_a_checked_paper(db):
    worker = await make_worker(db)
    job_id = await new_job(db)
    assert await worker.run_one()
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "succeeded", job["error"]
    out = await results.outputs(db, job_id)
    md = out["paper_md"]
    assert md.startswith("# Retrieval augmented generation grounding: a literature review")
    assert "[E" not in md  # every card marker became a reference number
    numbers = {int(n) for n in re.findall(r"\[(\d+)\] ", md)}
    assert numbers and numbers == set(range(1, len(numbers) + 1))
    selected = {r["paper_id"] for r in await results.selection_table(db, job_id) if r["selected"]}
    assert len(selected) == 12
    cited_titles = re.findall(r"^\[\d+\] .*?\(\d{4}\)\. (.*?)\.", md, re.M)
    selected_titles = {c.title for c in await papers.load_candidates(db, list(selected))}
    assert cited_titles and set(cited_titles) <= selected_titles
    for card in await results.evidence(db, job_id):
        chunk = (await papers.load_chunks_by_id(db, [card["chunk_id"]]))[0]
        assert quote_in_source(card["quote"], chunk.text)
    report = out["report"]
    assert report["funnel"]["selected"] == 12 and report["ingest"].get("pdf", 0) > 0
    # The report is written inside assemble, so assemble itself is not listed as finished yet.
    assert report["outline"]["k"] >= 2 and {s["node"] for s in report["steps"]} >= {"plan", "rank", "select", "outline", "finish"}


async def test_approval_pause_and_resume_with_edits(db):
    worker = await make_worker(db)
    job_id = await new_job(db, auto_approve=False)
    await worker.run_one()
    assert (await jobs.get_job(db, job_id))["status"] == "waiting_approval"
    picked = [r["paper_id"] for r in await results.selection_table(db, job_id) if r["selected"]]
    await jobs.approve(db, job_id, {"remove": [picked[0]], "add": []})
    await worker.run_one()
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "succeeded", job["error"]
    still = {r["paper_id"] for r in await results.selection_table(db, job_id) if r["selected"]}
    assert picked[0] not in still and len(still) == 11


class HangingBrain(OfflineBrain):
    """Stops forever inside outline, so the test can 'crash' the worker mid-job."""

    def __init__(self, reached: asyncio.Event):
        super().__init__(max_cards=6)
        self.reached = reached

    async def name_sections(self, clusters):
        self.reached.set()
        await asyncio.sleep(3600)


async def test_crash_mid_job_resumes_from_checkpoint_on_another_worker(db):
    reached = asyncio.Event()
    first = await make_worker(db, HangingBrain(reached), "w1")
    job_id = await new_job(db)
    task = asyncio.create_task(first.run_one())
    await asyncio.wait_for(reached.wait(), timeout=60)
    task.cancel()  # the worker process "dies": no status update, the lease is still held
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert (await jobs.get_job(db, job_id))["status"] == "running"
    async with db.connection() as conn:
        await conn.execute("UPDATE jobs SET lease_until = now() - interval '1 second' WHERE id = %s", (job_id,))
    assert await jobs.reap_expired(db, max_attempts=3) == 1
    second = await make_worker(db, worker_id="w2")
    assert await second.run_one()
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "succeeded", job["error"]
    starts = [e["node"] for e in await jobs.events(db, job_id) if e["event"] == "start"]
    assert starts.count("gather") == 1 and starts.count("ingest") == 1  # finished steps were not redone
    assert starts.count("outline") == 2  # the interrupted step ran again
    sections = await results.sections(db, job_id)
    assert len({s["idx"] for s in sections}) == len(sections)


class CancelingBrain(OfflineBrain):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.job_id = None

    async def plan(self, topic):
        await jobs.cancel(self.db, self.job_id)
        return await super().plan(topic)


async def test_cancel_stops_the_job_at_the_next_step(db):
    brain = CancelingBrain(db)
    worker = await make_worker(db, brain)
    brain.job_id = await new_job(db)
    await worker.run_one()
    job = await jobs.get_job(db, brain.job_id)
    assert job["status"] == "canceled"
    assert "gather" not in {e["node"] for e in await jobs.events(db, brain.job_id)}


class SpendyBrain(OfflineBrain):
    def __init__(self, db):
        super().__init__()
        self.db = db

    async def plan(self, topic):
        await record_usage(self.db, "plan", Usage(model="claude-opus-5", input_tokens=1, output_tokens=1,
                                                  cache_read_tokens=0, cache_write_tokens=0, cost_usd=5.0, latency_ms=1))
        return await super().plan(topic)


async def test_budget_cap_fails_the_job(db):
    worker = await make_worker(db, SpendyBrain(db))
    job_id = await new_job(db, budget_usd=1.0)
    await worker.run_one()
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "failed" and "BudgetExceeded" in job["error"]


async def test_topic_with_no_papers_fails_clearly(db):
    worker = await make_worker(db)
    job_id = await jobs.create_job(db, JobSettings(topic="zzzz qqqq xxxx vvvv"))
    await worker.run_one()
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "failed" and "NoPapersFound" in job["error"]


async def test_approval_that_removes_every_paper_fails_clearly(db):
    worker = await make_worker(db)
    job_id = await new_job(db, auto_approve=False)
    await worker.run_one()
    picked = [r["paper_id"] for r in await results.selection_table(db, job_id) if r["selected"]]
    await jobs.approve(db, job_id, {"remove": picked, "add": []})
    await worker.run_one()
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "failed" and "NoPapersFound" in job["error"]
