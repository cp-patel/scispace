import asyncio

import pytest

from research_assistant.brain.base import Usage
from research_assistant.config import JobSettings
from research_assistant.db import jobs, papers, results
from research_assistant.embed import HashEmbedder
from research_assistant.funnel import Candidate

pytestmark = pytest.mark.db


EMB = HashEmbedder()


def cand(i: int, title: str, year: int = 2020, date: str | None = None) -> Candidate:
    return Candidate(id=f"W{i}", title=title, abstract=f"{title} abstract text", embedding=EMB.embed_docs([title])[0],
                     year=year, publication_date=date or f"{year}-01-01", authors=("Ana Smith",), referenced_ids=("W99",))


async def test_two_workers_never_claim_the_same_job(db):
    await jobs.create_job(db, JobSettings(topic="a topic long enough"))
    claimed = await asyncio.gather(jobs.claim_job(db, "a", 60), jobs.claim_job(db, "b", 60))
    assert sum(c is not None for c in claimed) == 1


async def test_lease_expiry_requeues_then_fails_after_max_attempts(db):
    job_id = await jobs.create_job(db, JobSettings(topic="a topic long enough"))
    for attempt in range(1, 4):
        job = await jobs.claim_job(db, "w", 60)
        assert job["attempts"] == attempt
        async with db.connection() as conn:
            await conn.execute("UPDATE jobs SET lease_until = now() - interval '1 second' WHERE id = %s", (job_id,))
        assert await jobs.reap_expired(db, max_attempts=3) == 1
    job = await jobs.get_job(db, job_id)
    assert job["status"] == "failed" and "lease expired" in job["error"]


async def test_lease_renewal_status_and_state_conflicts(db):
    job_id = await jobs.create_job(db, JobSettings(topic="a topic long enough"))
    with pytest.raises(jobs.StateConflict):
        await jobs.approve(db, job_id, {"remove": []})
    await jobs.claim_job(db, "w1", 60)
    assert await jobs.renew_lease(db, job_id, "w1", 60)
    assert not await jobs.renew_lease(db, job_id, "w2", 60)
    assert await jobs.run_status(db, job_id, "w1") == "running"
    assert await jobs.run_status(db, job_id, "w2") == "lost"
    await jobs.wait_for_approval(db, job_id, "w1")
    await jobs.approve(db, job_id, {"remove": ["W1"]})
    assert (await jobs.get_job(db, job_id))["status"] == "queued" and await jobs.get_resume(db, job_id) == {"remove": ["W1"]}
    await jobs.cancel(db, job_id)
    assert await jobs.run_status(db, job_id, None) == "canceled"
    with pytest.raises(jobs.StateConflict):
        await jobs.cancel(db, job_id)


async def test_llm_costs_add_up(db):
    job_id = await jobs.create_job(db, JobSettings(topic="a topic long enough"))
    usage = Usage(model="claude-opus-5", input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.25, latency_ms=3)
    assert await jobs.add_llm_call(db, job_id, "plan", usage) == pytest.approx(0.25)
    assert await jobs.add_llm_call(db, job_id, "screen", usage) == pytest.approx(0.5)
    assert await jobs.cost_by_node(db, job_id) == {"plan": pytest.approx(0.25), "screen": pytest.approx(0.25)}


async def test_results_round_trip(db):
    job_id = await jobs.create_job(db, JobSettings(topic="a topic long enough"))
    await papers.upsert_papers(db, [cand(1, "retrieval augmented generation"), cand(2, "graph networks")])
    await results.save_job_papers(db, job_id, [
        {"paper_id": "W1", "signals": {"facet_rel": [0.9]}, "ranks": {"fused": 1}, "fused": 0.5, "fused_rank": 1},
        {"paper_id": "W2", "signals": {"facet_rel": [0.1]}, "ranks": {"fused": 2}, "fused": 0.1, "fused_rank": 2},
    ])
    await results.save_screening(db, job_id, [("W1", {"include": True}, 0.8), ("W2", None, 0.2)])
    await results.mark_selected(db, job_id, ["W1"])
    table = await results.selection_table(db, job_id)
    assert [(r["paper_id"], r["selected"], r["selection_order"]) for r in table] == [("W1", True, 1), ("W2", False, None)]
    await results.save_sections(db, job_id, [{"idx": 0, "title": "T", "goal": "G", "paper_ids": ["W1"]}])
    await results.update_section(db, job_id, 0, draft="text", verification={"flags": {}})
    assert (await results.get_section(db, job_id, 0))["draft"] == "text"
    with pytest.raises(ValueError):
        await results.update_section(db, job_id, 0, title="nope")
    await results.save_outputs(db, job_id, "# paper", "@article{x}", {"a": 1}, "# report")
    assert (await results.outputs(db, job_id))["report"] == {"a": 1}
