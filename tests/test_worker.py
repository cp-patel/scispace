import asyncio

from research_assistant.agent.nodes import Deps
from research_assistant.config import Settings
from research_assistant.db import jobs
from research_assistant.worker import Worker


def make_worker() -> Worker:
    settings = Settings(_env_file=None, heartbeat_seconds=0.01)
    deps = Deps(pool=None, brains={}, embedder=None, openalex=None, arxiv=None, settings=settings)
    return Worker(deps, graph=None, worker_id="w")


async def test_heartbeat_survives_a_transient_database_error(monkeypatch):
    calls = []

    async def flaky(pool, job_id, worker_id, lease_seconds):
        calls.append(job_id)
        if len(calls) == 1:
            raise OSError("connection reset")
        return True

    monkeypatch.setattr(jobs, "renew_lease", flaky)
    task = asyncio.create_task(make_worker()._heartbeat("J1"))
    await asyncio.sleep(0.1)
    task.cancel()
    assert len(calls) >= 3


async def test_worker_loop_survives_an_error(monkeypatch):
    worker = make_worker()
    attempts = []

    async def flaky_reap(pool, max_attempts):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("database restarting")
        if len(attempts) >= 3:
            worker.deps.stopping.set()
        return 0

    async def zero(pool):
        return 0

    async def nothing(pool, worker_id, lease_seconds):
        return None

    monkeypatch.setattr(jobs, "reap_expired", flaky_reap)
    monkeypatch.setattr(jobs, "queue_depth", zero)
    monkeypatch.setattr(jobs, "claim_job", nothing)
    await asyncio.wait_for(worker.run_forever(poll_seconds=0.01), timeout=5)
    assert len(attempts) >= 3
