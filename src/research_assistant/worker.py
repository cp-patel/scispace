"""Worker process: claim jobs from Postgres and run the agent graph.

Run with: python -m research_assistant.worker
A heartbeat renews the lease while a job runs; any worker's reaper requeues jobs whose lease ran
out (the worker died). LangGraph checkpoints let the next worker continue from the last finished step.
"""

import asyncio
import logging
import os
import signal
import socket
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from prometheus_client import start_http_server

from research_assistant.agent.graph import build_graph
from research_assistant.agent.nodes import Deps, JobCanceled, JobNodes, LeaseLost, WorkerStopping
from research_assistant.config import get_settings
from research_assistant.db import apply_schema, create_pool, jobs
from research_assistant.observability import JOBS_TOTAL, QUEUE_DEPTH, JobContext, job_context, setup_logging

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, deps: Deps, graph, worker_id: str | None = None):
        self.deps = deps
        self.graph = graph
        self.settings = deps.settings
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    async def run_forever(self, poll_seconds: float = 2.0) -> None:
        while not self.deps.stopping.is_set():
            try:
                await jobs.reap_expired(self.deps.pool, self.settings.max_attempts)
                QUEUE_DEPTH.set(await jobs.queue_depth(self.deps.pool))
                busy = await self.run_one()
            except Exception:
                log.exception("worker loop error; retrying after a pause")
                busy = False
            if not busy:
                try:
                    await asyncio.wait_for(self.deps.stopping.wait(), timeout=poll_seconds)
                except TimeoutError:
                    pass

    async def run_one(self) -> bool:
        job = await jobs.claim_job(self.deps.pool, self.worker_id, self.settings.lease_seconds)
        if job is None:
            return False
        await self.run_job(job)
        return True

    async def run_job(self, job: dict) -> str:
        job_id = str(job["id"])
        config = {"configurable": {"thread_id": job_id}}
        token = job_context.set(JobContext(job_id=job_id, worker_id=self.worker_id))
        heartbeat = asyncio.create_task(self._heartbeat(job_id))
        try:
            payload = await self._payload(job_id, job["settings"], config)
            async with asyncio.timeout(self.settings.job_deadline_min * 60):
                out = await self.graph.ainvoke(payload, config, durability="sync")
            if out.get("__interrupt__"):
                await jobs.wait_for_approval(self.deps.pool, job_id, self.worker_id)
                return "waiting_approval"
            await jobs.finish_job(self.deps.pool, job_id, self.worker_id, "succeeded")
            status = "succeeded"
        except JobCanceled:
            status = "canceled"  # the API already set the status
        except LeaseLost:
            log.warning("lost the lease; another worker may own this job now")
            return "lost"
        except WorkerStopping:
            await jobs.requeue(self.deps.pool, job_id, self.worker_id)
            return "queued"
        except TimeoutError:
            await jobs.finish_job(self.deps.pool, job_id, self.worker_id, "failed", "the job passed its deadline")
            status = "failed"
        except Exception as exc:
            log.exception("job failed")
            await jobs.finish_job(self.deps.pool, job_id, self.worker_id, "failed", f"{type(exc).__name__}: {exc}")
            status = "failed"
        finally:
            heartbeat.cancel()
            job_context.reset(token)
        JOBS_TOTAL.labels(status).inc()
        return status

    async def _payload(self, job_id: str, settings_json: dict, config: dict):
        """Fresh start, plain resume after a crash, or resume with the human's approval answer."""
        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            return {"job_id": job_id, "settings": settings_json}
        if any(task.interrupts for task in snapshot.tasks):
            answer = await jobs.get_resume(self.deps.pool, job_id)
            return Command(resume=answer) if answer is not None else None
        return None

    async def _heartbeat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.settings.heartbeat_seconds)
            try:
                renewed = await jobs.renew_lease(self.deps.pool, job_id, self.worker_id, self.settings.lease_seconds)
            except Exception:
                log.warning("lease renewal failed; trying again next beat", exc_info=True)
                continue
            if not renewed:
                log.warning("could not renew the lease")
                return


async def main() -> None:
    from research_assistant.runtime import build_deps

    settings = get_settings()
    setup_logging(settings.log_level)
    start_http_server(settings.worker_metrics_port)
    await apply_schema(settings.database_url)
    pool = await create_pool(settings.database_url)
    deps = build_deps(settings, pool)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, deps.stopping.set)
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    worker = Worker(deps, build_graph(JobNodes(deps), saver))
    log.info("worker started", extra={"worker_id": worker.worker_id})
    try:
        await worker.run_forever()
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
