"""The job queue and per-job bookkeeping (events, LLM calls, cost).

Postgres is the queue: a worker claims the oldest queued job with FOR UPDATE SKIP LOCKED, so two
workers never grab the same row, and holds a lease that it renews with a heartbeat. If a worker dies,
its lease runs out and the reaper puts the job back in the queue.
"""

import uuid

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from research_assistant.brain.base import Usage
from research_assistant.config import JobSettings

FINAL = ("succeeded", "failed", "canceled")


class StateConflict(Exception):
    """The job is not in a state that allows this action."""


async def create_job(pool: AsyncConnectionPool, settings: JobSettings) -> str:
    job_id = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO jobs (id, topic, settings, status) VALUES (%s, %s, %s, 'queued')",
            (job_id, settings.topic, Jsonb(settings.model_dump())),
        )
    return job_id


async def get_job(pool: AsyncConnectionPool, job_id: str) -> dict | None:
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))).fetchone()


async def list_jobs(pool: AsyncConnectionPool, limit: int, offset: int) -> list[dict]:
    async with pool.connection() as conn:
        return await (await conn.execute(
            "SELECT id, topic, status, created_at, updated_at, error FROM jobs ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )).fetchall()


async def claim_job(pool: AsyncConnectionPool, worker_id: str, lease_seconds: int) -> dict | None:
    async with pool.connection() as conn:
        return await (await conn.execute(
            """UPDATE jobs SET status = 'running', worker_id = %(worker)s, attempts = attempts + 1,
                   lease_until = now() + make_interval(secs => %(lease)s), updated_at = now()
               WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at
                           FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING *""",
            {"worker": worker_id, "lease": lease_seconds},
        )).fetchone()


async def renew_lease(pool: AsyncConnectionPool, job_id: str, worker_id: str, lease_seconds: int) -> bool:
    async with pool.connection() as conn:
        row = await (await conn.execute(
            """UPDATE jobs SET lease_until = now() + make_interval(secs => %s), updated_at = now()
               WHERE id = %s AND worker_id = %s AND status = 'running' RETURNING id""",
            (lease_seconds, job_id, worker_id),
        )).fetchone()
    return row is not None


async def run_status(pool: AsyncConnectionPool, job_id: str, worker_id: str | None) -> str:
    """'running' while this worker holds the job; 'canceled' if canceled; otherwise 'lost'."""
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT status, worker_id FROM jobs WHERE id = %s", (job_id,))).fetchone()
    if row is None:
        return "lost"
    if row["status"] == "canceled":
        return "canceled"
    if row["status"] == "running" and (worker_id is None or row["worker_id"] == worker_id):
        return "running"
    return "lost"


async def reap_expired(pool: AsyncConnectionPool, max_attempts: int) -> int:
    """Return expired running jobs to the queue, or fail them after max_attempts. Returns how many changed."""
    async with pool.connection() as conn:
        rows = await (await conn.execute(
            """UPDATE jobs SET
                   status = CASE WHEN attempts >= %(max)s THEN 'failed' ELSE 'queued' END,
                   error = CASE WHEN attempts >= %(max)s THEN 'worker lease expired too many times' ELSE error END,
                   worker_id = NULL, lease_until = NULL, updated_at = now()
               WHERE status = 'running' AND lease_until < now() RETURNING id""",
            {"max": max_attempts},
        )).fetchall()
    return len(rows)


async def finish_job(pool: AsyncConnectionPool, job_id: str, worker_id: str, status: str, error: str | None = None) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            """UPDATE jobs SET status = %s, error = %s, lease_until = NULL, resume = NULL, updated_at = now()
               WHERE id = %s AND worker_id = %s AND status = 'running'""",
            (status, error, job_id, worker_id),
        )


async def requeue(pool: AsyncConnectionPool, job_id: str, worker_id: str) -> None:
    """Hand a job back without counting the attempt (used on shutdown)."""
    async with pool.connection() as conn:
        await conn.execute(
            """UPDATE jobs SET status = 'queued', attempts = greatest(attempts - 1, 0), worker_id = NULL,
                   lease_until = NULL, updated_at = now() WHERE id = %s AND worker_id = %s AND status = 'running'""",
            (job_id, worker_id),
        )


async def wait_for_approval(pool: AsyncConnectionPool, job_id: str, worker_id: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            """UPDATE jobs SET status = 'waiting_approval', worker_id = NULL, lease_until = NULL, resume = NULL,
                   updated_at = now() WHERE id = %s AND worker_id = %s AND status = 'running'""",
            (job_id, worker_id),
        )


async def approve(pool: AsyncConnectionPool, job_id: str, answer: dict) -> None:
    async with pool.connection() as conn:
        row = await (await conn.execute(
            """UPDATE jobs SET status = 'queued', resume = %s, attempts = 0, updated_at = now()
               WHERE id = %s AND status = 'waiting_approval' RETURNING id""",
            (Jsonb(answer), job_id),
        )).fetchone()
    if row is None:
        raise StateConflict("the job is not waiting for approval")


async def cancel(pool: AsyncConnectionPool, job_id: str) -> None:
    async with pool.connection() as conn:
        row = await (await conn.execute(
            """UPDATE jobs SET status = 'canceled', lease_until = NULL, updated_at = now()
               WHERE id = %s AND status IN ('queued', 'running', 'waiting_approval') RETURNING id""",
            (job_id,),
        )).fetchone()
    if row is None:
        raise StateConflict("the job has already finished")


async def get_resume(pool: AsyncConnectionPool, job_id: str) -> dict | None:
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT resume FROM jobs WHERE id = %s", (job_id,))).fetchone()
    return row["resume"] if row else None


async def add_event(pool: AsyncConnectionPool, job_id: str, node: str, event: str, data: dict | None = None) -> None:
    async with pool.connection() as conn:
        await conn.execute("INSERT INTO job_events (job_id, node, event, data) VALUES (%s, %s, %s, %s)",
                           (job_id, node, event, Jsonb(data or {})))


async def events(pool: AsyncConnectionPool, job_id: str) -> list[dict]:
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT node, event, data, created_at FROM job_events WHERE job_id = %s ORDER BY id", (job_id,))).fetchall()


async def add_llm_call(pool: AsyncConnectionPool, job_id: str, node: str, usage: Usage) -> float:
    """Record one call and return the job's total LLM cost so far."""
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO llm_calls (job_id, node, model, input_tokens, output_tokens, cache_read_tokens, cost_usd, latency_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (job_id, node, usage.model, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cost_usd, usage.latency_ms),
        )
        row = await (await conn.execute("SELECT coalesce(sum(cost_usd), 0) AS total FROM llm_calls WHERE job_id = %s", (job_id,))).fetchone()
    return float(row["total"])


async def cost_by_node(pool: AsyncConnectionPool, job_id: str) -> dict[str, float]:
    async with pool.connection() as conn:
        rows = await (await conn.execute(
            "SELECT node, sum(cost_usd) AS cost FROM llm_calls WHERE job_id = %s GROUP BY node", (job_id,),
        )).fetchall()
    return {r["node"]: float(r["cost"]) for r in rows}


async def queue_depth(pool: AsyncConnectionPool) -> int:
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT count(*) AS n FROM jobs WHERE status = 'queued'")).fetchone()
    return row["n"]
