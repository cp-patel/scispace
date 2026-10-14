"""Shared fixtures. Database tests use a separate database, ra_test, on the docker compose Postgres,
so running tests never touches development data."""

import os

import psycopg
import pytest
import pytest_asyncio
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from research_assistant.db import apply_schema, create_pool

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://ra:ra@localhost:55432/ra_test")
APP_TABLES = "job_events, llm_calls, outputs, evidence, sections, job_papers, jobs, chunks, papers"
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


def ensure_test_database() -> None:
    params = conninfo_to_dict(TEST_DB)
    admin = make_conninfo(TEST_DB, dbname="postgres")
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (params["dbname"],)).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{params["dbname"]}"')
    except psycopg.OperationalError as exc:
        pytest.fail(f"database tests need Postgres (run: docker compose up -d postgres): {exc}")


def reset_tables() -> None:
    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        existing = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        tables = APP_TABLES + "".join(f", {t}" for t in CHECKPOINT_TABLES if t in existing)
        conn.execute(f"TRUNCATE {tables} CASCADE")


@pytest_asyncio.fixture
async def db():
    """A fresh, empty database for one test, with an open connection pool."""
    ensure_test_database()
    await apply_schema(TEST_DB)
    reset_tables()
    pool = await create_pool(TEST_DB, max_size=8)
    yield pool
    await pool.close()
