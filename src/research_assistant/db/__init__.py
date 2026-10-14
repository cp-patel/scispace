"""Connection pool and schema. Queries are plain SQL in db/papers.py, db/jobs.py and db/results.py."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql")


async def apply_schema(url: str) -> None:
    """Create the extension and tables if missing. Runs before the pool opens, because the pool
    registers the vector type, which needs the extension to exist."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(SCHEMA.read_text())


async def create_pool(url: str, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        url, min_size=min_size, max_size=max_size, open=False, configure=register_vector_async,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    return pool
