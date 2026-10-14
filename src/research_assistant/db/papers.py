"""Papers and chunks: the indexed corpus. Vectors use pgvector; keywords use Postgres full-text search."""

from datetime import date

import numpy as np
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from research_assistant.funnel import Candidate
from research_assistant.text.chunking import Chunk
from research_assistant.writing import ChunkRow


def _date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _candidate(row: dict) -> Candidate:
    return Candidate(
        id=row["id"], title=row["title"], abstract=row["abstract"], embedding=row["embedding"].to_numpy(),
        year=row["year"], publication_date=row["publication_date"].isoformat() if row["publication_date"] else None,
        doi=row["doi"], venue=row["venue"], cited_by_count=row["cited_by_count"], authors=tuple(row["authors"]),
        referenced_ids=tuple(row["referenced_ids"]), arxiv_id=row["arxiv_id"],
    )


async def upsert_papers(pool: AsyncConnectionPool, cands: list[Candidate]) -> None:
    if not cands:
        return
    rows = [(c.id, c.doi, c.title, c.abstract, c.year, _date(c.publication_date), c.venue, c.cited_by_count,
             Jsonb(list(c.authors)), list(c.referenced_ids), c.arxiv_id, np.asarray(c.embedding, dtype=np.float32))
            for c in cands]
    async with pool.connection() as conn:
        await conn.cursor().executemany(
            """INSERT INTO papers (id, doi, title, abstract, year, publication_date, venue, cited_by_count, authors,
                                   referenced_ids, arxiv_id, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET cited_by_count = EXCLUDED.cited_by_count,
                   referenced_ids = EXCLUDED.referenced_ids, arxiv_id = COALESCE(papers.arxiv_id, EXCLUDED.arxiv_id),
                   fetched_at = now()""",
            rows,
        )


async def load_candidates(pool: AsyncConnectionPool, ids: list[str]) -> list[Candidate]:
    """Papers in the same order as `ids` (unknown ids are skipped)."""
    async with pool.connection() as conn:
        rows = await (await conn.execute("SELECT * FROM papers WHERE id = ANY(%s)", (ids,))).fetchall()
    by_id = {r["id"]: _candidate(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


class PgLocalSearch:
    """Search the stored corpus before going to OpenAlex: vector top-k plus keyword top-k."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def search(self, facet_vec: np.ndarray, keywords: str, limit: int,
                     published_before: str | None, exclude: set[str]) -> list[Candidate]:
        cutoff = _date(published_before)
        async with self.pool.connection() as conn:
            # Iterative scans keep the HNSW index returning rows when a filter removes many of them.
            await conn.execute("SET hnsw.iterative_scan = relaxed_order")
            vector_rows = await (await conn.execute(
                """SELECT * FROM papers WHERE (%(cutoff)s::date IS NULL OR publication_date <= %(cutoff)s)
                   AND NOT (id = ANY(%(exclude)s)) ORDER BY embedding <=> %(vec)s LIMIT %(limit)s""",
                {"cutoff": cutoff, "exclude": list(exclude), "vec": np.asarray(facet_vec, dtype=np.float32), "limit": limit},
            )).fetchall()
            keyword_rows = await (await conn.execute(
                """SELECT p.* FROM papers p, websearch_to_tsquery('english', %(q)s) q
                   WHERE p.tsv @@ q AND (%(cutoff)s::date IS NULL OR p.publication_date <= %(cutoff)s)
                   AND NOT (p.id = ANY(%(exclude)s)) ORDER BY ts_rank_cd(p.tsv, q) DESC LIMIT %(limit)s""",
                {"q": keywords.replace(" ", " OR "), "cutoff": cutoff, "exclude": list(exclude), "limit": limit},
            )).fetchall()
        seen, out = set(), []
        for row in vector_rows + keyword_rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                out.append(_candidate(row))
        return out


async def has_chunks(pool: AsyncConnectionPool, paper_id: str) -> bool:
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT EXISTS (SELECT 1 FROM chunks WHERE paper_id = %s) AS found", (paper_id,))).fetchone()
    return row["found"]


async def replace_chunks(pool: AsyncConnectionPool, paper_id: str, chunks: list[Chunk], vectors: np.ndarray, source: str) -> None:
    """Swap in a paper's chunks in one transaction, so a crash never leaves half a paper."""
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))
        await conn.cursor().executemany(
            "INSERT INTO chunks (paper_id, ord, section, text, tokens, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
            [(paper_id, c.ord, c.section, c.text, c.tokens, np.asarray(v, dtype=np.float32)) for c, v in zip(chunks, vectors)],
        )
        await conn.execute("UPDATE papers SET fulltext = %s WHERE id = %s", (source, paper_id))


def _chunk(row: dict) -> ChunkRow:
    return ChunkRow(id=row["id"], paper_id=row["paper_id"], ord=row["ord"], section=row["section"], text=row["text"],
                    tokens=row["tokens"], embedding=row["embedding"].to_numpy())


async def load_chunks(pool: AsyncConnectionPool, paper_ids: list[str]) -> list[ChunkRow]:
    async with pool.connection() as conn:
        rows = await (await conn.execute("SELECT * FROM chunks WHERE paper_id = ANY(%s) ORDER BY paper_id, ord", (paper_ids,))).fetchall()
    return [_chunk(r) for r in rows]


async def load_chunks_by_id(pool: AsyncConnectionPool, chunk_ids: list[int]) -> list[ChunkRow]:
    async with pool.connection() as conn:
        rows = await (await conn.execute("SELECT * FROM chunks WHERE id = ANY(%s)", (chunk_ids,))).fetchall()
    by_id = {r["id"]: _chunk(r) for r in rows}
    return [by_id[i] for i in chunk_ids if i in by_id]


async def fulltext_sources(pool: AsyncConnectionPool, paper_ids: list[str]) -> dict[str, int]:
    async with pool.connection() as conn:
        rows = await (await conn.execute(
            "SELECT coalesce(fulltext, 'missing') AS source, count(*) AS n FROM papers WHERE id = ANY(%s) GROUP BY 1", (paper_ids,),
        )).fetchall()
    return {r["source"]: r["n"] for r in rows}
