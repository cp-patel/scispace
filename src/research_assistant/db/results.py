"""Per-job results: scores for every candidate, sections, evidence cards, and final outputs."""

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


async def save_job_papers(pool: AsyncConnectionPool, job_id: str, rows: list[dict]) -> None:
    """rows: paper_id, signals, ranks, fused, fused_rank. Re-running replaces earlier scores."""
    async with pool.connection() as conn:
        await conn.cursor().executemany(
            """INSERT INTO job_papers (job_id, paper_id, signals, ranks, fused, fused_rank)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (job_id, paper_id) DO UPDATE SET signals = EXCLUDED.signals, ranks = EXCLUDED.ranks,
                   fused = EXCLUDED.fused, fused_rank = EXCLUDED.fused_rank""",
            [(job_id, r["paper_id"], Jsonb(r["signals"]), Jsonb(r["ranks"]), r["fused"], r["fused_rank"]) for r in rows],
        )


async def job_papers(pool: AsyncConnectionPool, job_id: str, paper_ids: list[str]) -> dict[str, dict]:
    async with pool.connection() as conn:
        rows = await (await conn.execute(
            "SELECT * FROM job_papers WHERE job_id = %s AND paper_id = ANY(%s)", (job_id, paper_ids),
        )).fetchall()
    return {r["paper_id"]: r for r in rows}


async def save_screening(pool: AsyncConnectionPool, job_id: str, rows: list[tuple[str, dict | None, float]]) -> None:
    """rows: (paper_id, verdict or None, posterior)."""
    async with pool.connection() as conn:
        await conn.cursor().executemany(
            "UPDATE job_papers SET verdict = %s, posterior = %s WHERE job_id = %s AND paper_id = %s",
            [(Jsonb(v) if v is not None else None, p, job_id, pid) for pid, v, p in rows],
        )


async def mark_selected(pool: AsyncConnectionPool, job_id: str, ordered_ids: list[str]) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("UPDATE job_papers SET selected = false, selection_order = NULL WHERE job_id = %s", (job_id,))
        await conn.cursor().executemany(
            "UPDATE job_papers SET selected = true, selection_order = %s WHERE job_id = %s AND paper_id = %s",
            [(i, job_id, pid) for i, pid in enumerate(ordered_ids, start=1)],
        )


async def selection_table(pool: AsyncConnectionPool, job_id: str, limit: int = 300) -> list[dict]:
    """The shortlist with every score, best fused rank first, joined with titles."""
    async with pool.connection() as conn:
        return await (await conn.execute(
            """SELECT jp.paper_id, p.title, p.year, jp.fused, jp.fused_rank, jp.ranks, jp.signals, jp.verdict,
                      jp.posterior, jp.selected, jp.selection_order
               FROM job_papers jp JOIN papers p ON p.id = jp.paper_id
               WHERE jp.job_id = %s ORDER BY jp.fused_rank LIMIT %s""",
            (job_id, limit),
        )).fetchall()


async def save_sections(pool: AsyncConnectionPool, job_id: str, sections: list[dict]) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM evidence WHERE job_id = %s", (job_id,))
        await conn.execute("DELETE FROM sections WHERE job_id = %s", (job_id,))
        await conn.cursor().executemany(
            "INSERT INTO sections (job_id, idx, title, goal, paper_ids) VALUES (%s, %s, %s, %s, %s)",
            [(job_id, s["idx"], s["title"], s["goal"], s["paper_ids"]) for s in sections],
        )


async def get_section(pool: AsyncConnectionPool, job_id: str, idx: int) -> dict:
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT * FROM sections WHERE job_id = %s AND idx = %s", (job_id, idx))).fetchone()


async def sections(pool: AsyncConnectionPool, job_id: str) -> list[dict]:
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT * FROM sections WHERE job_id = %s ORDER BY idx", (job_id,))).fetchall()


async def update_section(pool: AsyncConnectionPool, job_id: str, idx: int, **fields) -> None:
    allowed = {"draft", "final", "status", "verification"}
    if not fields or set(fields) - allowed:
        raise ValueError(f"can only update {sorted(allowed)}")
    values = [Jsonb(v) if k == "verification" else v for k, v in fields.items()]
    assignments = ", ".join(f"{k} = %s" for k in fields)
    async with pool.connection() as conn:
        await conn.execute(f"UPDATE sections SET {assignments} WHERE job_id = %s AND idx = %s", (*values, job_id, idx))


async def replace_evidence(pool: AsyncConnectionPool, job_id: str, section_idx: int, cards: list[dict]) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM evidence WHERE job_id = %s AND section_idx = %s", (job_id, section_idx))
        await conn.cursor().executemany(
            "INSERT INTO evidence (job_id, card_id, section_idx, paper_id, chunk_id, claim, quote) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(job_id, c["card_id"], section_idx, c["paper_id"], c["chunk_id"], c["claim"], c["quote"]) for c in cards],
        )


async def evidence(pool: AsyncConnectionPool, job_id: str, section_idx: int | None = None) -> list[dict]:
    async with pool.connection() as conn:
        return await (await conn.execute(
            """SELECT e.*, p.authors, p.year, p.title FROM evidence e JOIN papers p ON p.id = e.paper_id
               WHERE e.job_id = %s AND (%s::int IS NULL OR e.section_idx = %s) ORDER BY e.section_idx, e.card_id""",
            (job_id, section_idx, section_idx),
        )).fetchall()


async def paper_rows(pool: AsyncConnectionPool, paper_ids: list[str]) -> dict[str, dict]:
    async with pool.connection() as conn:
        rows = await (await conn.execute(
            "SELECT id, title, authors, year, venue, doi FROM papers WHERE id = ANY(%s)", (paper_ids,),
        )).fetchall()
    return {r["id"]: r for r in rows}


async def save_outputs(pool: AsyncConnectionPool, job_id: str, paper_md: str, references_bib: str, report: dict, report_md: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO outputs (job_id, paper_md, references_bib, report, report_md) VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (job_id) DO UPDATE SET paper_md = EXCLUDED.paper_md, references_bib = EXCLUDED.references_bib,
                   report = EXCLUDED.report, report_md = EXCLUDED.report_md, created_at = now()""",
            (job_id, paper_md, references_bib, Jsonb(report), report_md),
        )


async def outputs(pool: AsyncConnectionPool, job_id: str) -> dict | None:
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT * FROM outputs WHERE job_id = %s", (job_id,))).fetchone()
