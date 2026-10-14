"""Writing eval over finished jobs in the database.

Exact checks by code: every reference marker resolved, every evidence quote found in its chunk.
Verification outcomes per sentence: cited, partial, transition, and deleted (by reason).
The supported-claim rate here is whatever the job's brain judged: a word-overlap proxy for the
offline brain, an LLM judge for the Claude brain.

Run: python -m evals.writing_eval [--jobs id1,id2]   (default: every succeeded job)
"""

import argparse
import asyncio
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from research_assistant.citations import quote_in_source
from research_assistant.config import get_settings
from research_assistant.db import create_pool, papers, results

RESULTS = Path(__file__).with_name("results")


async def job_metrics(pool, job_id: str) -> dict:
    out = await results.outputs(pool, job_id)
    cards = await results.evidence(pool, job_id)
    chunks = {c.id: c for c in await papers.load_chunks_by_id(pool, [c["chunk_id"] for c in cards])}
    verified = sum(1 for c in cards if c["chunk_id"] in chunks and quote_in_source(c["quote"], chunks[c["chunk_id"]].text))
    counts: Counter[str] = Counter()
    for section in await results.sections(pool, job_id):
        counts.update((section["verification"] or {}).get("counts", {}))
    judged = counts["cited"] + counts["partial"] + counts["deleted_unsupported"]
    return {
        "job_id": job_id,
        "unresolved_markers": out["paper_md"].count("[E") if out else None,
        "cards": len(cards),
        "quotes_verified": verified / len(cards) if cards else None,
        "sentence_counts": dict(counts),
        "supported_rate": counts["cited"] / judged if judged else None,
        "partial_rate": counts["partial"] / judged if judged else None,
    }


async def run(job_ids: list[str] | None) -> None:
    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        if job_ids is None:
            async with pool.connection() as conn:
                job_ids = [str(r["id"]) for r in await (await conn.execute("SELECT id FROM jobs WHERE status = 'succeeded' ORDER BY created_at")).fetchall()]
        rows = [await job_metrics(pool, j) for j in job_ids]
    finally:
        await pool.close()
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.date.today().isoformat()
    (RESULTS / f"writing-{stamp}.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Writing eval", "",
             "| Job | Unresolved markers | Cards | Quotes verified | Supported | Partial | Sentence counts |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['job_id'][:8]} | {r['unresolved_markers']} | {r['cards']} | {r['quotes_verified']} | "
                     f"{r['supported_rate']} | {r['partial_rate']} | {r['sentence_counts']} |")
    (RESULTS / f"writing-{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", help="comma-separated job ids (default: all succeeded jobs)")
    args = parser.parse_args()
    asyncio.run(run(args.jobs.split(",") if args.jobs else None))


if __name__ == "__main__":
    main()
