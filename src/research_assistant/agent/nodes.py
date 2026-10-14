"""Graph nodes: thin I/O wrappers. Each loads what it needs from Postgres, calls the pure logic in
funnel.py or writing.py, and saves the result. Every node is safe to run twice."""

import asyncio
import functools
import logging
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from langgraph.types import Send, interrupt
from psycopg_pool import AsyncConnectionPool

from research_assistant import funnel, writing
from research_assistant.brain.base import Brain, CardView, ChunkView, Flag, Frame, Plan, SectionSpec
from research_assistant.citations import paper_label
from research_assistant.config import JobSettings, Settings
from research_assistant.db import jobs, papers, results
from research_assistant.embed import Embedder
from research_assistant.observability import NODE_SECONDS, JobContext, job_context
from research_assistant.ranking.cluster import merge_small, spectral_clusters
from research_assistant.sources.arxiv import Arxiv
from research_assistant.sources.http import HttpError
from research_assistant.sources.openalex import OpenAlex
from research_assistant.text.chunking import chunk_paper
from research_assistant.text.pdf import PdfError, clean_text, extract_text

log = logging.getLogger(__name__)


class JobCanceled(Exception):
    pass


class LeaseLost(Exception):
    pass


class WorkerStopping(Exception):
    pass


class BudgetExceeded(Exception):
    def __init__(self, job_id: str, spent: float, budget: float):
        super().__init__(f"job {job_id} spent ${spent:.2f}, over its ${budget:.2f} budget")


@dataclass
class Deps:
    pool: AsyncConnectionPool
    brains: dict[str, Brain]
    embedder: Embedder
    openalex: OpenAlex
    arxiv: Arxiv
    settings: Settings
    calibration: dict | None = None
    stopping: asyncio.Event = field(default_factory=asyncio.Event)


def step(name: str):
    """Wrap a node: stop if the job was canceled, the lease was lost, or the worker is shutting down;
    set the job context for logs and cost tracking; record start and end events and the duration."""

    def wrap(fn):
        @functools.wraps(fn)
        async def run(self: "JobNodes", state: dict) -> dict:
            job_id = state["job_id"]
            outer = job_context.get()
            if self.deps.stopping.is_set():
                raise WorkerStopping(job_id)
            status = await jobs.run_status(self.deps.pool, job_id, outer.worker_id)
            if status == "canceled":
                raise JobCanceled(job_id)
            if status == "lost":
                raise LeaseLost(job_id)
            settings = JobSettings(**state["settings"])
            token = job_context.set(JobContext(job_id=job_id, node=name, worker_id=outer.worker_id,
                                               budget_usd=settings.budget_usd or self.deps.settings.job_budget_usd))
            start = time.monotonic()
            await jobs.add_event(self.deps.pool, job_id, name, "start")
            try:
                out = await fn(self, state)
            finally:
                job_context.reset(token)
            seconds = time.monotonic() - start
            NODE_SECONDS.labels(name).observe(seconds)
            counts = out.pop("_counts", {}) if isinstance(out, dict) else {}
            await jobs.add_event(self.deps.pool, job_id, name, "end", {"seconds": round(seconds, 3), **counts})
            return out

        return run

    return wrap


async def record_usage(pool: AsyncConnectionPool, node: str, usage) -> None:
    """ClaudeBrain's usage callback: bill the job in context and stop it once it passes its budget."""
    ctx = job_context.get()
    if ctx.job_id is None:
        return
    spent = await jobs.add_llm_call(pool, ctx.job_id, node, usage)
    if ctx.budget_usd is not None and spent > ctx.budget_usd:
        raise BudgetExceeded(ctx.job_id, spent, ctx.budget_usd)


class JobNodes:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.algo = deps.settings.algo

    def brain(self, state: dict) -> Brain:
        return self.deps.brains[state["settings"].get("brain") or self.deps.settings.brain]

    def job(self, state: dict) -> JobSettings:
        return JobSettings(**state["settings"])

    async def facet_vectors(self, plan: Plan) -> np.ndarray:
        return await asyncio.to_thread(self.deps.embedder.embed_queries, [f.question for f in plan.facets])

    # ---- level 1: pick the papers ----

    @step("plan")
    async def plan(self, state: dict) -> dict:
        plan = await self.brain(state).plan(self.job(state).topic)
        return {"plan": plan.model_dump(), "_counts": {"facets": len(plan.facets)}}

    @step("gather")
    async def gather(self, state: dict) -> dict:
        job = self.job(state)
        pool, new = await funnel.gather(
            Plan(**state["plan"]), openalex=self.deps.openalex, embedder=self.deps.embedder, algo=self.algo,
            local=papers.PgLocalSearch(self.deps.pool), year_from=job.year_from, year_to=job.year_to,
            published_before=job.published_before, exclude_ids=job.exclude_ids,
        )
        must = await self.resolve(job.must_include)
        have = {c.id for c in pool}
        pool += [c for c in must if c.id not in have]
        if not pool:
            raise funnel.NoPapersFound(f"no papers with abstracts were found for: {job.topic}")
        await papers.upsert_papers(self.deps.pool, new + must)
        return {"candidate_ids": [c.id for c in pool], "must_ids": [c.id for c in must],
                "_counts": {"pool": len(pool), "new": len(new), "must_include": len(must)}}

    async def resolve(self, refs: list[str]) -> list[funnel.Candidate]:
        """Turn OpenAlex ids or DOIs into candidates, fetching from OpenAlex when needed."""
        works = []
        ids = [r for r in refs if r.upper().startswith("W")]
        if ids:
            works += await self.deps.openalex.by_ids([i.upper() for i in ids])
        for doi in (r for r in refs if not r.upper().startswith("W")):
            work = await self.deps.openalex.by_doi(doi.lower().removeprefix("https://doi.org/"))
            if work is not None:
                works.append(work)
        return await funnel.embed_works(works, self.deps.embedder)

    @step("rank")
    async def rank(self, state: dict) -> dict:
        plan = Plan(**state["plan"])
        pool = await papers.load_candidates(self.deps.pool, state["candidate_ids"])
        signals = await asyncio.to_thread(funnel.compute_signals, pool, plan, await self.facet_vectors(plan), self.algo)
        short = funnel.shortlist(signals, self.job(state).shortlist_size)
        rows = []
        for i, c in enumerate(pool):
            rows.append({
                "paper_id": c.id,
                "signals": {"bm25": signals.bm25[i].round(4).tolist(), "dense": signals.dense[i].round(4).tolist(),
                            "facet_rel": signals.facet_rel[i].round(4).tolist(), "ppr": float(signals.ppr[i])},
                "ranks": {name: (None if np.isinf(r[i]) else int(r[i])) for name, r in signals.ranks.items()},
                "fused": float(signals.fused[i]),
                "fused_rank": int(signals.ranks["fused"][i]),
            })
        await results.save_job_papers(self.deps.pool, state["job_id"], rows)
        return {"shortlist_ids": [pool[i].id for i in short], "_counts": {"in": len(pool), "out": len(short)}}

    async def shortlisted(self, state: dict, ids: list[str]) -> tuple[list[funnel.Candidate], np.ndarray, dict]:
        cands = await papers.load_candidates(self.deps.pool, ids)
        rows = await results.job_papers(self.deps.pool, state["job_id"], [c.id for c in cands])
        facet_rel = np.array([rows[c.id]["signals"]["facet_rel"] for c in cands]) if cands else np.zeros((0, 1))
        return cands, facet_rel, rows

    @step("screen")
    async def screen(self, state: dict) -> dict:
        cands, facet_rel, _ = await self.shortlisted(state, state["shortlist_ids"])
        verdicts, post = await funnel.screen(self.brain(state), Plan(**state["plan"]), cands, facet_rel, self.algo, self.deps.calibration)
        await results.save_screening(self.deps.pool, state["job_id"], [
            (c.id, verdicts[c.id].model_dump() if c.id in verdicts else None, float(p)) for c, p in zip(cands, post)
        ])
        included = sum(1 for v in verdicts.values() if v.include)
        return {"_counts": {"judged": len(verdicts), "included": included}}

    @step("select")
    async def select(self, state: dict) -> dict:
        ids = state["shortlist_ids"] + [m for m in state.get("must_ids", []) if m not in state["shortlist_ids"]]
        cands, facet_rel, rows = await self.shortlisted(state, ids)
        post = np.array([rows[c.id]["posterior"] if rows[c.id]["posterior"] is not None else 1.0 for c in cands])
        must = [i for i, c in enumerate(cands) if c.id in set(state.get("must_ids", []))]
        job = self.job(state)
        chosen = funnel.select(job.selection_method, cands, post, facet_rel, job.max_papers, self.algo, must)
        selected = [cands[i].id for i in chosen.indices]
        await results.mark_selected(self.deps.pool, state["job_id"], selected)
        return {"selected_ids": selected, "_counts": {"selected": len(selected), "objective": chosen.objective}}

    @step("approve")
    async def approve(self, state: dict) -> dict:
        if self.job(state).auto_approve:
            return {}
        chosen = await papers.load_candidates(self.deps.pool, state["selected_ids"])
        answer = interrupt({"message": "Review the selected papers, then approve (optionally removing or adding some).",
                            "selected": [{"id": c.id, "title": c.title, "year": c.year} for c in chosen]})
        removed = set(answer.get("remove") or [])
        ids = [i for i in state["selected_ids"] if i not in removed]
        added = await self.resolve(answer.get("add") or [])
        await papers.upsert_papers(self.deps.pool, added)
        ids += [c.id for c in added if c.id not in ids]
        await results.mark_selected(self.deps.pool, state["job_id"], ids)
        return {"selected_ids": ids, "_counts": {"removed": len(removed), "added": len(added)}}

    # ---- level 2: read and write ----

    @step("ingest")
    async def ingest(self, state: dict) -> dict:
        """Full text from arXiv, one paper at a time (arXiv allows 1 request per 3 seconds).
        Papers already chunked are skipped, so a re-run after a crash only does the rest."""
        stats: Counter[str] = Counter()
        for c in await papers.load_candidates(self.deps.pool, state["selected_ids"]):
            if await papers.has_chunks(self.deps.pool, c.id):
                stats["already_ingested"] += 1
                continue
            text, source = None, "abstract"
            try:
                arxiv_id = c.arxiv_id or await self.deps.arxiv.find_by_title(c.title)
                if arxiv_id:
                    pdf = await self.deps.arxiv.fetch_pdf(arxiv_id, self.algo.pdf_max_bytes)
                    extracted = clean_text(await asyncio.to_thread(extract_text, pdf))
                    if len(extracted) >= self.algo.min_fulltext_chars:
                        text, source = extracted, "pdf"
            except (HttpError, PdfError) as exc:
                log.warning("full text unavailable for %s: %s", c.id, exc)
            chunks = chunk_paper(c.abstract, text, self.algo.chunk_tokens, self.algo.chunk_overlap)
            vectors = await asyncio.to_thread(self.deps.embedder.embed_docs, [ch.text for ch in chunks])
            await papers.replace_chunks(self.deps.pool, c.id, chunks, vectors, source)
            stats[source] += 1
        return {"_counts": dict(stats)}

    @step("outline")
    async def outline(self, state: dict) -> dict:
        chosen = await papers.load_candidates(self.deps.pool, state["selected_ids"])
        E = np.stack([c.embedding for c in chosen]) if chosen else np.zeros((0, self.deps.embedder.dim))
        labels, k, eigvals = await asyncio.to_thread(
            spectral_clusters, E, [set(c.referenced_ids) for c in chosen], self.algo.cluster_beta, self.algo.cluster_knn)
        if len(chosen) > 1:
            labels = merge_small(labels, E)
        groups = [[chosen[i] for i in np.where(labels == g)[0]] for g in range(int(labels.max()) + 1)] if len(chosen) else []
        specs = await self.brain(state).name_sections([[c.view() for c in g] for g in groups]) if groups else []
        sections = []
        for i, group in enumerate(groups):
            spec = specs[i] if i < len(specs) else SectionSpec(title=f"Theme {i + 1}", goal="Summarize this group of papers")
            sections.append({"idx": i, "title": spec.title, "goal": spec.goal, "paper_ids": [c.id for c in group]})
        await results.save_sections(self.deps.pool, state["job_id"], sections)
        return {"sections": sections, "_counts": {"sections": len(sections), "eigengap_k": k,
                                                  "eigenvalues": np.round(np.sort(eigvals)[:10], 4).tolist()}}

    def fan_out(self, state: dict) -> list[Send]:
        return [Send("section", {"job_id": state["job_id"], "settings": state["settings"], "idx": s["idx"],
                                 "selected_ids": state["selected_ids"], "round": 0})
                for s in state["sections"]]

    async def card_views(self, job_id: str, section_idx: int | None = None) -> tuple[dict[str, CardView], dict[str, str]]:
        rows = await results.evidence(self.deps.pool, job_id, section_idx)
        views = {r["card_id"]: CardView(card_id=r["card_id"], paper_label=paper_label(r["authors"], r["year"]),
                                        claim=r["claim"], quote=r["quote"]) for r in rows}
        return views, {r["card_id"]: r["paper_id"] for r in rows}

    @step("retrieve")
    async def retrieve(self, state: dict) -> dict:
        sec = await results.get_section(self.deps.pool, state["job_id"], state["idx"])
        chunks = await papers.load_chunks(self.deps.pool, state["selected_ids"])
        query = f"{sec['title']}: {sec['goal']}"
        query_vec = (await asyncio.to_thread(self.deps.embedder.embed_queries, [query]))[0]
        ranked = writing.rank_chunks(chunks, query, query_vec, set(sec["paper_ids"]), self.algo.rrf_k, self.algo.section_chunks)
        return {"chunk_ids": [c.id for c, _ in ranked], "chunk_scores": [s for _, s in ranked],
                "_counts": {"chunks_considered": len(chunks), "kept": len(ranked)}}

    @step("compress")
    async def compress(self, state: dict) -> dict:
        sec = await results.get_section(self.deps.pool, state["job_id"], state["idx"])
        chunks = await papers.load_chunks_by_id(self.deps.pool, state["chunk_ids"])
        score = dict(zip(state["chunk_ids"], state["chunk_scores"]))
        picked = writing.pick_within_budget([(c, score[c.id]) for c in chunks], self.algo.section_budget_tokens)
        spec = SectionSpec(title=sec["title"], goal=sec["goal"])
        proposed = await self.brain(state).compress(spec, [ChunkView(chunk_id=c.id, paper_id=c.paper_id, text=c.text) for c in picked])
        accepted = writing.accept_cards(proposed, {c.id: c for c in picked}, state["idx"])
        await results.replace_evidence(self.deps.pool, state["job_id"], state["idx"], [
            {"card_id": cid, "paper_id": ch.paper_id, "chunk_id": ch.id, "claim": card.claim, "quote": card.quote}
            for cid, card, ch in accepted
        ])
        return {"_counts": {"chunks": len(picked), "cards_proposed": len(proposed), "cards_kept": len(accepted)}}

    @step("write")
    async def write(self, state: dict) -> dict:
        sec = await results.get_section(self.deps.pool, state["job_id"], state["idx"])
        cards, _ = await self.card_views(state["job_id"], state["idx"])
        spec = SectionSpec(title=sec["title"], goal=sec["goal"])
        text = await self.brain(state).write_section(spec, list(cards.values())) if cards else ""
        await results.update_section(self.deps.pool, state["job_id"], state["idx"], draft=text)
        return {"round": 0, "_counts": {"cards": len(cards), "words": len(text.split())}}

    @step("verify")
    async def verify(self, state: dict) -> dict:
        sec = await results.get_section(self.deps.pool, state["job_id"], state["idx"])
        cards, _ = await self.card_views(state["job_id"], state["idx"])
        v = await writing.verify(self.brain(state), sec["draft"] or "", cards, self.algo)
        await results.update_section(self.deps.pool, state["job_id"], state["idx"],
                                     verification={"round": state.get("round", 0), "flags": {str(k): p for k, p in v.flags.items()}})
        hard = v.hard_flags()
        return {"flags": [f.model_dump() for f in hard], "_counts": {"sentences": len(v.sentences), "flagged": len(hard)}}

    def after_verify(self, state: dict) -> str:
        return "revise" if state.get("flags") and state.get("round", 0) < self.algo.revise_rounds else "finish"

    @step("revise")
    async def revise(self, state: dict) -> dict:
        sec = await results.get_section(self.deps.pool, state["job_id"], state["idx"])
        cards, _ = await self.card_views(state["job_id"], state["idx"])
        text = await self.brain(state).revise(SectionSpec(title=sec["title"], goal=sec["goal"]), sec["draft"] or "",
                                              [Flag(**f) for f in state["flags"]], list(cards.values()))
        await results.update_section(self.deps.pool, state["job_id"], state["idx"], draft=text)
        return {"round": state.get("round", 0) + 1}

    @step("finish")
    async def finish(self, state: dict) -> dict:
        """Apply the last verification: delete sentences that still fail; drop the section if too little is left."""
        sec = await results.get_section(self.deps.pool, state["job_id"], state["idx"])
        flags = {int(k): p for k, p in (sec["verification"] or {}).get("flags", {}).items()}
        final, counts = writing.finalize(writing.Verification(writing.parse_text(sec["draft"] or ""), flags))
        status = "done" if writing.kept_sentences(counts) >= self.algo.min_section_sentences else "dropped"
        await results.update_section(self.deps.pool, state["job_id"], state["idx"], final=final, status=status,
                                     verification={**(sec["verification"] or {}), "counts": counts})
        return {"done_sections": [state["idx"]], "_counts": {"status": status, **counts}}

    @step("assemble")
    async def assemble(self, state: dict) -> dict:
        job_id, brain = state["job_id"], self.brain(state)
        done = [s for s in await results.sections(self.deps.pool, job_id) if s["status"] == "done"]
        cards, card_paper = await self.card_views(job_id)
        pairs = [(SectionSpec(title=s["title"], goal=s["goal"]), s["final"]) for s in done]
        frame = await brain.write_frame(self.job(state).topic, pairs, list(cards.values()))
        fixed = {}
        for part in ("abstract", "introduction", "conclusion"):
            text = getattr(frame, part)
            v = await writing.verify(brain, text, cards, self.algo)
            if v.hard_flags():
                text = await brain.revise(SectionSpec(title=part.capitalize(), goal=f"The review's {part}"), text, v.hard_flags(), list(cards.values()))
                v = await writing.verify(brain, text, cards, self.algo)
            fixed[part], _ = writing.finalize(v)
        frame = Frame(title=frame.title, **fixed)
        paper_rows = await results.paper_rows(self.deps.pool, sorted(set(card_paper.values())))
        paper_md, bib, cited = writing.assemble_paper(frame, pairs, card_paper, paper_rows)
        report = await self.build_report(state, done)
        await results.save_outputs(self.deps.pool, job_id, paper_md, bib, report, writing.report_markdown(report))
        return {"_counts": {"sections": len(done), "references": len(cited)}}

    async def build_report(self, state: dict, done: list[dict]) -> dict:
        job_id = state["job_id"]
        table = await results.selection_table(self.deps.pool, job_id)
        selected = sorted((r for r in table if r["selected"]), key=lambda r: r["selection_order"])
        steps = []
        cost = await jobs.cost_by_node(self.deps.pool, job_id)
        objective, k = None, None
        for e in await jobs.events(self.deps.pool, job_id):
            if e["event"] == "end":
                steps.append({"node": e["node"], "seconds": e["data"].get("seconds"), "cost_usd": cost.get(e["node"], 0.0)})
                objective = e["data"].get("objective", objective) if e["node"] == "select" else objective
                k = e["data"].get("eigengap_k", k) if e["node"] == "outline" else k
        all_sections = await results.sections(self.deps.pool, job_id)
        return {
            "topic": self.job(state).topic,
            "settings": state["settings"],
            "facets": [f["question"] for f in state["plan"]["facets"]],
            "funnel": {"pool": len(state["candidate_ids"]), "shortlist": len(state["shortlist_ids"]), "selected": len(state["selected_ids"]),
                       "sections_written": len(done)},
            "selection": {"method": self.job(state).selection_method, "objective": objective},
            "selected": [{"id": r["paper_id"], "title": r["title"], "rank": r["fused_rank"], "posterior": r["posterior"] or 0.0} for r in selected],
            "outline": {"k": k},
            "sections": [{"title": s["title"], "status": s["status"], "counts": (s["verification"] or {}).get("counts", {})} for s in all_sections],
            "ingest": await papers.fulltext_sources(self.deps.pool, state["selected_ids"]),
            "steps": steps,
        }
