"""Level 2: passages, evidence cards, verification, and the final paper. No database access."""

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from research_assistant.brain.base import Brain, CardView, EvidenceCard, Flag, Frame, SectionSpec, SentenceCheck
from research_assistant.citations import (
    bibtex_entry,
    card_id,
    find_markers,
    has_malformed_markers,
    normalize_marker_placement,
    number_papers,
    quote_in_source,
    quote_key,
    reference_entry,
    render_markers,
    strip_markers,
)
from research_assistant.config import AlgoParams
from research_assistant.ranking.bm25 import bm25_scores
from research_assistant.ranking.fusion import ranks_from_scores, rrf
from research_assistant.ranking.select import FacilityLocation, budgeted_greedy
from research_assistant.text.tokenize import split_paragraphs, split_sentences, tokenize

HARD_PROBLEMS = ("unknown_citation", "uncited", "unsupported")


@dataclass
class ChunkRow:
    id: int
    paper_id: str
    ord: int
    section: str
    text: str
    tokens: int
    embedding: np.ndarray


@dataclass
class Sentence:
    para: int
    idx: int
    text: str
    cards: list[str]


@dataclass
class Verification:
    sentences: list[Sentence]
    flags: dict[int, str]  # sentence idx -> unknown_citation | uncited | unsupported | partial

    def hard_flags(self) -> list[Flag]:
        return [Flag(sentence=s.text, problem=self.flags[s.idx]) for s in self.sentences if self.flags.get(s.idx) in HARD_PROBLEMS]


def rank_chunks(chunks: list[ChunkRow], query: str, query_vec: np.ndarray, own_papers: set[str], k: float, limit: int) -> list[tuple[ChunkRow, float]]:
    """Hybrid passage retrieval: RRF over BM25, dense cosine, and dense restricted to the section's own
    papers (which prefers them without excluding the rest)."""
    if not chunks:
        return []
    keyword = bm25_scores([tokenize(c.text) for c in chunks], tokenize(query))
    dense = np.stack([c.embedding for c in chunks]).astype(np.float32) @ np.asarray(query_vec, dtype=np.float32)
    own = np.array([c.paper_id in own_papers for c in chunks])
    lists = [ranks_from_scores(keyword, present=keyword > 0), ranks_from_scores(dense)]
    if own.any():
        lists.append(ranks_from_scores(dense, present=own))
    fused = rrf(lists, k=k)
    order = np.argsort(-fused, kind="stable")[:limit]
    return [(chunks[i], float(fused[i])) for i in order]


def pick_within_budget(ranked: list[tuple[ChunkRow, float]], budget_tokens: int) -> list[ChunkRow]:
    """Budgeted facility location: cover the retrieved passages as well as possible within the token budget."""
    if not ranked:
        return []
    rel = np.array([score for _, score in ranked])
    rel = rel / rel.max()
    E = np.stack([c.embedding for c, _ in ranked]).astype(float)
    costs = np.array([max(c.tokens, 1) for c, _ in ranked], dtype=float)
    picked = budgeted_greedy(FacilityLocation(rel, np.maximum(E @ E.T, 0.0)), costs, budget_tokens)
    return [ranked[i][0] for i in sorted(picked)]


_GLUED = re.compile(r"[A-Za-z]{30,}")  # PDF text that lost its spaces, like 'havedevelopedaQAapplication...'


def accept_cards(cards: list[EvidenceCard], chunks: dict[int, ChunkRow], section_idx: int) -> list[tuple[str, EvidenceCard, ChunkRow]]:
    """Keep cards whose quote is in their chunk word for word and has no glued words; drop repeats; give ids E{section}.{n}."""
    kept: list[tuple[str, EvidenceCard, ChunkRow]] = []
    seen: set[str] = set()
    for card in cards:
        chunk = chunks.get(card.chunk_id)
        key = quote_key(card.quote)
        if chunk is None or key in seen or _GLUED.search(card.quote) or not quote_in_source(card.quote, chunk.text):
            continue
        seen.add(key)
        kept.append((card_id(section_idx, len(kept) + 1), card, chunk))
    return kept


def parse_text(text: str) -> list[Sentence]:
    out: list[Sentence] = []
    for p, para in enumerate(split_paragraphs(normalize_marker_placement(text))):
        for s in split_sentences(para):
            out.append(Sentence(para=p, idx=len(out), text=s, cards=find_markers(s)))
    return out


def structural_flags(sentences: list[Sentence], allowed: set[str], max_uncited_words: int) -> dict[int, str]:
    """Checks that need no judgment: every cited id must exist; long or unsupported-paragraph sentences need a citation."""
    cited_paras = {s.para for s in sentences if s.cards}
    flags: dict[int, str] = {}
    for s in sentences:
        if any(c not in allowed for c in s.cards) or has_malformed_markers(s.text):
            flags[s.idx] = "unknown_citation"
        elif not s.cards and (len(strip_markers(s.text).split()) > max_uncited_words or s.para not in cited_paras):
            flags[s.idx] = "uncited"
    return flags


async def verify(brain: Brain, text: str, cards: dict[str, CardView], algo: AlgoParams) -> Verification:
    sentences = parse_text(text)
    flags = structural_flags(sentences, set(cards), algo.uncited_max_words)
    checks = [SentenceCheck(idx=s.idx, sentence=s.text, quotes=[cards[c].quote for c in s.cards])
              for s in sentences if s.cards and s.idx not in flags]
    if checks:
        labels = {label.idx: label.label for label in await brain.check_support(checks)}
        for check in checks:
            label = labels.get(check.idx, "unsupported")  # skipped by the judge: not shown to be supported
            if label != "supported":
                flags[check.idx] = label
    return Verification(sentences, flags)


def finalize(v: Verification) -> tuple[str, dict[str, int]]:
    """Delete sentences that still have a hard problem; keep 'partial' ones but count them."""
    paragraphs: dict[int, list[str]] = {}
    counts: Counter[str] = Counter()
    for s in v.sentences:
        problem = v.flags.get(s.idx)
        if problem in HARD_PROBLEMS:
            counts[f"deleted_{problem}"] += 1
            continue
        counts["partial" if problem == "partial" else ("cited" if s.cards else "transition")] += 1
        paragraphs.setdefault(s.para, []).append(s.text)
    return "\n\n".join(" ".join(ss) for _, ss in sorted(paragraphs.items())), dict(counts)


def kept_sentences(counts: dict[str, int]) -> int:
    return counts.get("cited", 0) + counts.get("partial", 0) + counts.get("transition", 0)


def assemble_paper(frame: Frame, sections: list[tuple[SectionSpec, str]], card_paper: dict[str, str],
                   papers: dict[str, dict]) -> tuple[str, str, list[str]]:
    """Returns (paper markdown, BibTeX, cited paper ids in reference order)."""
    texts = [frame.abstract, frame.introduction] + [t for _, t in sections] + [frame.conclusion]
    numbers = number_papers(texts, card_paper)

    def render(text: str) -> str:
        return render_markers(text, card_paper, numbers)

    lines = [f"# {frame.title}", "", "## Abstract", "", render(frame.abstract), "", "## 1. Introduction", "", render(frame.introduction), ""]
    for i, (spec, text) in enumerate(sections, start=2):
        lines += [f"## {i}. {spec.title}", "", render(text), ""]
    lines += [f"## {len(sections) + 2}. Conclusion", "", render(frame.conclusion), "", "## References", ""]
    cited = [pid for pid, _ in sorted(numbers.items(), key=lambda kv: kv[1])]
    lines += [reference_entry(numbers[pid], papers[pid]) for pid in cited]
    bib = "\n\n".join(bibtex_entry(papers[pid]) for pid in cited)
    return "\n".join(lines).rstrip() + "\n", bib + "\n", cited


def report_markdown(report: dict) -> str:
    """A readable version of the job report."""
    lines = [f"# Run report: {report['topic']}", ""]
    lines += ["## Funnel", "", "| Stage | Papers |", "|---|---|"]
    lines += [f"| {stage} | {count} |" for stage, count in report["funnel"].items()]
    lines += ["", f"Selection method: **{report['selection']['method']}**, objective value: {report['selection']['objective']}", ""]
    lines += ["## Selected papers", "", "| # | Paper | Fused rank | P(relevant) |", "|---|---|---|---|"]
    lines += [f"| {i} | {p['title']} | {p['rank']} | {p['posterior']:.2f} |" for i, p in enumerate(report["selected"], start=1)]
    lines += ["", "## Outline", "", f"Sections chosen by eigengap: {report['outline']['k']}", ""]
    lines += [f"- {s['title']} ({s['status']}): {s['counts']}" for s in report["sections"]]
    lines += ["", "## Full text coverage", "", ", ".join(f"{k}: {v}" for k, v in report["ingest"].items()) or "none", ""]
    lines += ["## Cost and time per step", "", "| Step | Seconds | LLM cost (USD) |", "|---|---|---|"]
    lines += [f"| {s['node']} | {s['seconds']} | {s['cost_usd']:.4f} |" for s in report["steps"]]
    return "\n".join(lines) + "\n"
