"""OfflineBrain: rule-based stand-ins for every LLM step. Free and deterministic.

It is the default until an API key exists, the test double for the whole agent, and the baseline
an LLM must beat in evals. It writes extractive text (quotes stitched together with citations)
and can only delete sentences, never rewrite them.
"""

import math
from collections import Counter

import numpy as np

from research_assistant.brain.base import (
    CardView,
    ChunkView,
    Criteria,
    EvidenceCard,
    Facet,
    Flag,
    Frame,
    PaperView,
    Plan,
    SectionSpec,
    SentenceCheck,
    SupportLabel,
    Verdict,
)
from research_assistant.citations import find_markers, strip_markers
from research_assistant.ranking.bm25 import bm25_scores
from research_assistant.text.tokenize import normalize_for_match, split_paragraphs, split_sentences, tokenize

ASPECTS = (
    ("methods and techniques", "methods"),
    ("evaluation and benchmarks", "evaluation"),
    ("applications", "applications"),
    ("limitations and open problems", "limitations"),
)
TEMPLATE_WORDS = frozenset({"et", "al", "report", "reports", "that"})


def _strip_end(text: str) -> str:
    return text.strip().rstrip(".!?;:,").strip()


def _lower_first(text: str) -> str:
    return text if text[:2].isupper() else text[:1].lower() + text[1:]


class OfflineBrain:
    name = "offline"

    def __init__(self, max_cards: int = 8, support_threshold: float = 0.6, partial_threshold: float = 0.3):
        self.max_cards = max_cards
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold

    async def plan(self, topic: str) -> Plan:
        base = tokenize(topic)[:6] or [topic.lower()]
        facets = [Facet(id=0, question=topic, keywords=base)]
        for i, (label, word) in enumerate(ASPECTS, start=1):
            facets.append(Facet(id=i, question=f"{topic}: {label}", keywords=base + [word]))
        return Plan(facets=facets, criteria=Criteria(include=[f"Studies {topic}"], exclude=["Not about the topic"]))

    async def screen(self, plan: Plan, papers: list[PaperView]) -> list[Verdict]:
        topic_words = set(plan.facets[0].keywords)
        verdicts = []
        for p in papers:
            matched = bool(set(tokenize(f"{p.title} {p.abstract}")) & topic_words)
            in_top = p.shortlist_size == 0 or p.rank <= 0.6 * p.shortlist_size
            facets = [i for i, r in enumerate(p.facet_relevance) if r >= 0.5]
            verdicts.append(Verdict(paper_id=p.id, include=in_top and matched, facets=facets,
                                    reason=f"rank {p.rank} of {p.shortlist_size}; topic word match: {matched}"))
        return verdicts

    async def name_sections(self, clusters: list[list[PaperView]]) -> list[SectionSpec]:
        """Class-based TF-IDF: words frequent in this cluster but rare in the others."""
        docs = [[t for p in cluster for t in tokenize(f"{p.title} {p.abstract}")] for cluster in clusters]
        df = Counter(t for d in docs for t in set(d))
        specs = []
        for d in docs:
            tf = Counter(d)
            total = sum(tf.values()) or 1
            score = {t: (n / total) * math.log(1 + len(docs) / df[t]) for t, n in tf.items()}
            top = [t for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
            if top:
                specs.append(SectionSpec(title=", ".join(top).capitalize(), goal=f"Summarize work on {', '.join(top)}"))
            else:
                specs.append(SectionSpec(title="Related work", goal="Summarize this group of papers"))
        return specs

    async def compress(self, section: SectionSpec, chunks: list[ChunkView]) -> list[EvidenceCard]:
        """Pick the sentences that best match the section goal (BM25). The quote is the sentence itself."""
        seen, sentences = set(), []
        for chunk in chunks:
            for s in split_sentences(chunk.text):
                key = normalize_for_match(s)
                if 8 <= len(s.split()) <= 60 and key not in seen:
                    seen.add(key)
                    sentences.append((chunk.chunk_id, s))
        if not sentences:
            return []
        scores = bm25_scores([tokenize(s) for _, s in sentences], tokenize(f"{section.title} {section.goal}"))
        order = np.argsort(-scores, kind="stable")[: self.max_cards]
        return [EvidenceCard(chunk_id=sentences[i][0], claim=sentences[i][1], quote=sentences[i][1]) for i in order]

    async def write_section(self, section: SectionSpec, cards: list[CardView]) -> str:
        sentences = [f"{c.paper_label} report that {_lower_first(_strip_end(c.quote))} [{c.card_id}]." for c in cards]
        return "\n\n".join(" ".join(sentences[i:i + 3]) for i in range(0, len(sentences), 3))

    async def check_support(self, items: list[SentenceCheck]) -> list[SupportLabel]:
        """Share of the sentence's content words found in its quotes."""
        labels = []
        for item in items:
            words = {w for w in tokenize(strip_markers(item.sentence)) if w not in TEMPLATE_WORDS and not w.isdigit()}
            quote_words = set(tokenize(" ".join(item.quotes)))
            share = len(words & quote_words) / len(words) if words else 0.0
            if share >= self.support_threshold:
                label = "supported"
            elif share >= self.partial_threshold:
                label = "partial"
            else:
                label = "unsupported"
            labels.append(SupportLabel(idx=item.idx, label=label, reason=f"{share:.0%} of content words appear in the quotes"))
        return labels

    async def revise(self, section: SectionSpec, text: str, flags: list[Flag], cards: list[CardView]) -> str:
        """Rules cannot rewrite, so flagged sentences are deleted."""
        flagged = {normalize_for_match(f.sentence) for f in flags}
        paragraphs = []
        for para in split_paragraphs(text):
            kept = [s for s in split_sentences(para) if normalize_for_match(s) not in flagged]
            if kept:
                paragraphs.append(" ".join(kept))
        return "\n\n".join(paragraphs)

    async def write_frame(self, topic: str, sections: list[tuple[SectionSpec, str]], cards: list[CardView]) -> Frame:
        cited = [[s for s in split_sentences(" ".join(split_paragraphs(text))) if find_markers(s)] for _, text in sections]
        firsts = [c[0] for c in cited if c]
        lasts = [c[-1] for c in cited if len(c) > 1]
        title = f"{topic[:1].upper()}{topic[1:]}: a literature review"
        introduction = "This review groups the selected papers into themes. " + " ".join(firsts)
        return Frame(title=title, abstract=" ".join(firsts[:4]), introduction=introduction.strip(),
                     conclusion=" ".join(lasts) or " ".join(firsts[-2:]))
