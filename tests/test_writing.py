import numpy as np

from research_assistant import writing
from research_assistant.brain.base import CardView, EvidenceCard, Frame, SectionSpec, SupportLabel
from research_assistant.config import AlgoParams
from research_assistant.embed import HashEmbedder

ALGO = AlgoParams()
EMB = HashEmbedder(dim=64)


def chunk(cid: int, paper: str, text: str, tokens: int = 100) -> writing.ChunkRow:
    return writing.ChunkRow(id=cid, paper_id=paper, ord=0, section="Body", text=text, tokens=tokens, embedding=EMB.embed_docs([text])[0])


class Judge:
    """Labels sentences by a keyword, so verification can be tested without any model."""

    name = "test"

    async def check_support(self, items):
        out = []
        for item in items:
            label = "unsupported" if "wrong" in item.sentence else "partial" if "maybe" in item.sentence else "supported"
            out.append(SupportLabel(idx=item.idx, label=label, reason=""))
        return out


def cards(*ids: str) -> dict[str, CardView]:
    return {i: CardView(card_id=i, paper_label="Smith 2020", claim="c", quote="q") for i in ids}


def test_rank_chunks_prefers_matching_text_and_own_papers():
    chunks = [chunk(1, "W1", "protein folding residues"), chunk(2, "W2", "retrieval reduces hallucination"),
              chunk(3, "W3", "retrieval reduces hallucination too")]
    ranked = writing.rank_chunks(chunks, "retrieval hallucination", EMB.embed_queries(["retrieval hallucination"])[0], {"W3"}, 60, 2)
    assert [c.id for c, _ in ranked] == [3, 2]
    assert writing.rank_chunks([], "q", np.zeros(64), set(), 60, 5) == []


def test_pick_within_budget_never_exceeds_budget():
    ranked = [(chunk(i, "W1", f"text number {i} about retrieval", tokens=t), 1.0 / (i + 1)) for i, t in enumerate([300, 250, 400, 100])]
    picked = writing.pick_within_budget(ranked, 600)
    assert picked and sum(c.tokens for c in picked) <= 600


def test_accept_cards_keeps_only_verbatim_quotes_and_numbers_them():
    source = chunk(7, "W1", "Retrieval reduces hallucination by 20 percent. It is slower.")
    proposed = [
        EvidenceCard(chunk_id=7, claim="a", quote="Retrieval reduces hallucination by 20 percent."),
        EvidenceCard(chunk_id=7, claim="b", quote="retrieval reduces hallucination by 20 percent"),  # duplicate
        EvidenceCard(chunk_id=7, claim="c", quote="Retrieval eliminates hallucination."),  # not in source
        EvidenceCard(chunk_id=99, claim="d", quote="It is slower."),  # unknown chunk
        EvidenceCard(chunk_id=7, claim="e", quote="It is slower."),
    ]
    kept = writing.accept_cards(proposed, {7: source}, section_idx=1)
    assert [(cid, c.claim) for cid, c, _ in kept] == [("E2.1", "a"), ("E2.2", "e")]


async def test_verify_and_finalize_delete_hard_problems_and_keep_partial():
    text = (
        "Retrieval helps grounding [E1.1]. Short transition here.\n\n"
        "This claim is wrong [E1.1]. This maybe holds [E1.2]. Unknown ids appear [E9.9]. "
        "A long uncited sentence that states a fact without any citation at all today.\n\n"
        "A paragraph with no citations."
    )
    v = await writing.verify(Judge(), text, cards("E1.1", "E1.2"), ALGO)
    assert {v.sentences[i].text: p for i, p in v.flags.items()} == {
        "This claim is wrong [E1.1].": "unsupported",
        "This maybe holds [E1.2].": "partial",
        "Unknown ids appear [E9.9].": "unknown_citation",
        "A long uncited sentence that states a fact without any citation at all today.": "uncited",
        "A paragraph with no citations.": "uncited",
    }
    assert len(v.hard_flags()) == 4
    final, counts = writing.finalize(v)
    assert final == "Retrieval helps grounding [E1.1]. Short transition here.\n\nThis maybe holds [E1.2]."
    assert counts == {"cited": 1, "transition": 1, "partial": 1, "deleted_unsupported": 1,
                      "deleted_unknown_citation": 1, "deleted_uncited": 2}
    assert writing.kept_sentences(counts) == 3


async def test_markers_after_the_period_and_malformed_markers():
    v = await writing.verify(Judge(), "Claim one. [E1.1] Claim two [E3].", cards("E1.1"), ALGO)
    assert v.sentences[0].text == "Claim one [E1.1]." and v.sentences[0].cards == ["E1.1"]
    assert v.flags.get(1) == "unknown_citation"  # [E3] looks like a citation but names no card


def test_assemble_paper_numbers_references_by_first_use():
    frame = Frame(title="A review", abstract="Abstract cites [E2.1].", introduction="Intro [E1.1, E2.1].", conclusion="End.")
    sections = [(SectionSpec(title="Theme one", goal="g"), "Body [E1.2].")]
    card_paper = {"E1.1": "W1", "E1.2": "W1", "E2.1": "W2"}
    papers = {"W1": {"id": "W1", "title": "First", "authors": ["Ana Smith"], "year": 2020},
              "W2": {"id": "W2", "title": "Second", "authors": ["Bo Lee"], "year": 2021}}
    md, bib, cited = writing.assemble_paper(frame, sections, card_paper, papers)
    assert cited == ["W2", "W1"]
    assert "Abstract cites [1]." in md and "Intro [1, 2]." in md and "## 2. Theme one" in md and "## 3. Conclusion" in md
    assert "[1] Bo Lee (2021). Second." in md and bib.count("@article") == 2


def test_report_markdown_renders():
    report = {
        "topic": "t", "funnel": {"pool": 10, "shortlist": 5, "selected": 2},
        "selection": {"method": "submodular", "objective": 1.23},
        "selected": [{"title": "P", "rank": 1, "posterior": 0.9}],
        "outline": {"k": 2}, "sections": [{"title": "S", "status": "done", "counts": {"cited": 3}}],
        "ingest": {"pdf": 1, "abstract": 1}, "steps": [{"node": "plan", "seconds": 0.1, "cost_usd": 0.0}],
    }
    md = writing.report_markdown(report)
    assert "| pool | 10 |" in md and "submodular" in md and "| plan | 0.1 | 0.0000 |" in md
