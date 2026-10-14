from research_assistant.brain.base import CardView, ChunkView, Flag, PaperView, SectionSpec, SentenceCheck
from research_assistant.brain.offline import OfflineBrain
from research_assistant.citations import quote_in_source


async def test_offline_plan_has_topic_facet_and_aspects():
    plan = await OfflineBrain().plan("retrieval augmented generation for science")
    assert plan.facets[0].question == "retrieval augmented generation for science"
    assert len(plan.facets) == 5 and plan.facets[1].keywords[-1] == "methods"


async def test_offline_screen_uses_rank_and_topic_words():
    brain = OfflineBrain()
    plan = await brain.plan("graph neural networks")
    papers = [
        PaperView(id="A", title="Graph neural networks for chemistry", abstract="x", rank=1, shortlist_size=10, facet_relevance=[0.9, 0.2]),
        PaperView(id="B", title="Graph methods", abstract="networks", rank=9, shortlist_size=10),
        PaperView(id="C", title="Protein folding", abstract="unrelated", rank=2, shortlist_size=10),
    ]
    verdicts = {v.paper_id: v for v in await brain.screen(plan, papers)}
    assert verdicts["A"].include and verdicts["A"].facets == [0]
    assert not verdicts["B"].include  # below the top 60%
    assert not verdicts["C"].include  # no topic word


async def test_offline_writing_loop_produces_verifiable_text():
    brain = OfflineBrain(max_cards=4)
    section = SectionSpec(title="Retrieval for hallucination", goal="Explain how retrieval reduces hallucination")
    chunk = ChunkView(chunk_id=7, paper_id="W1", text=(
        "Retrieval reduces hallucination in large language models by grounding answers in documents. "
        "Short one. The method retrieves passages before every generation step to reduce hallucination."))
    cards = await brain.compress(section, [chunk])
    assert cards and all(c.chunk_id == 7 and quote_in_source(c.quote, chunk.text) for c in cards)
    views = [CardView(card_id=f"E1.{i + 1}", paper_label="Jiang et al. 2023", claim=c.claim, quote=c.quote) for i, c in enumerate(cards)]
    text = await brain.write_section(section, views)
    assert text.count("[E1.") == len(views)
    labels = await brain.check_support([SentenceCheck(idx=0, sentence=text.split("\n\n")[0].split("]. ")[0] + "].", quotes=[views[0].quote])])
    assert labels[0].label == "supported"
    bad = await brain.check_support([SentenceCheck(idx=1, sentence="Quantum chemistry solves protein folding [E1.1].", quotes=[views[0].quote])])
    assert bad[0].label == "unsupported"
    trimmed = await brain.revise(section, "Keep this sentence here. Drop this one now.", [Flag(sentence="Drop this one now.", problem="unsupported")], views)
    assert trimmed == "Keep this sentence here."


async def test_offline_name_sections_and_frame():
    brain = OfflineBrain()
    clusters = [[PaperView(id="1", title="Graph neural networks", abstract="message passing graphs")],
                [PaperView(id="2", title="Protein folding", abstract="protein structure prediction")]]
    specs = await brain.name_sections(clusters)
    assert len(specs) == 2 and "graph" in specs[0].title.lower() and "protein" in specs[1].title.lower()
    frame = await brain.write_frame("graph learning", [(specs[0], "A claim here [E1.1]. Second claim [E1.2]."), (specs[1], "")], [])
    assert frame.title == "Graph learning: a literature review"
    assert "[E1.1]" in frame.abstract and "[E1.2]" in frame.conclusion
