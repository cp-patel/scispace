import numpy as np
import pytest

from research_assistant.db import papers
from research_assistant.embed import HashEmbedder
from research_assistant.funnel import Candidate
from research_assistant.text.chunking import chunk_paper

pytestmark = pytest.mark.db


EMB = HashEmbedder()


def cand(i: int, title: str, year: int = 2020, date: str | None = None) -> Candidate:
    return Candidate(id=f"W{i}", title=title, abstract=f"{title} abstract text", embedding=EMB.embed_docs([title])[0],
                     year=year, publication_date=date or f"{year}-01-01", authors=("Ana Smith",), referenced_ids=("W99",))


async def test_upsert_and_load_keep_order_and_vectors(db):
    await papers.upsert_papers(db, [cand(1, "retrieval augmented generation"), cand(2, "graph neural networks")])
    await papers.upsert_papers(db, [cand(1, "retrieval augmented generation")])  # second insert is an update
    loaded = await papers.load_candidates(db, ["W2", "W404", "W1"])
    assert [c.id for c in loaded] == ["W2", "W1"]
    assert np.allclose(loaded[1].embedding, EMB.embed_docs(["retrieval augmented generation"])[0], atol=1e-6)
    assert loaded[1].authors == ("Ana Smith",) and loaded[1].publication_date == "2020-01-01"


async def test_local_search_combines_vector_and_keyword_and_filters(db):
    await papers.upsert_papers(db, [cand(1, "retrieval augmented generation", 2019), cand(2, "retrieval for question answering", 2023),
                                    cand(3, "protein folding structure", 2019)])
    search = papers.PgLocalSearch(db)
    vec = EMB.embed_queries(["retrieval augmented generation"])[0]
    found = [c.id for c in await search.search(vec, "retrieval generation", 2, None, set())]
    assert found[0] == "W1" and "W2" in found
    found = [c.id for c in await search.search(vec, "retrieval generation", 5, "2020-01-01", {"W3"})]
    assert found == ["W1"]


async def test_chunks_replace_atomically(db):
    await papers.upsert_papers(db, [cand(1, "retrieval augmented generation")])
    chunks = chunk_paper("Abstract text here.", "1 Introduction\nRetrieval helps a lot.", 500, 50)
    await papers.replace_chunks(db, "W1", chunks, EMB.embed_docs([c.text for c in chunks]), "pdf")
    await papers.replace_chunks(db, "W1", chunks, EMB.embed_docs([c.text for c in chunks]), "pdf")
    rows = await papers.load_chunks(db, ["W1"])
    assert [r.ord for r in rows] == [0, 1] and await papers.has_chunks(db, "W1")
    assert await papers.fulltext_sources(db, ["W1"]) == {"pdf": 1}
    assert [c.id for c in await papers.load_chunks_by_id(db, [rows[1].id, rows[0].id])] == [rows[1].id, rows[0].id]
