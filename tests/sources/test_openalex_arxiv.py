import httpx
import pytest

from research_assistant.sources.arxiv import Arxiv, arxiv_id_from_text, same_title
from research_assistant.sources.http import HttpClient, HttpError
from research_assistant.sources.openalex import OpenAlex, parse_work, rebuild_abstract, year_filter

RAW_WORK = {
    "id": "https://openalex.org/W4389519118",
    "doi": "https://doi.org/10.18653/v1/2023.emnlp-main.495",
    "title": "Active  Retrieval Augmented Generation",
    "publication_year": 2023,
    "publication_date": "2023-01-01",
    "cited_by_count": 460,
    "abstract_inverted_index": {"Large": [0], "models": [1, 4], "hallucinate;": [2], "retrieval": [3]},
    "referenced_works": ["https://openalex.org/W1571498871", "https://openalex.org/W2154652894"],
    "locations": [
        {"source": {"display_name": "EMNLP"}, "pdf_url": "https://aclanthology.org/2023.emnlp-main.495.pdf",
         "landing_page_url": "https://doi.org/10.18653/v1/2023.emnlp-main.495"},
        {"source": {"display_name": "arXiv (Cornell University)"}, "pdf_url": None,
         "landing_page_url": "http://arxiv.org/abs/2305.06983"},
    ],
    "primary_location": {"source": {"display_name": "Conference on Empirical Methods in Natural Language Processing"}},
    "authorships": [{"author": {"display_name": "Zhengbao Jiang"}}, {"author": {"display_name": "Frank F. Xu"}}, {"author": {}}],
    "type": "conference-paper",
    "ids": {"openalex": "https://openalex.org/W4389519118"},
    "relevance_score": 812.5,
}


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2305.06983v2</id><title>Active Retrieval
  Augmented Generation</title></entry>
</feed>"""


async def no_sleep(_seconds: float) -> None:
    return None


def client(handler, **kwargs) -> HttpClient:
    return HttpClient("test", rate_per_second=1000, transport=httpx.MockTransport(handler), sleep=no_sleep, **kwargs)


def test_parse_work_and_abstract():
    w = parse_work(RAW_WORK)
    assert w.id == "W4389519118" and w.doi == "10.18653/v1/2023.emnlp-main.495"
    assert w.title == "Active Retrieval Augmented Generation"
    assert w.abstract == "Large models hallucinate; retrieval models"
    assert w.authors == ("Zhengbao Jiang", "Frank F. Xu")
    assert w.referenced_ids == ("W1571498871", "W2154652894")
    assert w.arxiv_id == "2305.06983"
    assert w.venue.startswith("Conference on Empirical")
    assert rebuild_abstract(None) == ""


def test_parse_work_tolerates_missing_fields():
    w = parse_work({"id": "https://openalex.org/W1"})
    assert w.id == "W1" and w.title == "" and w.abstract == "" and w.authors == () and w.arxiv_id is None


def test_arxiv_ids_and_titles():
    assert arxiv_id_from_text("https://arxiv.org/pdf/2305.06983v2") == "2305.06983"
    assert arxiv_id_from_text("10.48550/arXiv.2312.10997") == "2312.10997"
    assert arxiv_id_from_text("http://arxiv.org/abs/hep-th/9901001v1") == "hep-th/9901001"
    assert arxiv_id_from_text("https://aclanthology.org/2023.emnlp-main.495.pdf") is None
    assert same_title("Active Retrieval Augmented Generation", "Active retrieval-augmented generation")
    assert not same_title("Active Retrieval Augmented Generation", "Passive Retrieval")


def test_year_filter():
    assert year_filter(2020, 2024) == "publication_year:2020-2024"
    assert year_filter(2020, None) == "publication_year:>2019"
    assert year_filter(None, 2024) == "publication_year:<2025"
    assert year_filter(None, None) is None


async def test_openalex_search_pages_and_filters():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        page = int(request.url.params["page"])
        results = [dict(RAW_WORK, id=f"https://openalex.org/W{page}{i}") for i in range(100 if page == 1 else 3)]
        results.append({"id": "https://openalex.org/WX", "title": "No abstract"})  # dropped: no abstract
        return httpx.Response(200, json={"meta": {}, "results": results})

    works = await OpenAlex(client(handler)).search("rag", year_from=2020, year_to=2024, published_before="2024-06-01")
    assert len(works) == 103
    assert seen[0]["filter"] == "has_abstract:true,publication_year:2020-2024,to_publication_date:2024-06-01"
    assert seen[0]["search"] == "rag" and len(seen) == 2


async def test_openalex_by_ids_batches_100_and_by_doi_404():
    seen = []

    def handler(request):
        if "/works/doi:" in request.url.path:
            return httpx.Response(404)
        seen.append(request.url.params["filter"])
        return httpx.Response(200, json={"results": []})

    oa = OpenAlex(client(handler))
    await oa.by_ids([f"W{i}" for i in range(250)] + ["W1"])
    assert len(seen) == 3 and seen[0].startswith("openalex:W0|W1|")
    assert await oa.by_doi("10.1/none") is None


async def test_arxiv_find_by_title_and_pdf_checks():
    def handler(request):
        if request.url.host == "export.arxiv.org":
            assert request.url.params["search_query"] == 'ti:"Active Retrieval Augmented Generation"'
            return httpx.Response(200, text=ATOM)
        if request.url.path.endswith("html"):
            return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
        return httpx.Response(200, content=b"%PDF-1.4 ...", headers={"content-type": "application/pdf"})

    ax = Arxiv(client(handler))
    assert await ax.find_by_title("Active Retrieval Augmented Generation") == "2305.06983"
    assert (await ax.fetch_pdf("2305.06983", 1000)).startswith(b"%PDF")
    with pytest.raises(HttpError):
        await ax.fetch_pdf("html", 1000)


async def test_arxiv_title_search_tolerates_a_broken_feed():
    ax = Arxiv(client(lambda r: httpx.Response(200, text="<html><body>Service unavailable")))
    assert await ax.find_by_title("Active Retrieval Augmented Generation") is None
