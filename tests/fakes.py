"""A small fake world of papers, plus fake OpenAlex and arXiv servers, so whole pipelines run offline and repeatably."""

import random

import httpx

from research_assistant.sources.arxiv import arxiv_id_from_text, same_title
from research_assistant.sources.http import HttpClient
from research_assistant.sources.openalex import rebuild_abstract
from tests.pdf_helper import make_pdf

TOPICS = {
    "rag": "retrieval augmented generation passages grounding hallucination retriever reader knowledge",
    "gnn": "graph neural networks message passing nodes edges molecules representation",
    "protein": "protein folding structure residues alphafold sequence contacts domains",
}
GENERIC = "we propose method results show improves performance experiments benchmark approach data".split()
EMPTY_FEED = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _sentence(rng: random.Random, vocab: list[str]) -> str:
    words = rng.sample(vocab, 5) + rng.sample(GENERIC, 3)
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


def make_world(per_topic: int = 30, seed: int = 0) -> dict[str, dict]:
    """Raw OpenAlex-shaped records keyed by short id. Papers cite papers on their own topic.
    Even-numbered papers have an arXiv location (so they get full text); odd ones do not."""
    rng = random.Random(seed)
    world: dict[str, dict] = {}
    for t, words in enumerate(TOPICS.values(), start=1):
        vocab = words.split()
        for i in range(per_topic):
            wid = f"W{t}{i:03d}"
            abstract = " ".join(_sentence(rng, vocab) for _ in range(4))
            index: dict[str, list[int]] = {}
            for pos, word in enumerate(abstract.split()):
                index.setdefault(word, []).append(pos)
            year = 2015 + i % 10
            refs = [f"W{t}{j:03d}" for j in rng.sample(range(per_topic), 6) if j != i]
            world[wid] = {
                "id": f"https://openalex.org/{wid}",
                "doi": f"https://doi.org/10.1/{wid.lower()}",
                "title": " ".join(rng.sample(vocab, 4)).capitalize() + f" study {wid}",
                "publication_year": year,
                "publication_date": f"{year}-06-01",
                "cited_by_count": rng.randint(0, 500),
                "abstract_inverted_index": index,
                "referenced_works": [f"https://openalex.org/{r}" for r in refs],
                "authorships": [{"author": {"display_name": f"Ana Smith{t}{i}"}}, {"author": {"display_name": "Bo Lee"}}],
                "locations": [{"landing_page_url": f"http://arxiv.org/abs/2101.{t}{i:04d}"}] if i % 2 == 0 else [],
                "primary_location": {"source": {"display_name": "Journal of Tests"}},
                "type": "article",
                "_vocab": vocab,
            }
    return world


def paper_lines(record: dict, seed: int = 1) -> list[str]:
    rng = random.Random(record["id"] + str(seed))
    vocab = record["_vocab"]
    lines = [record["title"], "Abstract", rebuild_abstract(record["abstract_inverted_index"])]
    for n, head in enumerate(["Introduction", "Method", "Results"], start=1):
        lines.append(f"{n} {head}")
        lines += [_sentence(rng, vocab) for _ in range(6)]
    return lines + ["References", "[1] Some reference."]


def openalex_handler(world: dict[str, dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        params, path = request.url.params, request.url.path
        if path.startswith("/works/doi:"):
            doi = path.split("doi:", 1)[1].lower()
            match = next((r for r in world.values() if r["doi"].endswith(doi)), None)
            return httpx.Response(200, json=match) if match else httpx.Response(404, json={"error": "not found"})
        if path.startswith("/works/W"):
            wid = path.rsplit("/", 1)[1]
            return httpx.Response(200, json=world[wid]) if wid in world else httpx.Response(404)
        filters = dict(part.split(":", 1) for part in params.get("filter", "").split(",") if part)
        records = list(world.values())
        if "openalex" in filters:
            wanted = set(filters["openalex"].split("|"))
            records = [r for r in records if r["id"].rsplit("/", 1)[1] in wanted]
        if "cites" in filters:
            records = [r for r in records if f"https://openalex.org/{filters['cites']}" in r["referenced_works"]]
        if "to_publication_date" in filters:
            records = [r for r in records if r["publication_date"] <= filters["to_publication_date"]]
        if "search" in params:
            query = set(params["search"].lower().split())
            scored = [(len(query & set(f"{r['title']} {rebuild_abstract(r['abstract_inverted_index'])}".lower().split())), r) for r in records]
            records = [r for score, r in sorted(scored, key=lambda pair: -pair[0]) if score > 0]
        per_page, page = int(params.get("per_page", 25)), int(params.get("page", 1))
        cost = "0.001" if "search" in params else "0.0001"
        return httpx.Response(200, json={"meta": {"count": len(records)}, "results": records[(page - 1) * per_page: page * per_page]},
                              headers={"x-ratelimit-cost-usd": cost})

    return handler


def arxiv_handler(world: dict[str, dict]):
    by_arxiv = {arxiv_id_from_text(r["locations"][0]["landing_page_url"]): r for r in world.values() if r["locations"]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "export.arxiv.org":
            wanted = request.url.params["search_query"][4:-1]
            for aid, r in by_arxiv.items():
                if same_title(r["title"], wanted):
                    entry = f"<entry><id>http://arxiv.org/abs/{aid}v1</id><title>{r['title']}</title></entry>"
                    return httpx.Response(200, text=EMPTY_FEED.replace("</feed>", entry + "</feed>"))
            return httpx.Response(200, text=EMPTY_FEED)
        record = by_arxiv.get(request.url.path.split("/pdf/", 1)[1])
        if record is None:
            return httpx.Response(404)
        return httpx.Response(200, content=make_pdf(paper_lines(record)), headers={"content-type": "application/pdf"})

    return handler


async def _no_sleep(_seconds: float) -> None:
    return None


def fake_http(handler, service: str = "fake") -> HttpClient:
    return HttpClient(service, rate_per_second=10_000, transport=httpx.MockTransport(handler), sleep=_no_sleep)
