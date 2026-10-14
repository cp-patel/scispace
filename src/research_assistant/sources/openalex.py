"""OpenAlex client: keyword search, fetch by ids, citing works, DOI lookups, and bulk samples.

Costs (2026 pricing): search calls $1 per 1,000; list/filter calls $0.10 per 1,000; single-record
lookups are free. Abstracts arrive as an inverted index (word -> positions) and are rebuilt here.
"""

import math
from dataclasses import dataclass

from research_assistant.sources.arxiv import arxiv_id_from_text
from research_assistant.sources.http import HttpClient, HttpError

BASE = "https://api.openalex.org"
SELECT = (
    "id,doi,title,publication_year,publication_date,cited_by_count,abstract_inverted_index,"
    "referenced_works,locations,primary_location,authorships,type,ids,relevance_score"
)
EXCLUDED_TYPES = frozenset({"paratext", "erratum", "editorial", "letter", "retraction", "peer-review"})
MAX_IDS_PER_CALL = 100


@dataclass(frozen=True)
class Work:
    id: str
    title: str
    abstract: str
    year: int | None = None
    publication_date: str | None = None
    doi: str | None = None
    venue: str | None = None
    cited_by_count: int = 0
    authors: tuple[str, ...] = ()
    referenced_ids: tuple[str, ...] = ()
    arxiv_id: str | None = None
    type: str | None = None
    relevance_score: float | None = None


def short_id(value: str) -> str:
    """'https://openalex.org/W123' -> 'W123'."""
    return value.rstrip("/").rsplit("/", 1)[-1]


def rebuild_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(pos, word) for word, where in index.items() for pos in where]
    return " ".join(word for _, word in sorted(positions))


def find_arxiv_id(raw: dict) -> str | None:
    texts = [raw.get("doi")]
    for location in raw.get("locations") or []:
        texts += [location.get("landing_page_url"), location.get("pdf_url")]
    texts += [str(v) for v in (raw.get("ids") or {}).values()]
    for text in texts:
        found = arxiv_id_from_text(text)
        if found:
            return found
    return None


def parse_work(raw: dict) -> Work:
    doi = raw.get("doi")
    if doi:
        doi = doi.lower().removeprefix("https://doi.org/")
    source = ((raw.get("primary_location") or {}).get("source")) or {}
    authors = tuple(
        a["author"]["display_name"] for a in raw.get("authorships") or [] if (a.get("author") or {}).get("display_name")
    )
    return Work(
        id=short_id(raw["id"]),
        title=" ".join((raw.get("title") or "").split()),
        abstract=rebuild_abstract(raw.get("abstract_inverted_index")),
        year=raw.get("publication_year"),
        publication_date=raw.get("publication_date"),
        doi=doi,
        venue=source.get("display_name"),
        cited_by_count=raw.get("cited_by_count") or 0,
        authors=authors,
        referenced_ids=tuple(short_id(r) for r in raw.get("referenced_works") or []),
        arxiv_id=find_arxiv_id(raw),
        type=raw.get("type"),
        relevance_score=raw.get("relevance_score"),
    )


def usable(work: Work) -> bool:
    """Has a title and an abstract, and is not front matter or a correction notice."""
    return bool(work.title and work.abstract) and work.type not in EXCLUDED_TYPES


def year_filter(year_from: int | None, year_to: int | None) -> str | None:
    if year_from and year_to:
        return f"publication_year:{year_from}-{year_to}"
    if year_from:
        return f"publication_year:>{year_from - 1}"
    if year_to:
        return f"publication_year:<{year_to + 1}"
    return None


def build_filter(*parts: str | None) -> str:
    return ",".join(p for p in parts if p)


class OpenAlex:
    def __init__(self, http: HttpClient):
        self.http = http

    async def search(self, query: str, *, year_from: int | None = None, year_to: int | None = None,
                     published_before: str | None = None, pages: int = 2, per_page: int = 100) -> list[Work]:
        works: list[Work] = []
        filters = build_filter("has_abstract:true", year_filter(year_from, year_to),
                               f"to_publication_date:{published_before}" if published_before else None)
        for page in range(1, pages + 1):
            data = await self.http.get_json(f"{BASE}/works", {
                "search": query, "filter": filters, "select": SELECT, "per_page": per_page, "page": page,
            })
            batch = [parse_work(r) for r in data.get("results", [])]
            works += batch
            if len(batch) < per_page:
                break
        return [w for w in works if usable(w)]

    async def by_ids(self, ids: list[str], *, published_before: str | None = None) -> list[Work]:
        unique = list(dict.fromkeys(ids))
        works: list[Work] = []
        for start in range(0, len(unique), MAX_IDS_PER_CALL):
            batch = unique[start:start + MAX_IDS_PER_CALL]
            filters = build_filter("openalex:" + "|".join(batch),
                                   f"to_publication_date:{published_before}" if published_before else None)
            data = await self.http.get_json(f"{BASE}/works", {"filter": filters, "select": SELECT, "per_page": MAX_IDS_PER_CALL})
            works += [parse_work(r) for r in data.get("results", [])]
        return [w for w in works if usable(w)]

    async def citing(self, work_id: str, *, year_from: int | None = None, year_to: int | None = None,
                     published_before: str | None = None, per_page: int = 100) -> list[Work]:
        filters = build_filter(f"cites:{work_id}", "has_abstract:true", year_filter(year_from, year_to),
                               f"to_publication_date:{published_before}" if published_before else None)
        data = await self.http.get_json(f"{BASE}/works", {
            "filter": filters, "select": SELECT, "per_page": per_page, "sort": "cited_by_count:desc",
        })
        return [w for w in (parse_work(r) for r in data.get("results", [])) if usable(w)]

    async def get_raw(self, work_id: str) -> dict:
        """One record by OpenAlex id (free). Returns the raw JSON."""
        return await self.http.get_json(f"{BASE}/works/{work_id}")

    async def by_doi(self, doi: str) -> Work | None:
        """One record by DOI (free). None when OpenAlex does not know the DOI."""
        try:
            raw = await self.http.get_json(f"{BASE}/works/doi:{doi}", {"select": SELECT})
        except HttpError as exc:
            if exc.status == 404:
                return None
            raise
        return parse_work(raw)

    async def sample(self, n: int, *, year_from: int = 2015, year_to: int = 2024) -> list[Work]:
        """About n highly cited works with abstracts, spread over the years (cursor paging per year;
        each year stays under the 10,000-result limit that applies without an API key)."""
        years = list(range(year_from, year_to + 1))
        per_year = math.ceil(n / len(years))
        works: list[Work] = []
        for year in years:
            got, cursor = 0, "*"
            while got < per_year and cursor:
                data = await self.http.get_json(f"{BASE}/works", {
                    "filter": build_filter("has_abstract:true", f"publication_year:{year}"),
                    "select": "id,title,abstract_inverted_index,publication_year,type",
                    "per_page": 100, "cursor": cursor, "sort": "cited_by_count:desc",
                })
                batch = [w for w in (parse_work(r) for r in data.get("results", [])) if usable(w)]
                works += batch[:per_year - got]
                got += len(batch)
                cursor = (data.get("meta") or {}).get("next_cursor")
                if not data.get("results"):
                    break
        return works[:n]
