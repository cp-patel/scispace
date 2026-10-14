"""arXiv client: find a paper's arXiv id by title, and download its PDF.

arXiv asks automated clients for at most one request every 3 seconds on a single connection,
so the HttpClient for arXiv is built with rate_per_second=1/3 and max_connections=1.
"""

import logging
import re
import xml.etree.ElementTree as ET

from research_assistant.sources.http import HttpClient, HttpError

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{}"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv\.)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)


def arxiv_id_from_text(text: str | None) -> str | None:
    """Pull an arXiv id out of a URL or DOI, e.g. 'https://arxiv.org/abs/2305.06983v2' -> '2305.06983'."""
    match = _ARXIV_ID.search(text or "")
    return match.group(1) if match else None


def _title_tokens(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def same_title(a: str, b: str, threshold: float = 0.9) -> bool:
    """Token Jaccard similarity of two titles is at least `threshold`."""
    ta, tb = _title_tokens(a), _title_tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold


class Arxiv:
    def __init__(self, http: HttpClient):
        self.http = http

    async def find_by_title(self, title: str) -> str | None:
        words = re.findall(r"[A-Za-z0-9]+", title)
        if not words:
            return None
        feed = await self.http.get_text(API_URL, {"search_query": f'ti:"{" ".join(words)}"', "max_results": 3})
        try:
            root = ET.fromstring(feed)
        except ET.ParseError:
            log.warning("arXiv returned a feed that is not valid XML; treating the title as not found")
            return None
        for entry in root.findall("a:entry", _ATOM):
            found = " ".join((entry.findtext("a:title", "", _ATOM)).split())
            if same_title(found, title):
                return arxiv_id_from_text(entry.findtext("a:id", "", _ATOM))
        return None

    async def fetch_pdf(self, arxiv_id: str, max_bytes: int) -> bytes:
        body, content_type = await self.http.get_bytes(PDF_URL.format(arxiv_id), max_bytes)
        if "pdf" not in content_type.lower() and not body.startswith(b"%PDF"):
            raise HttpError(None, PDF_URL.format(arxiv_id), f"expected a PDF, got {content_type or 'unknown content'}")
        return body
