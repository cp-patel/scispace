"""Citation markers, quote checks, reference numbering, and BibTeX.

The LLM never writes a reference. It cites evidence cards by id, like [E2.3]; code maps each card
to its paper and builds the reference list from OpenAlex metadata. A made-up reference cannot happen.
"""

import re

from research_assistant.text.tokenize import normalize_for_match

MARKER = re.compile(r"\[\s*(E\d+\.\d+(?:\s*,\s*E\d+\.\d+)*)\s*\]")
_LOOSE_MARKER = re.compile(r"\[\s*E[^\]]*\]")
_MARKER_AFTER_PERIOD = re.compile(r"([.!?])\s*(\[\s*E\d+\.\d+(?:\s*,\s*E\d+\.\d+)*\s*\])")
_BIBTEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                   "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def card_id(section_idx: int, n: int) -> str:
    """Card ids name their section (1-based) and position: E3.7 is section 3, card 7. Parallel
    sections can number cards without coordinating."""
    return f"E{section_idx + 1}.{n}"


def find_markers(text: str) -> list[str]:
    """Card ids in order of appearance."""
    return [c.strip() for group in MARKER.findall(text) for c in group.split(",")]


def has_malformed_markers(text: str) -> bool:
    """True when something looks like a card citation but is not one, such as [E3] or [E2.x]."""
    return len(_LOOSE_MARKER.findall(text)) > len(MARKER.findall(text))


def quote_key(quote: str) -> str:
    """Comparison key for quotes: normalized, without surrounding punctuation."""
    return normalize_for_match(quote).strip(" .;:,\"'")


def strip_markers(text: str) -> str:
    return " ".join(MARKER.sub("", text).split())


def normalize_marker_placement(text: str) -> str:
    """Move markers that follow the period to before it: 'claim. [E1.2]' -> 'claim [E1.2].'"""
    return _MARKER_AFTER_PERIOD.sub(lambda m: f" {m.group(2)}{m.group(1)}", text)


def quote_in_source(quote: str, source: str) -> bool:
    """True when the quote appears word for word in the source (ignoring case, spacing, and end punctuation)."""
    q = quote_key(quote)
    return len(q) > 0 and q in normalize_for_match(source)


def surname(full_name: str) -> str:
    parts = full_name.split()
    return parts[-1] if parts else "Unknown"


def paper_label(authors: list[str] | tuple[str, ...], year: int | None) -> str:
    when = str(year) if year else "n.d."
    if not authors:
        return f"Unknown {when}"
    if len(authors) == 1:
        return f"{surname(authors[0])} {when}"
    if len(authors) == 2:
        return f"{surname(authors[0])} and {surname(authors[1])} {when}"
    return f"{surname(authors[0])} et al. {when}"


def number_papers(texts: list[str], card_paper: dict[str, str]) -> dict[str, int]:
    """Number papers 1, 2, 3... by first citation across the texts. Unknown card ids are ignored."""
    numbers: dict[str, int] = {}
    for text in texts:
        for cid in find_markers(text):
            paper = card_paper.get(cid)
            if paper is not None and paper not in numbers:
                numbers[paper] = len(numbers) + 1
    return numbers


def render_markers(text: str, card_paper: dict[str, str], numbers: dict[str, int]) -> str:
    """Replace [E1.2, E3.1] with [1, 4]. Markers that point nowhere are removed."""

    def replace(match: re.Match) -> str:
        nums = sorted({numbers[card_paper[c.strip()]] for c in match.group(1).split(",")
                       if c.strip() in card_paper and card_paper[c.strip()] in numbers})
        return f"[{', '.join(map(str, nums))}]" if nums else ""

    return re.sub(r"\s+([.!?,;:])", r"\1", MARKER.sub(replace, text)).strip()


def reference_entry(n: int, paper: dict) -> str:
    authors = list(paper.get("authors") or [])
    names = ", ".join(authors[:6]) + (", et al." if len(authors) > 6 else "") if authors else "Unknown authors"
    year = paper.get("year") or "n.d."
    venue = f" {paper['venue']}." if paper.get("venue") else ""
    doi = f" https://doi.org/{paper['doi']}" if paper.get("doi") else ""
    return f"[{n}] {names} ({year}). {paper.get('title') or 'Untitled'}.{venue}{doi}"


def bibtex_escape(text: str) -> str:
    return "".join(_BIBTEX_SPECIAL.get(ch, ch) for ch in text)


def bibtex_key(paper: dict) -> str:
    authors = paper.get("authors") or []
    first = re.sub(r"[^a-z]", "", surname(authors[0]).lower()) if authors else "anon"
    word = next((w for w in re.findall(r"[a-z]+", (paper.get("title") or "").lower()) if len(w) > 3), "paper")
    return f"{first or 'anon'}{paper.get('year') or ''}{word}"


def bibtex_entry(paper: dict) -> str:
    fields = {
        "title": "{" + bibtex_escape(paper.get("title") or "Untitled") + "}",
        "author": bibtex_escape(" and ".join(paper.get("authors") or ["Unknown"])),
        "year": str(paper.get("year") or ""),
        "journal": bibtex_escape(paper.get("venue") or ""),
        "doi": bibtex_escape(paper.get("doi") or ""),
        "note": f"OpenAlex {paper.get('id', '')}",
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@article{{{bibtex_key(paper)},\n{body}\n}}"
