"""Split a paper into chunks: the abstract first, then section-aware windows of about 500 tokens.

Section-aware chunking keeps each chunk inside one section, so a retrieved passage is about one
thing (a method, a result) instead of straddling two. Windows pack whole sentences, so a quote is
never cut in half, and consecutive windows overlap a little so a claim near a boundary appears whole
in at least one chunk.
"""

import re
from dataclasses import dataclass

from research_assistant.text.tokenize import approx_tokens, split_sentences

_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z][^\n]{2,80})$")
_NAMED = {
    "introduction", "related work", "background", "method", "methods", "methodology", "approach", "model",
    "experiments", "experimental setup", "evaluation", "results", "discussion", "conclusion", "conclusions",
    "limitations",
}
_SKIP = {"abstract"}
_STOP = {"references", "bibliography", "acknowledgments", "acknowledgements", "appendix"}


@dataclass(frozen=True)
class Chunk:
    ord: int
    section: str
    text: str
    tokens: int


def heading(line: str) -> str | None:
    """Return the heading text if the line looks like a section heading, else None."""
    s = line.strip()
    match = _NUMBERED.match(s)
    if match:
        title = match.group(2).strip()
        return None if sum(ch.isdigit() for ch in title) > 3 else title  # table rows are digit-heavy
    name = s.rstrip(":").strip()
    return name if name.lower() in _NAMED | _SKIP | _STOP else None


def strip_back_matter(text: str) -> str:
    """Cut everything from the first References / Bibliography / Appendix heading onward."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().rstrip(":").lower() in _STOP:
            return "\n".join(lines[:i])
    return text


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs in order. The paper's own Abstract section is skipped (the metadata abstract is chunk 0)."""
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in strip_back_matter(text).splitlines():
        head = heading(line)
        if head is not None:
            if head.lower() in _SKIP:
                current = None
                continue
            current = []
            sections.append((head, current))
        elif current is not None:
            current.append(line)
    out = []
    for head, body in sections:
        joined = "\n".join(body).strip()
        if joined:
            out.append((head, joined))
    return out


def window(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack whole sentences into windows of about size_tokens. Each window starts with the last
    ~overlap_tokens of the previous one."""
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in split_sentences(" ".join(text.split())):
        t = approx_tokens(sentence)
        if current and current_tokens + t > size_tokens:
            windows.append(" ".join(current))
            carry: list[str] = []
            carry_tokens = 0
            for previous in reversed(current):
                pt = approx_tokens(previous)
                if carry_tokens + pt > overlap_tokens:
                    break
                carry.insert(0, previous)
                carry_tokens += pt
            current, current_tokens = carry, carry_tokens
        current.append(sentence)
        current_tokens += t
    if current:
        windows.append(" ".join(current))
    return windows


def chunk_paper(abstract: str, full_text: str | None, size_tokens: int = 500, overlap_tokens: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    if abstract.strip():
        chunks.append(Chunk(0, "Abstract", abstract.strip(), approx_tokens(abstract)))
    if full_text:
        sections = split_sections(full_text) or [("Body", strip_back_matter(full_text))]
        for head, body in sections:
            for piece in window(body, size_tokens, overlap_tokens):
                chunks.append(Chunk(len(chunks), head, piece, approx_tokens(piece)))
    return chunks
