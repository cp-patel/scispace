"""Turn PDF bytes into clean plain text. The PDF itself is never stored."""

import io
import re

from pypdf import PdfReader

_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": ""})


class PdfError(Exception):
    """The PDF could not be read."""


def extract_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises many different types on broken files
        raise PdfError(str(exc)) from exc
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Fix ligatures and soft hyphens, re-join words split across lines, and tidy whitespace (keeping line breaks)."""
    text = text.translate(_LIGATURES)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
