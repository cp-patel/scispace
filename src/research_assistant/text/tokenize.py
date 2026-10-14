"""Shared text helpers: tokens for BM25, rough token counts, sentence and paragraph splits, and text normalization for quote matching."""

import math
import re

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too under until up very
    via was we were what when where which while who whom why will with would you your yours yourself yourselves
    use used using based paper study""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_ABBREVIATION = re.compile(r"\b(?:et al|e\.g|i\.e|fig|figs|eq|eqs|vs|cf|approx|sec|no|ref|refs)\.", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Hyphenated terms like 'retrieval-augmented' stay whole.
    Stopwords and one-character tokens are dropped."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def approx_tokens(text: str) -> int:
    """Rough LLM token count, about 1.3 tokens per word. Good enough for budgets."""
    return math.ceil(len(text.split()) * 1.3)


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by a capital, digit, bracket, or quote. 'et al.', 'e.g.', 'Fig.' and decimals do not split."""
    protected = _ABBREVIATION.sub(lambda m: m.group(0)[:-1] + "\x00", text.strip())
    return [p.replace("\x00", ".").strip() for p in _SENTENCE_BREAK.split(protected) if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def normalize_for_match(text: str) -> str:
    """Lowercase, straighten curly quotes and dashes, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCT)).strip()
