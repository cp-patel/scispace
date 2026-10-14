"""The Brain interface: every step that needs judgment or writing goes through it.

Two implementations exist: OfflineBrain (rules, free, deterministic) and ClaudeBrain (Anthropic API).
The agent only talks to this protocol, so swapping brains needs no other code change.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class Facet(BaseModel):
    id: int
    question: str
    keywords: list[str]


class Criteria(BaseModel):
    include: list[str]
    exclude: list[str]


class Plan(BaseModel):
    facets: list[Facet] = Field(min_length=1, max_length=8)
    criteria: Criteria


class PaperView(BaseModel):
    """What a brain sees of a paper. rank, shortlist_size and facet_relevance feed only the offline rules."""

    id: str
    title: str
    abstract: str
    year: int | None = None
    rank: int = 0
    shortlist_size: int = 0
    facet_relevance: list[float] = Field(default_factory=list)


class Verdict(BaseModel):
    paper_id: str
    include: bool
    facets: list[int]
    reason: str


class SectionSpec(BaseModel):
    title: str
    goal: str


class ChunkView(BaseModel):
    chunk_id: int
    paper_id: str
    text: str


class EvidenceCard(BaseModel):
    """A claim plus a quote copied word for word from one chunk. Code assigns the card id later."""

    chunk_id: int
    claim: str
    quote: str


class CardView(BaseModel):
    card_id: str
    paper_label: str
    claim: str
    quote: str


class SentenceCheck(BaseModel):
    idx: int
    sentence: str
    quotes: list[str]


class SupportLabel(BaseModel):
    idx: int
    label: Literal["supported", "partial", "unsupported"]
    reason: str


class Flag(BaseModel):
    sentence: str
    problem: Literal["unsupported", "uncited", "unknown_citation"]


class Frame(BaseModel):
    title: str
    abstract: str
    introduction: str
    conclusion: str


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    latency_ms: int


UsageCallback = Callable[[str, Usage], Awaitable[None]]


class BrainError(Exception):
    """A brain step returned something unusable."""


class BrainRefusal(BrainError):
    def __init__(self, step: str, category: str | None):
        super().__init__(f"{step}: the model declined (category: {category})")
        self.step = step
        self.category = category


class Brain(Protocol):
    name: str

    async def plan(self, topic: str) -> Plan: ...
    async def screen(self, plan: Plan, papers: list[PaperView]) -> list[Verdict]: ...
    async def name_sections(self, clusters: list[list[PaperView]]) -> list[SectionSpec]: ...
    async def compress(self, section: SectionSpec, chunks: list[ChunkView]) -> list[EvidenceCard]: ...
    async def write_section(self, section: SectionSpec, cards: list[CardView]) -> str: ...
    async def check_support(self, items: list[SentenceCheck]) -> list[SupportLabel]: ...
    async def revise(self, section: SectionSpec, text: str, flags: list[Flag], cards: list[CardView]) -> str: ...
    async def write_frame(self, topic: str, sections: list[tuple[SectionSpec, str]], cards: list[CardView]) -> Frame: ...
