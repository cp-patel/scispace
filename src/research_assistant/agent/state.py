"""Graph state. It holds ids and small values only; bulk data lives in Postgres.

LangGraph writes a checkpoint after every step, so a small state keeps checkpoints fast and cheap.
"""

import operator
from typing import Annotated, TypedDict


class JobState(TypedDict, total=False):
    job_id: str
    settings: dict
    plan: dict
    candidate_ids: list[str]
    must_ids: list[str]
    shortlist_ids: list[str]
    selected_ids: list[str]
    sections: list[dict]
    done_sections: Annotated[list[int], operator.add]  # each parallel section appends its index


class SectionState(TypedDict, total=False):
    job_id: str
    settings: dict
    idx: int
    selected_ids: list[str]
    round: int
    chunk_ids: list[int]
    chunk_scores: list[float]
    flags: list[dict]
    done_sections: Annotated[list[int], operator.add]


class SectionOut(TypedDict):
    """Only this key flows back to the parent graph, so parallel sections never collide on job_id or settings."""

    done_sections: Annotated[list[int], operator.add]
