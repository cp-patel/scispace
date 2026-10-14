import pytest
from pydantic import ValidationError

from research_assistant.config import STEPS, JobSettings, Settings


def test_defaults_match_the_spec():
    s = Settings(_env_file=None)
    assert s.brain == "offline"
    assert s.model_default == "claude-opus-5"
    assert s.algo.rrf_k == 60 and s.algo.bm25_k1 == 1.2 and s.algo.ppr_alpha == 0.85
    assert s.algo.section_budget_tokens == 6000 and s.job_budget_usd == 10.0
    assert all(s.model_for(step) == "claude-opus-5" for step in STEPS)
    assert s.effort_for("screen") == "low" and s.effort_for("write_section") == "high"


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("ALGO__RRF_K", "30")
    monkeypatch.setenv("MODEL_SCREEN", "claude-haiku-4-5")
    monkeypatch.setenv("BRAIN", "claude")
    s = Settings(_env_file=None)
    assert s.algo.rrf_k == 30
    assert s.model_for("screen") == "claude-haiku-4-5"
    assert s.model_for("plan") == "claude-opus-5"
    assert s.brain == "claude"


def test_job_settings_defaults_and_trimming():
    job = JobSettings(topic="  retrieval augmented generation  ")
    assert job.topic == "retrieval augmented generation"
    assert job.max_papers == 40 and job.shortlist_size == 150 and job.selection_method == "submodular"
    assert job.auto_approve is True and job.budget_usd is None


@pytest.mark.parametrize(
    "bad",
    [
        {"topic": "short"},
        {"topic": "x" * 301},
        {"topic": "a valid topic here", "max_papers": 60, "shortlist_size": 50},
        {"topic": "a valid topic here", "year_from": 2024, "year_to": 2020},
        {"topic": "a valid topic here", "must_include": ["W1"] * 21},
        {"topic": "a valid topic here", "selection_method": "random"},
        {"topic": "a valid topic here", "published_before": "2024/01/01"},
        {"topic": "a valid topic here", "budget_usd": 0},
    ],
)
def test_job_settings_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        JobSettings(**bad)
