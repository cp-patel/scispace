import json

from evals.selection_eval import SURVEYS, calibration, evaluate_case, load_case, markdown, summarize
from research_assistant.brain.offline import OfflineBrain
from research_assistant.config import AlgoParams
from research_assistant.embed import HashEmbedder
from research_assistant.sources.openalex import OpenAlex
from tests.fakes import fake_http, make_world, openalex_handler

ALGO = AlgoParams(openalex_pages=1, snowball_seeds=5, snowball_forward_seeds=2, ppr_seeds=10, rocchio_top=5)


def test_survey_list_meets_the_rules():
    surveys = json.loads(SURVEYS.read_text())
    assert 6 <= len(surveys) <= 8
    assert len({s["field"] for s in surveys}) >= 3
    assert all(s["id"].startswith("W") and s["doi"] for s in surveys)


async def test_eval_runs_on_a_fake_survey_and_writes_calibration():
    world = make_world()
    survey = world["W1029"]  # pretend this paper is a survey of the 'rag' topic, published 2024
    survey["publication_date"] = "2024-12-31"
    survey["referenced_works"] = [f"https://openalex.org/W1{i:03d}" for i in range(0, 20)]
    oa = OpenAlex(fake_http(openalex_handler(world)))
    case = await load_case(oa, {"key": "fake_rag", "id": "W1029"})
    assert case.id == "W1029" and len(case.answer) == 20
    rows = [await evaluate_case(case, oa, HashEmbedder(), OfflineBrain(), ALGO, shortlist_size=30, k=10)]
    row = rows[0]
    assert "W1029" not in row["ranking"] and row["in_pool"] >= 15
    for metrics in list(row["ranking"].values()) + list(row["selection"].values()):
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert row["ranking"]["full"]["recall@150"] >= 0.75
    summary = summarize(rows * 2)
    assert summary["surveys"] == 2 and summary["selection"]["submodular"]["recall@40"]["ci95"]
    cal = calibration(rows, "offline", {"judge_rates": {"claude": {"sensitivity": 0.9, "specificity": 0.8, "kappa": 0.5}}})
    assert cal["prior_table"][0][:2] == (1, 10) and set(cal["judge_rates"]) == {"offline", "claude"}
    assert "## Selection" in markdown(summary, rows, cal)
