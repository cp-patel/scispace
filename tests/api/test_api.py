"""API tests. TestClient runs the app (and its lifespan, which opens its own pool) in a background thread."""

import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from research_assistant.api.app import create_app
from research_assistant.config import Settings
from tests.conftest import TEST_DB, ensure_test_database, reset_tables

pytestmark = pytest.mark.db
KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client():
    ensure_test_database()
    settings = Settings(_env_file=None, database_url=TEST_DB, api_key="test-key")
    with TestClient(create_app(settings)) as c:
        reset_tables()
        yield c


def test_health_needs_no_key_but_jobs_do(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    r = client.get("/v1/jobs")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert client.get("/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401


def test_create_get_list_and_cancel(client):
    r = client.post("/v1/jobs", json={"topic": "retrieval augmented generation"}, headers=KEY)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    job = client.get(f"/v1/jobs/{job_id}", headers=KEY).json()
    assert job["status"] == "queued" and job["current_step"] is None and job["cost_usd"] == 0
    assert [j["id"] for j in client.get("/v1/jobs", headers=KEY).json()] == [job_id]
    assert client.get(f"/v1/jobs/{job_id}/selection", headers=KEY).json() == []
    assert client.post(f"/v1/jobs/{job_id}/approve", json={}, headers=KEY).status_code == 409
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=KEY).json() == {"status": "canceled"}
    again = client.post(f"/v1/jobs/{job_id}/cancel", headers=KEY)
    assert again.status_code == 409 and again.json()["error"]["code"] == "conflict"


@pytest.mark.parametrize("body", [
    {"topic": "short"},
    {"topic": "retrieval augmented generation", "max_papers": 5},
    {"topic": "retrieval augmented generation", "selection_method": "random"},
    {"topic": "retrieval augmented generation", "brain": "claude"},  # no Anthropic key on this server
])
def test_bad_input_is_rejected_with_one_error_shape(client, body):
    r = client.post("/v1/jobs", json=body, headers=KEY)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"


def test_unknown_job_and_output_not_ready(client):
    assert client.get(f"/v1/jobs/{uuid.uuid4()}", headers=KEY).json()["error"]["code"] == "not_found"
    job_id = client.post("/v1/jobs", json={"topic": "retrieval augmented generation"}, headers=KEY).json()["job_id"]
    r = client.get(f"/v1/jobs/{job_id}/paper", headers=KEY)
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_ready"


def test_outputs_trace_and_metrics(client):
    job_id = client.post("/v1/jobs", json={"topic": "retrieval augmented generation"}, headers=KEY).json()["job_id"]
    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        conn.execute("INSERT INTO outputs (job_id, paper_md, references_bib, report, report_md) VALUES (%s, %s, %s, %s, %s)",
                     (job_id, "# Paper", "@article{x}", Jsonb({"topic": "t"}), "# Report"))
        conn.execute("INSERT INTO job_events (job_id, node, event, data) VALUES (%s, 'rank', 'end', %s)",
                     (job_id, Jsonb({"seconds": 0.4, "in": 1912, "out": 150})))
    paper = client.get(f"/v1/jobs/{job_id}/paper", headers=KEY)
    assert paper.text == "# Paper" and paper.headers["content-type"].startswith("text/markdown")
    assert client.get(f"/v1/jobs/{job_id}/references.bib", headers=KEY).text == "@article{x}"
    assert client.get(f"/v1/jobs/{job_id}/report", headers=KEY).json() == {"topic": "t"}
    trace = client.get(f"/v1/jobs/{job_id}/trace", headers=KEY).json()
    assert trace[0]["node"] == "rank" and trace[0]["in"] == 1912 and trace[0]["cost_usd"] == 0.0
    assert "jobs_total" in client.get("/metrics").text
