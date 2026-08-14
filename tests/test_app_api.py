"""API smoke tests for the FastAPI server via TestClient (no network)."""

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_demo_ui_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert r.text.lstrip().lower().startswith("<!doctype html")


def test_reset_and_step_roundtrip():
    r = client.post("/reset", json={"task_id": "easy"})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs["task_id"] == "easy"
    assert obs["reference_document"]

    r = client.post(
        "/step",
        json={
            "action": {
                "has_hallucination": True,
                "hallucinated_claim": "something wrong",
                "correct_fact": "something right",
                "confidence": 0.6,
            }
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "reward" in body
    assert "observation" in body


@pytest.mark.parametrize(
    "endpoint",
    [
        "/state",
        "/adversarial/info",
        "/leaderboard",
        "/stats",
        "/taxonomy",
        "/curriculum/status",
        "/oversight/status",
        "/elo/standings",
        "/calibration",
        "/metadata",
        "/schema",
        "/tasks/summary",
    ],
)
def test_get_endpoints_respond(endpoint):
    r = client.get(endpoint)
    assert r.status_code == 200


def test_invalid_task_id_handled():
    r = client.post("/reset", json={"task_id": "nonexistent"})
    # Server should either 4xx or fall back — never 500.
    assert r.status_code < 500
