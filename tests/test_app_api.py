"""API smoke tests for the FastAPI server via TestClient (no network)."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
STATIC_SUMMARY = os.path.join(RESULTS_DIR, "grpo_static_summary.json")
ADVERSARIAL_SUMMARY = os.path.join(RESULTS_DIR, "grpo_adversarial_summary.json")


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


def test_ablation_results_unavailable_by_default():
    """Verifies the 'not run yet' response shape. Repo-local runs may
    already have real summary files checked into results/ (this project's
    own ablation has been run) — temporarily move them aside rather than
    assuming a clean environment."""
    backups = []
    for path in (STATIC_SUMMARY, ADVERSARIAL_SUMMARY):
        if os.path.isfile(path):
            backup = path + ".bak"
            os.replace(path, backup)
            backups.append((path, backup))
    try:
        r = client.get("/ablation/results")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert "instructions" in body
    finally:
        for path, backup in backups:
            os.replace(backup, path)


def test_ablation_results_available_when_both_summaries_present(tmp_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    static_existed = os.path.isfile(STATIC_SUMMARY)
    adv_existed = os.path.isfile(ADVERSARIAL_SUMMARY)
    try:
        with open(STATIC_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(
                {"per_tier_scores": {"easy": 0.5}, "final_mean_reward": 0.4, "steps_trained": 10},
                f,
            )
        with open(ADVERSARIAL_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(
                {"per_tier_scores": {"easy": 0.6}, "final_mean_reward": 0.5, "steps_trained": 10},
                f,
            )
        r = client.get("/ablation/results")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["verdict"] == "adversarial_wins"
        assert body["per_tier_comparison"][0]["tier"] == "easy"
    finally:
        if not static_existed and os.path.isfile(STATIC_SUMMARY):
            os.remove(STATIC_SUMMARY)
        if not adv_existed and os.path.isfile(ADVERSARIAL_SUMMARY):
            os.remove(ADVERSARIAL_SUMMARY)
