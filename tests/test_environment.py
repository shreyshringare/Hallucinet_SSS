"""Integration tests for server.environment.HallucinationEnvironment."""

import pytest

from server.environment import HallucinationEnvironment
from models import HallucinationAction


def make_env(task_id="easy"):
    env = HallucinationEnvironment()
    obs = env.reset(task_id=task_id)
    return env, obs


def test_reset_returns_first_sample():
    env, obs = make_env()
    assert obs.done is False
    assert obs.task_id == "easy"
    assert obs.sample_index == 0
    assert obs.total_samples > 0
    assert obs.reference_document
    assert obs.llm_response
    assert "episode_id" in obs.metadata


def test_step_before_reset_raises():
    env = HallucinationEnvironment()
    with pytest.raises(RuntimeError):
        env.step(HallucinationAction(has_hallucination=True, confidence=0.5))


def test_step_advances_and_scores():
    env, obs = make_env()
    action = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="wrong",
        correct_fact="right",
        confidence=0.5,
    )
    obs2 = env.step(action)
    assert obs2.steps_taken == 1
    assert obs2.reward is not None
    assert 0.0 < obs2.score < 1.0


def test_episode_terminates_at_sample_exhaustion():
    env, obs = make_env()
    action = HallucinationAction(has_hallucination=False, confidence=0.5)
    last = obs
    for _ in range(obs.max_steps):
        last = env.step(action)
        if last.done:
            break
    assert last.done is True


def test_step_after_done_raises():
    env, obs = make_env()
    action = HallucinationAction(has_hallucination=False, confidence=0.5)
    while True:
        o = env.step(action)
        if o.done:
            break
    with pytest.raises(RuntimeError):
        env.step(action)


def test_state_tracks_progress():
    env, obs = make_env()
    env.step(HallucinationAction(has_hallucination=True, confidence=0.7))
    state = env.state()
    assert state.task_id == "easy"
    assert state.steps_taken == 1
    assert state.sample_index == 1
    assert 0.0 < state.episode_score < 1.0


def test_episode_summary_after_completion():
    env, obs = make_env()
    action = HallucinationAction(has_hallucination=True, confidence=0.9)
    while True:
        o = env.step(action)
        if o.done:
            break
    summary = env.get_episode_summary()
    assert summary["task_id"] == "easy"
    assert 0.0 <= summary["detector_catch_rate"] <= 1.0
    assert 0.0 <= summary["generator_fooling_rate"] <= 1.0


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard", "expert", "adversarial"])
def test_all_tiers_resettable(task_id):
    env, obs = make_env(task_id)
    assert obs.task_id == task_id
    assert obs.total_samples > 0
