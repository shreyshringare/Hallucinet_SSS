"""Unit tests for curriculum.AdversarialCurriculumManager."""

from curriculum import AdversarialCurriculumManager, TASK_ORDER


def make_summary(det=0.5, gen=0.5):
    return {"detector_catch_rate": det, "generator_fooling_rate": gen}


def test_starts_at_easy():
    assert AdversarialCurriculumManager().current_task == "easy"


def test_no_promotion_before_window_full():
    c = AdversarialCurriculumManager()
    entry = c.record_session(make_summary(det=0.9))
    assert entry["decision"] == "stay"
    assert c.current_task == "easy"


def test_promotes_on_sustained_detector_mastery():
    c = AdversarialCurriculumManager()
    decisions = [c.record_session(make_summary(det=0.9))["decision"] for _ in range(3)]
    assert decisions[-1] == "promote"
    assert c.current_task == "medium"
    assert c.promotions == 1


def test_promotes_when_generator_dominates():
    c = AdversarialCurriculumManager()
    for _ in range(3):
        entry = c.record_session(make_summary(det=0.5, gen=0.9))
    assert entry["decision"] == "promote"
    assert c.current_task == "medium"


def test_demotes_on_sustained_failure():
    c = AdversarialCurriculumManager()
    c.current_level = 2  # hard
    for _ in range(3):
        entry = c.record_session(make_summary(det=0.2))
    assert entry["decision"] == "demote"
    assert c.current_task == "medium"
    assert c.demotions == 1


def test_no_demotion_below_easy():
    c = AdversarialCurriculumManager()
    for _ in range(6):
        c.record_session(make_summary(det=0.1))
    assert c.current_task == "easy"
    assert c.demotions == 0


def test_no_promotion_above_max():
    c = AdversarialCurriculumManager()
    c.current_level = len(TASK_ORDER) - 1
    for _ in range(6):
        c.record_session(make_summary(det=0.99))
    assert c.current_task == TASK_ORDER[-1]


def test_oscillation_guard_blocks_detector_promotion_from_demoted_level():
    c = AdversarialCurriculumManager()
    c.current_level = 1  # medium
    for _ in range(3):
        c.record_session(make_summary(det=0.2))  # demote to easy
    assert c.current_task == "easy"
    # Mastering easy re-promotes back to medium (allowed)...
    for _ in range(3):
        c.record_session(make_summary(det=0.9))
    assert c.current_task == "medium"
    # ...but the guard blocks detector-driven promotion from the level it
    # was demoted from, preventing promote/demote ping-pong.
    for _ in range(3):
        entry = c.record_session(make_summary(det=0.9, gen=0.1))
    assert entry["decision"] == "stay"
    assert c.current_task == "medium"


def test_history_window_rolls():
    c = AdversarialCurriculumManager()
    for _ in range(5):
        c.record_session(make_summary(det=0.5))
    assert len(c.detector_history) == c.window


def test_get_status_shape():
    c = AdversarialCurriculumManager()
    status = c.get_status()
    assert status["current_task"] == "easy"
    assert "next_promotion" in status
    assert status["promotion_threshold"] == 0.75
