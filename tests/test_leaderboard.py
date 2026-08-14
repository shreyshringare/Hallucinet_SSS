"""Unit tests for server.leaderboard.Leaderboard using a temp file."""

from server.leaderboard import Leaderboard, TASK_KEYS


def make_lb(tmp_path):
    return Leaderboard(path=str(tmp_path / "leaderboard.json"))


def test_empty_leaderboard(tmp_path):
    assert make_lb(tmp_path).get_leaderboard() == []


def test_record_and_rank(tmp_path):
    lb = make_lb(tmp_path)
    lb.record_result("model-a", "easy", 0.9)
    lb.record_result("model-b", "easy", 0.5)
    rows = lb.get_leaderboard()
    assert rows[0]["model"] == "model-a"
    assert rows[0]["rank"] == 1
    assert rows[1]["rank"] == 2


def test_overall_is_mean_over_all_task_keys(tmp_path):
    lb = make_lb(tmp_path)
    lb.record_result("m", "easy", 1.0)
    row = lb.get_leaderboard()[0]
    assert row["overall"] == round(1.0 / len(TASK_KEYS), 4)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "leaderboard.json"
    lb = Leaderboard(path=str(path))
    lb.record_result("m", "hard", 0.7, trained=True)
    reloaded = Leaderboard(path=str(path))
    assert reloaded.get_or_default("m", "hard") == 0.7
    assert reloaded.get_leaderboard()[0]["trained"] is True


def test_corrupt_file_recovers(tmp_path):
    path = tmp_path / "leaderboard.json"
    path.write_text("{not json", encoding="utf-8")
    lb = Leaderboard(path=str(path))
    assert lb.get_leaderboard() == []
