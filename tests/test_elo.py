"""Unit tests for server.elo.ELOTracker."""

from server.elo import ELOTracker


def test_default_rating_is_1000():
    elo = ELOTracker()
    assert elo.get_rating("detector") == 1000.0
    assert elo.get_rating("generator") == 1000.0


def test_winner_gains_loser_loses():
    elo = ELOTracker()
    elo.update(winner="detector", loser="generator")
    assert elo.get_rating("detector") > 1000.0
    assert elo.get_rating("generator") < 1000.0


def test_equal_ratings_symmetric_update():
    elo = ELOTracker(k=32)
    elo.update(winner="detector", loser="generator")
    # With equal ratings, expected score is 0.5 -> winner gains k/2 = 16.
    assert abs(elo.get_rating("detector") - 1016.0) < 1e-9
    assert abs(elo.get_rating("generator") - 984.0) < 1e-9


def test_upset_gains_more_than_expected_win():
    elo = ELOTracker()
    elo.ratings = {"underdog": 800.0, "favorite": 1200.0}
    entry = elo.update(winner="underdog", loser="favorite")
    underdog_gain = elo.get_rating("underdog") - 800.0
    # Beating a much stronger opponent should award more than k/2.
    assert underdog_gain > 16.0
    assert entry["winner"] == "underdog"


def test_history_and_standings():
    elo = ELOTracker()
    elo.update(winner="detector", loser="generator")
    elo.update(winner="detector", loser="generator")
    standings = elo.get_standings()
    assert standings["total_rounds"] == 2
    assert standings["current_leader"] == "detector"
    assert standings["detector_elo"] > standings["generator_elo"]


def test_standings_empty():
    assert ELOTracker().get_standings()["current_leader"] == "none"
