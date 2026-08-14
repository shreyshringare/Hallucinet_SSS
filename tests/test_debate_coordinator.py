"""Unit tests for server.debate_coordinator.DebateCoordinator."""

from server.debate_coordinator import DebateCoordinator


REFERENCE = "The Eiffel Tower was completed in 1889 in Paris."
GENERATED = "The Eiffel Tower was completed in 1902 in Paris, France."
GT_PHRASES = ["1902"]


def run(defense):
    return DebateCoordinator().run_debate(
        reference=REFERENCE,
        generated_response=GENERATED,
        detector_claim="completed in 1902",
        generator_defense=defense,
        ground_truth_phrases=GT_PHRASES,
    )


def test_empty_defense_loses():
    result = run("")
    assert result["outcome"] == "detector_wins"
    assert result["generator_final_reward_delta"] == -0.15


def test_short_defense_loses():
    result = run("no it is right")
    assert result["outcome"] == "detector_wins"


def test_defense_quoting_hallucination_admits_error():
    result = run(
        "The completion date of 1902 that I stated is well documented in many "
        "sources about the Eiffel Tower construction in Paris"
    )
    assert result["defense_contradicts_truth"] is True
    assert result["outcome"] == "detector_wins"
    assert result["generator_final_reward_delta"] == -0.30


def test_long_grounded_defense_is_inconclusive():
    result = run(
        "My response about the Eiffel Tower completion in Paris France is "
        "consistent with the historical record as I described it, and the "
        "construction timeline I gave matches several accounts of the tower"
    )
    assert result["outcome"] == "inconclusive"
    assert result["generator_final_reward_delta"] == 0.10


def test_stats_aggregate():
    coord = DebateCoordinator()
    coord.run_debate(REFERENCE, GENERATED, "claim", "", GT_PHRASES)
    coord.run_debate(REFERENCE, GENERATED, "claim", "", GT_PHRASES)
    stats = coord.get_stats()
    assert stats["total_debates"] == 2
    assert stats["detector_wins"] == 2
    assert stats["detector_win_rate"] == 1.0
