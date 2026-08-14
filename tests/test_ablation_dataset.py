"""Unit tests for training.ablation_dataset (no GPU/network needed)."""

from training.ablation_dataset import (
    build_static_dataset,
    build_adversarial_dataset,
    dataset_summary,
    TIER_ORDER,
)


def test_static_dataset_shape_and_tiers():
    rows = build_static_dataset(n=50, seed=1)
    assert len(rows) == 50
    assert all(r["tier"] in TIER_ORDER for r in rows)
    assert all(r["source"] == "curated" for r in rows)


def test_static_dataset_covers_all_tiers():
    rows = build_static_dataset(n=100, seed=1)
    tiers_seen = {r["tier"] for r in rows}
    assert tiers_seen == set(TIER_ORDER)


def test_static_dataset_deterministic_with_seed():
    a = build_static_dataset(n=30, seed=7)
    b = build_static_dataset(n=30, seed=7)
    assert [r["llm_response"] for r in a] == [r["llm_response"] for r in b]


def test_row_schema_matches_environment_expectations():
    row = build_static_dataset(n=1, seed=1)[0]
    for key in (
        "reference_document",
        "llm_response",
        "ground_truth_has_hallucination",
        "ground_truth_hallucinated_phrases",
        "ground_truth_corrections",
        "prompt",
        "tier",
    ):
        assert key in row


def test_adversarial_dataset_no_generator_fn_still_works():
    rows = build_adversarial_dataset(n=50, seed=1, llm_fraction=0.3, generator_fn=None)
    assert len(rows) == 50
    sources = {r["source"] for r in rows}
    # No generator_fn passed -> gracefully degrades to curated + programmatic only.
    assert "generator_llm" not in sources
    assert "programmatic" in sources


def test_adversarial_dataset_uses_generator_fn_when_provided():
    calls = []

    def fake_generator(tier):
        calls.append(tier)
        return {
            "reference_document": "The sky is blue.",
            "llm_response": "The sky is green.",
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["green"],
            "ground_truth_corrections": ["blue"],
        }

    rows = build_adversarial_dataset(n=100, seed=3, llm_fraction=1.0, generator_fn=fake_generator)
    sources = {r["source"] for r in rows}
    assert sources == {"generator_llm"}
    assert len(calls) == 100


def test_adversarial_dataset_is_more_diverse_than_static():
    static_rows = build_static_dataset(n=150, seed=1)
    adversarial_rows = build_adversarial_dataset(n=150, seed=1, llm_fraction=0.0, generator_fn=None)
    static_diversity = dataset_summary(static_rows)["uniqueness_ratio"]
    adversarial_diversity = dataset_summary(adversarial_rows)["uniqueness_ratio"]
    # This is the whole point of the ablation's independent variable.
    assert adversarial_diversity > static_diversity


def test_dataset_summary_shape():
    rows = build_static_dataset(n=20, seed=1)
    summary = dataset_summary(rows)
    assert summary["total"] == 20
    assert sum(summary["by_tier"].values()) == 20
    assert 0.0 <= summary["uniqueness_ratio"] <= 1.0
