"""Unit tests for sample_generator — programmatic sample creation."""

import random

from sample_generator import (
    generate_hallucination_sample,
    generate_clean_sample,
    generate_batch,
)

REQUIRED_KEYS = {
    "reference_document",
    "llm_response",
    "ground_truth_has_hallucination",
    "ground_truth_hallucinated_phrases",
    "ground_truth_corrections",
}


def test_hallucination_sample_schema():
    s = generate_hallucination_sample()
    assert REQUIRED_KEYS <= set(s)
    assert s["ground_truth_has_hallucination"] is True
    assert len(s["ground_truth_hallucinated_phrases"]) == 1
    assert len(s["ground_truth_corrections"]) == 1


def test_hallucinated_phrase_appears_in_response_not_reference():
    random.seed(7)
    for _ in range(20):
        s = generate_hallucination_sample()
        wrong = s["ground_truth_hallucinated_phrases"][0]
        correct = s["ground_truth_corrections"][0]
        assert wrong in s["llm_response"]
        assert correct in s["reference_document"]
        assert wrong != correct


def test_clean_sample_has_no_hallucination():
    random.seed(3)
    for _ in range(20):
        s = generate_clean_sample()
        assert s["ground_truth_has_hallucination"] is False
        assert s["ground_truth_hallucinated_phrases"] == []


def test_batch_size_and_mix():
    random.seed(1)
    batch = generate_batch(n=10, clean_ratio=0.2)
    assert len(batch) == 10
    clean = sum(1 for s in batch if not s["ground_truth_has_hallucination"])
    assert clean >= 1  # at least one clean sample enforced


def test_batch_all_generated_flagged():
    batch = generate_batch(n=5)
    assert all(s.get("generated") for s in batch)
