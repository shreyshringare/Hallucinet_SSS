"""Unit tests for training.ablation_reward — GRPO reward wiring."""

import json

from training.ablation_dataset import build_static_dataset
from training.ablation_reward import (
    parse_completion,
    score_completion,
    make_reward_fn,
    UNPARSEABLE_FLOOR_REWARD,
)

SAMPLE = {
    "reference_document": "The Eiffel Tower was completed in 1889.",
    "llm_response": "The Eiffel Tower was completed in 1902.",
    "ground_truth_has_hallucination": True,
    "ground_truth_hallucinated_phrases": ["1902"],
    "ground_truth_corrections": ["1889"],
}


def test_parse_valid_json():
    text = json.dumps({
        "has_hallucination": True,
        "hallucinated_claim": "1902",
        "correct_fact": "1889",
        "confidence": 0.9,
    })
    action = parse_completion(text)
    assert action is not None
    assert action.has_hallucination is True
    assert action.confidence == 0.9


def test_parse_handles_markdown_fenced_json():
    text = "```json\n" + json.dumps({"has_hallucination": False, "confidence": 0.6}) + "\n```"
    action = parse_completion(text)
    assert action is not None
    assert action.has_hallucination is False


def test_parse_clamps_confidence_out_of_bounds():
    text = json.dumps({"has_hallucination": True, "confidence": 1.0})
    action = parse_completion(text)
    assert action is not None
    assert action.confidence < 1.0

    text2 = json.dumps({"has_hallucination": True, "confidence": 0.0})
    action2 = parse_completion(text2)
    assert action2.confidence > 0.0


def test_parse_garbage_returns_none():
    assert parse_completion("not json at all") is None
    assert parse_completion("") is None


def test_parse_missing_required_field_returns_none():
    assert parse_completion(json.dumps({"confidence": 0.5})) is None


def test_score_completion_rewards_correct_detection():
    text = json.dumps({
        "has_hallucination": True,
        "hallucinated_claim": "1902",
        "correct_fact": "1889",
        "confidence": 0.9,
    })
    score = score_completion(text, SAMPLE)
    assert score > 0.5


def test_score_completion_unparseable_gets_floor_reward():
    score = score_completion("garbage output", SAMPLE)
    assert score == UNPARSEABLE_FLOOR_REWARD


def test_reward_fn_aligns_prompts_to_completions():
    rows = build_static_dataset(n=5, seed=1)
    reward_fn = make_reward_fn(rows)
    prompts = [rows[0]["prompt"], rows[1]["prompt"]]
    completions = [
        json.dumps({"has_hallucination": rows[0]["ground_truth_has_hallucination"], "confidence": 0.8}),
        "unparseable garbage",
    ]
    rewards = reward_fn(prompts, completions)
    assert len(rewards) == 2
    assert rewards[1] == UNPARSEABLE_FLOOR_REWARD


def test_reward_fn_unknown_prompt_gets_floor_reward():
    rows = build_static_dataset(n=3, seed=1)
    reward_fn = make_reward_fn(rows)
    rewards = reward_fn(["prompt not in dataset"], ["anything"])
    assert rewards == [UNPARSEABLE_FLOOR_REWARD]
