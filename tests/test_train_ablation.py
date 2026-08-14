"""Unit tests for training.train_ablation's non-GPU logic (dataset dispatch,
generator-agent wiring). The actual GRPOTrainer training loop in main()
requires unsloth/trl/peft + a GPU and is intentionally not covered here —
see training/train_ablation.py's module docstring."""

import json
from types import SimpleNamespace

import pytest

from training.train_ablation import build_dataset, make_generator_fn


def test_build_dataset_static():
    rows = build_dataset("static", n=10, seed=1, llm_fraction=0.3)
    assert len(rows) == 10
    assert all(r["source"] == "curated" for r in rows)


def test_build_dataset_adversarial():
    rows = build_dataset("adversarial", n=10, seed=1, llm_fraction=0.0)
    assert len(rows) == 10


def test_build_dataset_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_dataset("bogus", n=10, seed=1, llm_fraction=0.0)


class _FakeCompletions:
    def create(self, model, messages, temperature, max_tokens):
        payload = {
            "reference_document": "The sky is blue during the day.",
            "llm_response": "The sky is green during the day.",
            "ground_truth_hallucinated_phrases": ["green"],
            "ground_truth_corrections": ["blue"],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_make_generator_fn_produces_valid_sample():
    generator_fn = make_generator_fn(_FakeClient(), model="fake-model")
    sample = generator_fn("easy")
    assert sample["ground_truth_has_hallucination"] is True
    assert sample["llm_response"] == "The sky is green during the day."
    assert sample["ground_truth_corrections"] == ["blue"]


def test_make_generator_fn_strips_markdown_fence():
    class FencedCompletions:
        def create(self, model, messages, temperature, max_tokens):
            payload = {
                "reference_document": "Water boils at 100C at sea level.",
                "llm_response": "Water boils at 120C at sea level.",
                "ground_truth_hallucinated_phrases": ["120C"],
                "ground_truth_corrections": ["100C"],
            }
            text = "```json\n" + json.dumps(payload) + "\n```"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FencedCompletions()))
    generator_fn = make_generator_fn(client, model="fake-model")
    sample = generator_fn("medium")
    assert sample["llm_response"] == "Water boils at 120C at sea level."


def test_make_generator_fn_uses_local_reference_not_model_output():
    """Regression test: earlier versions asked the model to copy the
    reference document back inside JSON, which broke on real inputs
    containing quotes. The reference must come from the local task data,
    not from the model's response."""
    generator_fn = make_generator_fn(_FakeClient(), model="fake-model")
    sample = generator_fn("easy")
    assert sample["reference_document"] != "The sky is blue during the day."
    assert len(sample["reference_document"]) > 0


class _MalformedThenValidCompletions:
    """Simulates a model that breaks JSON on the first call (e.g. by
    embedding an unescaped quote) and recovers on retry."""

    def __init__(self):
        self.calls = 0

    def create(self, model, messages, temperature, max_tokens):
        self.calls += 1
        if self.calls == 1:
            text = '{"llm_response": "The city\'s "old" bridge collapsed in 1990.", "ground_truth_hallucinated_phrases": ["1990"]}'
        else:
            payload = {
                "llm_response": "The bridge collapsed in 1990.",
                "ground_truth_hallucinated_phrases": ["1990"],
                "ground_truth_corrections": ["1985"],
            }
            text = json.dumps(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_make_generator_fn_retries_on_malformed_json():
    completions = _MalformedThenValidCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator_fn = make_generator_fn(client, model="fake-model", max_retries=3)
    sample = generator_fn("hard")
    assert sample["llm_response"] == "The bridge collapsed in 1990."
    assert completions.calls == 2


class _AlwaysMalformedCompletions:
    def create(self, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
        )


def test_make_generator_fn_raises_after_exhausting_retries():
    client = SimpleNamespace(chat=SimpleNamespace(completions=_AlwaysMalformedCompletions()))
    generator_fn = make_generator_fn(client, model="fake-model", max_retries=2)
    with pytest.raises(RuntimeError):
        generator_fn("expert")
