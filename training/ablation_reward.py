"""
ablation_reward.py — GRPO reward function shared by both ablation conditions.

Wraps the project's existing deterministic grader (`grader.py`) so the RL
reward signal is identical to every other evaluation path in this repo
(the live environment, the benchmark scripts). The only new logic here is
parsing a raw model completion into a HallucinationAction and handling
malformed output, per TRAINING.md's documented policy: unparseable JSON
gets a small non-zero floor reward (0.05) so GRPO can still push the model
toward valid structured output instead of a reward cliff.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from grader import grade
from models import HallucinationAction

UNPARSEABLE_FLOOR_REWARD = 0.05


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Models frequently emit `\'` to escape an apostrophe inside a
    # double-quoted string (e.g. "they\'re"), which is not a legal JSON
    # escape sequence (only \", \\, \n, \t, ... are valid) and breaks
    # json.loads even though the rest of the object is well-formed.
    # Normalizing it to a bare `'` is safe since JSON strings are
    # double-quoted, so an unescaped apostrophe needs no escaping at all.
    repaired = candidate.replace("\\'", "'")
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def parse_completion(text: str) -> Optional[HallucinationAction]:
    data = _extract_json(text)
    if data is None or "has_hallucination" not in data:
        return None
    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.001), 0.999)  # schema requires 0 < confidence < 1
    try:
        return HallucinationAction(
            has_hallucination=bool(data.get("has_hallucination")),
            hallucinated_claim=data.get("hallucinated_claim"),
            correct_fact=data.get("correct_fact"),
            confidence=confidence,
        )
    except Exception:
        return None


def score_completion(completion: str, sample: dict[str, Any]) -> float:
    """Single-completion reward. Returns UNPARSEABLE_FLOOR_REWARD if the
    model failed to produce a valid HallucinationAction."""
    action = parse_completion(completion)
    if action is None:
        return UNPARSEABLE_FLOOR_REWARD
    score, _feedback, _breakdown = grade(action, sample)
    return float(score)


def make_reward_fn(dataset_rows: list[dict]):
    """Builds a trl.GRPOTrainer-compatible reward function.

    trl calls reward_fn(prompts, completions, **kwargs) and expects a list
    of floats aligned to `completions`. We look up each prompt's original
    sample by exact prompt-string match against the dataset built in
    ablation_dataset.py, since GRPO trains directly from that dataset's
    `prompt` column.
    """
    prompt_to_sample = {row["prompt"]: row for row in dataset_rows}

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            sample = prompt_to_sample.get(prompt)
            if sample is None:
                rewards.append(UNPARSEABLE_FLOOR_REWARD)
                continue
            rewards.append(score_completion(completion, sample))
        return rewards

    return reward_fn
