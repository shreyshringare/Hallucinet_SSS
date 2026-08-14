"""
ablation_dataset.py — Builds the two training sets for the self-play ablation.

The ablation isolates one variable: does the detector generalize better when
trained against an ever-expanding, adversarially-generated task distribution
versus the same fixed 73-sample curated pool repeated?

Two conditions:

  STATIC       Cycles only `tasks.py`'s hand-curated samples. Finite,
               memorizable — this is the control condition.

  ADVERSARIAL  Same curated pool, plus unlimited programmatically-generated
               samples from `sample_generator.py` (novel fact/value swaps
               the detector can never have memorized), plus an optional
               fraction of genuinely LLM-generated adversarial hallucinations
               produced by a generator agent (mirrors the environment's
               actual generator-vs-detector self-play, not just templated
               substitution).

Both conditions use the same curriculum tier schedule (easy -> medium ->
hard -> expert -> adversarial) so difficulty progression is not a
confound — only task-content diversity differs between them.

Every row has the shape HallucinationEnvironment already expects:
{reference_document, llm_response, ground_truth_has_hallucination,
 ground_truth_hallucinated_phrases, ground_truth_corrections}
plus a `tier` tag used to build the GRPO prompt and to bucket results.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from tasks import get_task, TASKS
from sample_generator import generate_hallucination_sample, generate_clean_sample

TIER_ORDER = ["easy", "medium", "hard", "expert", "adversarial"]

DETECTOR_PROMPT_TEMPLATE = """You are a hallucination detector. Compare the RESPONSE against the REFERENCE document.

REFERENCE:
{reference}

RESPONSE:
{response}

Reply with strict JSON only:
{{"has_hallucination": true/false, "hallucinated_claim": "<exact wrong phrase or null>", "correct_fact": "<what the reference says or null>", "confidence": <0.0-1.0>}}"""


def _tier_schedule(n: int) -> list[str]:
    """Even split of n rows across the curriculum tiers, in order."""
    per_tier = max(1, n // len(TIER_ORDER))
    schedule = []
    for tier in TIER_ORDER:
        schedule.extend([tier] * per_tier)
    while len(schedule) < n:
        schedule.append(TIER_ORDER[len(schedule) % len(TIER_ORDER)])
    return schedule[:n]


def _to_row(sample: dict, tier: str, source: str) -> dict:
    return {
        "tier": tier,
        "source": source,
        "reference_document": sample["reference_document"],
        "llm_response": sample["llm_response"],
        "ground_truth_has_hallucination": bool(sample["ground_truth_has_hallucination"]),
        "ground_truth_hallucinated_phrases": list(sample.get("ground_truth_hallucinated_phrases") or []),
        "ground_truth_corrections": list(sample.get("ground_truth_corrections") or []),
        "prompt": DETECTOR_PROMPT_TEMPLATE.format(
            reference=sample["reference_document"],
            response=sample["llm_response"],
        ),
    }


def build_static_dataset(n: int = 200, seed: int = 42) -> list[dict]:
    """Control condition: cycle only the fixed curated pool. Finite content,
    difficulty follows the standard tier schedule."""
    import random

    rng = random.Random(seed)
    rows = []
    for tier in _tier_schedule(n):
        pool = get_task(tier)
        sample = rng.choice(pool)
        rows.append(_to_row(sample, tier, source="curated"))
    return rows


def build_adversarial_dataset(
    n: int = 200,
    seed: int = 42,
    llm_fraction: float = 0.3,
    generator_fn: Optional[Callable[[str], dict]] = None,
) -> list[dict]:
    """Treatment condition: curated pool + unlimited programmatic variety
    + (optionally) genuine LLM-generated adversarial hallucinations.

    generator_fn(tier) -> sample dict, same schema as sample_generator output.
    Pass a real generator-agent call (see training/train_ablation.py) to get
    true adversarial content; omitted in tests / dry runs to avoid network
    calls, in which case this condition degrades gracefully to
    curated + programmatic-only (still a valid, weaker treatment).
    """
    import random

    rng = random.Random(seed)
    rows = []
    for tier in _tier_schedule(n):
        roll = rng.random()
        if roll < llm_fraction and generator_fn is not None:
            sample = generator_fn(tier)
            rows.append(_to_row(sample, tier, source="generator_llm"))
        elif roll < llm_fraction + 0.35:
            sample = get_task(tier)[rng.randrange(len(get_task(tier)))]
            rows.append(_to_row(sample, tier, source="curated"))
        else:
            # Programmatic novel variation — unbounded, never memorizable.
            sample = (
                generate_hallucination_sample(difficulty=tier)
                if rng.random() < 0.8
                else generate_clean_sample()
            )
            rows.append(_to_row(sample, tier, source="programmatic"))
    return rows


def dataset_summary(rows: list[dict]) -> dict[str, Any]:
    """Diversity/composition stats used to sanity-check a built dataset
    before spending GPU hours training on it."""
    by_tier: dict[str, int] = {}
    by_source: dict[str, int] = {}
    unique_responses = set()
    for r in rows:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        unique_responses.add(r["llm_response"])
    return {
        "total": len(rows),
        "by_tier": by_tier,
        "by_source": by_source,
        "unique_responses": len(unique_responses),
        "uniqueness_ratio": round(len(unique_responses) / max(len(rows), 1), 4),
    }
