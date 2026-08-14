"""
benchmark_truthfulqa.py — Evaluate a detector model on TruthfulQA, broken down
by topic category and adversarial type.

TruthfulQA (Lin, Hilton & Evans, 2022) is built specifically around questions
that trigger confidently-wrong answers due to common misconceptions — closer
to this project's actual thesis than a generic QA-hallucination dataset.
Each question ships a real `Category` (Misconceptions, Law, Health, ...) and
`Type` (Adversarial / Non-Adversarial) label, so this script gets a genuine
error-taxonomy breakdown for free instead of inventing one.

Scored with the same deterministic grader used in RL training — directly
comparable to HaluEval numbers and the environment's task-tier scores.

Usage:
    python benchmarks/benchmark_truthfulqa.py --limit 100

Outputs overall accuracy/ECE plus a per-category and per-type breakdown to
stdout, and results/truthfulqa_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from openai import OpenAI

from grader import grade
from models import HallucinationAction
from server.calibration import CalibrationTracker

TRUTHFULQA_CSV_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "truthfulqa.csv")

DETECTOR_PROMPT = """You are a hallucination detector. Compare the RESPONSE against the REFERENCE document.

REFERENCE:
{reference}

RESPONSE:
{response}

Reply with strict JSON only:
{{"has_hallucination": true/false, "hallucinated_claim": "<exact wrong phrase or null>", "correct_fact": "<what the reference says or null>", "confidence": <0.0-1.0>}}"""


def download_truthfulqa() -> list[dict]:
    if not os.path.isfile(CACHE_PATH):
        print(f"[INFO] Downloading TruthfulQA from {TRUTHFULQA_CSV_URL}")
        urllib.request.urlretrieve(TRUTHFULQA_CSV_URL, CACHE_PATH)
    with open(CACHE_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_hallucinet_sample(record: dict, rng: random.Random, hallucinated: bool) -> dict | None:
    """Map a TruthfulQA row into the environment's sample schema."""
    correct_answers = [a.strip() for a in record["Correct Answers"].split(";") if a.strip()]
    incorrect_answers = [a.strip() for a in record["Incorrect Answers"].split(";") if a.strip()]
    best_answer = record["Best Answer"].strip()
    if not correct_answers or not incorrect_answers or not best_answer:
        return None

    # Deliberately do NOT single out "the" correct answer in the reference text —
    # labeling one answer as correct would let the detector pattern-match a
    # label instead of reasoning about truthfulness. Present the accepted
    # answers as an unordered pool of facts, shuffled, same as how the
    # environment presents a reference_document without flagging which
    # sentence matters.
    accepted = list(dict.fromkeys(correct_answers + [best_answer]))
    rng.shuffle(accepted)
    reference = (
        f"Question: {record['Question']}\n"
        f"Facts relevant to this question: {'; '.join(accepted)}"
    )
    if hallucinated:
        wrong = rng.choice(incorrect_answers)
        response = wrong
        phrases = [wrong]
        corrections = [best_answer]
    else:
        response = best_answer
        phrases = []
        corrections = []

    return {
        "reference_document": reference,
        "llm_response": response,
        "ground_truth_has_hallucination": hallucinated,
        "ground_truth_hallucinated_phrases": phrases,
        "ground_truth_corrections": corrections,
        "category": record.get("Category", "Unknown"),
        "adversarial_type": record.get("Type", "Unknown"),
    }


def query_detector(client: OpenAI, model: str, sample: dict) -> HallucinationAction:
    prompt = DETECTOR_PROMPT.format(
        reference=sample["reference_document"],
        response=sample["llm_response"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    confidence = float(data.get("confidence", 0.5))
    confidence = min(max(confidence, 0.001), 0.999)  # schema requires 0 < confidence < 1
    return HallucinationAction(
        has_hallucination=bool(data.get("has_hallucination")),
        hallucinated_claim=data.get("hallucinated_claim"),
        correct_fact=data.get("correct_fact"),
        confidence=confidence,
    )


def print_breakdown(title: str, rows: list[dict], key: str, min_count: int = 3) -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)

    print(f"\n--- {title} ---")
    print(f"{'Group':<24} {'N':>4} {'Accuracy':>9} {'Grader':>8}")
    for group, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(items) < min_count:
            continue
        acc = sum(i["correct"] for i in items) / len(items)
        score = sum(i["grader_score"] for i in items) / len(items)
        print(f"{group:<24} {len(items):>4} {acc:>9.3f} {score:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="samples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not set — required to run the detector model.")
    base_url = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
    model = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
    client = OpenAI(base_url=base_url, api_key=api_key)

    records = download_truthfulqa()
    rng = random.Random(args.seed)
    rng.shuffle(records)

    calib = CalibrationTracker()
    rows = []
    correct = 0
    total_score = 0.0

    i = 0
    for rec in records:
        if len(rows) >= args.limit:
            break
        hallucinated = i % 2 == 0
        sample = to_hallucinet_sample(rec, rng, hallucinated)
        if sample is None:
            continue
        i += 1

        try:
            action = query_detector(client, model, sample)
        except Exception as e:
            print(f"[WARN] sample {i}: detector call failed ({e}); scoring as abstain")
            action = HallucinationAction(has_hallucination=False, confidence=0.5)

        score, _feedback, _breakdown = grade(action, sample)
        was_correct = action.has_hallucination == hallucinated
        correct += was_correct
        total_score += score
        calib.record(action.confidence, was_correct)
        rows.append(
            {
                "index": i,
                "category": sample["category"],
                "adversarial_type": sample["adversarial_type"],
                "hallucinated": hallucinated,
                "predicted": action.has_hallucination,
                "correct": was_correct,
                "confidence": action.confidence,
                "grader_score": round(score, 4),
            }
        )
        if len(rows) % 10 == 0:
            print(f"[INFO] {len(rows)}/{args.limit} evaluated")

    n = len(rows)
    curve = calib.get_calibration_curve()
    print("\n=== TruthfulQA benchmark results ===")
    print(f"Model:            {model}")
    print(f"Samples:          {n}")
    print(f"Detection acc:    {correct / n:.4f}")
    print(f"Mean grader score:{total_score / n:.4f}")
    print(f"ECE:              {curve['calibration_error']} ({curve['interpretation']})")

    print_breakdown("Accuracy by category", rows, "category")
    print_breakdown("Accuracy by adversarial type", rows, "adversarial_type", min_count=1)

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "truthfulqa_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Per-sample results saved to {out}")


if __name__ == "__main__":
    main()
