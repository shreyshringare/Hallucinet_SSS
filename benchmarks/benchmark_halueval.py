"""
benchmark_halueval.py — Evaluate a detector model on the public HaluEval benchmark.

HaluEval (Li et al., EMNLP 2023) provides QA pairs with knowledge context,
a correct answer, and a hallucinated answer. This script maps each pair into
HalluciNet's sample schema and scores a detector model with the same
deterministic grader used for RL training — so numbers here are directly
comparable to the environment's task-tier scores.

Usage:
    # Requires GROQ_API_KEY (or any OpenAI-compatible endpoint) in .env
    python benchmarks/benchmark_halueval.py --limit 100

    # Different model / endpoint
    MODEL_NAME=llama-3.3-70b-versatile python benchmarks/benchmark_halueval.py

Outputs accuracy, grader score, and calibration (ECE) to stdout and
results/halueval_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from openai import OpenAI

from grader import grade
from models import HallucinationAction
from server.calibration import CalibrationTracker

HALUEVAL_QA_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "qa_data.json")

DETECTOR_PROMPT = """You are a hallucination detector. Compare the RESPONSE against the REFERENCE document.

REFERENCE:
{reference}

RESPONSE:
{response}

Reply with strict JSON only:
{{"has_hallucination": true/false, "hallucinated_claim": "<exact wrong phrase or null>", "correct_fact": "<what the reference says or null>", "confidence": <0.0-1.0>}}"""


def download_halueval() -> list[dict]:
    if not os.path.isfile(CACHE_PATH):
        print(f"[INFO] Downloading HaluEval QA data from {HALUEVAL_QA_URL}")
        urllib.request.urlretrieve(HALUEVAL_QA_URL, CACHE_PATH)
    samples = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def to_hallucinet_sample(record: dict, hallucinated: bool) -> dict:
    """Map a HaluEval QA record into the environment's sample schema."""
    answer = record["hallucinated_answer"] if hallucinated else record["right_answer"]
    return {
        "reference_document": f"{record['knowledge']}\nQuestion: {record['question']}",
        "llm_response": answer,
        "ground_truth_has_hallucination": hallucinated,
        "ground_truth_hallucinated_phrases": [record["hallucinated_answer"]] if hallucinated else [],
        "ground_truth_corrections": [record["right_answer"]] if hallucinated else [],
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
    return HallucinationAction(
        has_hallucination=bool(data.get("has_hallucination")),
        hallucinated_claim=data.get("hallucinated_claim"),
        correct_fact=data.get("correct_fact"),
        confidence=float(data.get("confidence", 0.5)),
    )


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

    records = download_halueval()
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: args.limit]

    calib = CalibrationTracker()
    rows = []
    correct = 0
    total_score = 0.0

    for i, rec in enumerate(records):
        # Alternate hallucinated / faithful so both classes are covered.
        hallucinated = i % 2 == 0
        sample = to_hallucinet_sample(rec, hallucinated)
        try:
            action = query_detector(client, model, sample)
        except Exception as e:
            print(f"[WARN] sample {i}: detector call failed ({e}); scoring as abstain")
            action = HallucinationAction(has_hallucination=False, confidence=0.5)

        score, _feedback, breakdown = grade(action, sample)
        was_correct = action.has_hallucination == hallucinated
        correct += was_correct
        total_score += score
        calib.record(action.confidence, was_correct)
        rows.append(
            {
                "index": i,
                "hallucinated": hallucinated,
                "predicted": action.has_hallucination,
                "confidence": action.confidence,
                "grader_score": round(score, 4),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[INFO] {i + 1}/{len(records)} evaluated")

    n = len(rows)
    curve = calib.get_calibration_curve()
    print("\n=== HaluEval benchmark results ===")
    print(f"Model:            {model}")
    print(f"Samples:          {n}")
    print(f"Detection acc:    {correct / n:.4f}")
    print(f"Mean grader score:{total_score / n:.4f}")
    print(f"ECE:              {curve['calibration_error']} ({curve['interpretation']})")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "halueval_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Per-sample results saved to {out}")


if __name__ == "__main__":
    main()
