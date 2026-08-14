"""
train_ablation.py — GRPO self-play ablation: static task pool vs adversarial
task generation.

This answers the one architectural claim the rest of this repo asserts but
never tested: does training the detector against an adversarially-expanding
task distribution (see ablation_dataset.build_adversarial_dataset) produce a
better detector than training on the fixed 73-sample curated pool alone
(build_static_dataset)?

Both runs use the exact config documented in TRAINING.md — same base model,
same LoRA rank/targets, same reward function (grader.py, via
training/ablation_reward.py) — so the task distribution is the only
independent variable.

Requires a GPU (T4 or better). Run in Colab or any CUDA box:

    pip install unsloth trl peft bitsandbytes accelerate
    python -m training.train_ablation --mode static
    python -m training.train_ablation --mode adversarial --llm-fraction 0.3

Outputs, per mode, to results/:
    grpo_<mode>_log.csv       per-step reward log
    grpo_<mode>_summary.json  final per-tier scores + external benchmark score

Heavy ML deps (unsloth/trl/peft) are imported lazily inside main() so this
module can be imported and its non-training helpers unit-tested on a
machine without a GPU or those packages installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.ablation_dataset import build_static_dataset, build_adversarial_dataset, dataset_summary
from training.ablation_reward import make_reward_fn

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"
LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def build_dataset(mode: str, n: int, seed: int, llm_fraction: float, generator_fn=None) -> list[dict]:
    if mode == "static":
        return build_static_dataset(n=n, seed=seed)
    if mode == "adversarial":
        return build_adversarial_dataset(n=n, seed=seed, llm_fraction=llm_fraction, generator_fn=generator_fn)
    raise ValueError(f"unknown mode: {mode!r} (expected 'static' or 'adversarial')")


def make_generator_fn(client, model: str):
    """Wraps a Groq/OpenAI-compatible client into a training-time generator
    agent: given a curriculum tier, produces one fresh adversarial
    hallucination sample by asking the model to subtly corrupt a fact from
    a real reference document."""
    import random
    from tasks import get_task

    GEN_PROMPT = """Given this REFERENCE document, write a RESPONSE that subtly misstates one fact from it \
(a wrong number, date, name, or a negation) while sounding equally confident and fluent. \
Reply with strict JSON only:
{{"reference_document": "<copy the reference below unchanged>", "llm_response": "<your subtly wrong version>", \
"ground_truth_hallucinated_phrases": ["<the wrong phrase you inserted>"], \
"ground_truth_corrections": ["<what the reference actually says>"]}}

REFERENCE:
{reference}"""

    def generator_fn(tier: str) -> dict:
        ref_sample = random.choice(get_task(tier))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": GEN_PROMPT.format(reference=ref_sample["reference_document"])}],
            temperature=1.0,
            max_tokens=400,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        return {
            "reference_document": data["reference_document"],
            "llm_response": data["llm_response"],
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": data.get("ground_truth_hallucinated_phrases", []),
            "ground_truth_corrections": data.get("ground_truth_corrections", []),
        }

    return generator_fn


def evaluate_per_tier(model, tokenizer, rows: list[dict]) -> dict:
    """Runs the trained model zero-shot (greedy) over held-out rows per
    tier, scored with the same deterministic grader used for training."""
    from training.ablation_reward import score_completion
    from collections import defaultdict

    by_tier = defaultdict(list)
    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.0, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        score = score_completion(completion, row)
        by_tier[row["tier"]].append(score)

    return {tier: round(sum(scores) / len(scores), 4) for tier, scores in by_tier.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "adversarial"], required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-fraction", type=float, default=0.3)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"hallucinet-grpo-{args.mode}"
    os.makedirs("results", exist_ok=True)

    generator_fn = None
    if args.mode == "adversarial" and args.llm_fraction > 0:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sys.exit("GROQ_API_KEY required for --mode adversarial with llm-fraction > 0 (generator agent).")
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        generator_fn = make_generator_fn(client, os.getenv("MODEL_NAME") or "llama-3.1-8b-instant")

    dataset_rows = build_dataset(args.mode, args.n_samples, args.seed, args.llm_fraction, generator_fn)
    summary = dataset_summary(dataset_rows)
    print(f"[INFO] mode={args.mode} dataset built: {summary}")

    # --- heavy ML deps, imported only when actually training ---
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    hf_dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in dataset_rows])
    reward_fn = make_reward_fn(dataset_rows)

    reward_log: list[dict] = []

    def logging_reward_fn(prompts, completions, **kwargs):
        rewards = reward_fn(prompts, completions, **kwargs)
        for r in rewards:
            reward_log.append({"step": len(reward_log), "reward": r})
        return rewards

    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=args.max_steps,
        temperature=1.0,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=hf_dataset,
        reward_funcs=[logging_reward_fn],
    )
    trainer.train()

    log_path = os.path.join("results", f"grpo_{args.mode}_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "reward"])
        writer.writeheader()
        writer.writerows(reward_log)
    print(f"[INFO] reward log saved to {log_path}")

    per_tier_scores = evaluate_per_tier(model, tokenizer, dataset_rows)
    final_summary = {
        "mode": args.mode,
        "base_model": args.base_model,
        "dataset_summary": summary,
        "per_tier_scores": per_tier_scores,
        "final_mean_reward": round(sum(r["reward"] for r in reward_log) / max(len(reward_log), 1), 4),
        "steps_trained": args.max_steps,
    }
    summary_path = os.path.join("results", f"grpo_{args.mode}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    print(f"[INFO] summary saved to {summary_path}")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
