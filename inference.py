"""
inference.py -- HalluciNet Adversarial Self-Play
Round 2: Multi-Agent + Self-Improvement

Generator Agent creates hallucinations.
Detector Agent catches them.
Both improve through competition.
Adaptive curriculum escalates difficulty as both agents improve.

Theme 1: Multi-Agent Interactions
Theme 4: Self-Improvement through adversarial self-play

MANDATORY:
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") -- no default, must be set
"""

import os
import csv
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI


import time

def call_with_retry(fn, retries=3, delay=2):
    """Call fn with retry logic. Returns None on all failures."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            print(f"[RETRY] attempt={attempt+1}/{retries} error={e}", flush=True)
            if attempt < retries - 1:
                time.sleep(delay)
    return None

from adversarial_coordinator import run_adversarial_session
from curriculum import AdversarialCurriculumManager

# Load .env from project root when running locally.
load_dotenv()

# Env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"

BENCHMARK = "hallucinet-adversarial"
SESSIONS = int(os.getenv("SESSIONS", "6"))
ROUNDS_PER_SESSION = int(os.getenv("ROUNDS_PER_SESSION", "8"))


# Log format

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# Main

def main() -> None:
    if not GROQ_API_KEY:
        print("[WARNING] GROQ_API_KEY not set. Inference will fail on API calls.", flush=True)
        print("[INFO] Set GROQ_API_KEY in your .env file or environment.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=GROQ_API_KEY)
    curriculum = AdversarialCurriculumManager()

    all_gen_rewards: List[float] = []
    all_det_rewards: List[float] = []
    session_results = []

    print(f"\n{'=' * 60}", flush=True)
    print("HalluciNet Adversarial -- Self-Play Training Simulation", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Sessions: {SESSIONS} | Rounds per session: {ROUNDS_PER_SESSION}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for session_num in range(1, SESSIONS + 1):
        task_id = curriculum.current_task

        log_start(
            task=f"{task_id}_session{session_num}",
            env=BENCHMARK,
            model=MODEL_NAME,
        )

        try:
            summary = run_adversarial_session(
                client=client,
                model=MODEL_NAME,
                task_id=task_id,
                rounds=ROUNDS_PER_SESSION,
            )

            gen_reward = summary["avg_generator_reward"]
            det_reward = summary["avg_detector_reward"]
            combined_score = min(max((gen_reward + det_reward) / 2, 0.001), 0.999)

            all_gen_rewards.append(gen_reward)
            all_det_rewards.append(det_reward)

            log_step(
                step=session_num,
                action=(
                    "adversarial_session "
                    f"task={task_id} "
                    f"gen_rate={summary['generator_fooling_rate']:.2f} "
                    f"det_rate={summary['detector_catch_rate']:.2f}"
                ),
                reward=combined_score,
                done=(session_num == SESSIONS),
                error=None,
            )

            curriculum_status = curriculum.record_session(summary)
            session_results.append(
                {
                    "session": session_num,
                    "task": task_id,
                    "gen_reward": gen_reward,
                    "det_reward": det_reward,
                    "combined": combined_score,
                    "curriculum_decision": curriculum_status["decision"],
                }
            )

            print(
                f"[CURRICULUM] session={session_num} "
                f"task={task_id} "
                f"decision={curriculum_status['decision']} "
                f"next={curriculum_status['next_task']}",
                flush=True,
            )

            success = combined_score >= 0.5

        except Exception as e:
            print(f"[ERROR] Session {session_num}: {e}", flush=True)
            success = False
            combined_score = 0.001
            all_gen_rewards.append(0.001)
            all_det_rewards.append(0.001)

        finally:
            log_end(
                success=success,
                steps=session_num * ROUNDS_PER_SESSION,
                score=combined_score,
                rewards=all_det_rewards[-ROUNDS_PER_SESSION:],
            )

    curriculum.print_log()

    avg_gen = sum(all_gen_rewards) / max(len(all_gen_rewards), 1)
    avg_det = sum(all_det_rewards) / max(len(all_det_rewards), 1)

    print(f"\n{'=' * 60}", flush=True)
    print("FINAL ADVERSARIAL RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Sessions completed:    {len(session_results)}", flush=True)
    print(f"  Final difficulty:      {curriculum.current_task}", flush=True)
    print(f"  Avg generator reward:  {avg_gen:.4f}", flush=True)
    print(f"  Avg detector reward:   {avg_det:.4f}", flush=True)
    print(f"  Curriculum promotions: {curriculum.promotions}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    if session_results:
        os.makedirs("results", exist_ok=True)
        with open("results/adversarial_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=session_results[0].keys())
            writer.writeheader()
            writer.writerows(session_results)
        print("[INFO] Results saved to results/adversarial_results.csv", flush=True)
    else:
        print("[INFO] No session results to save", flush=True)


if __name__ == "__main__":
    main()
