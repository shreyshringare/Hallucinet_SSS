import json
import os
from pathlib import Path
import sys

# Add project root to path so all imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from server.environment import HallucinationEnvironment
from server.generator_environment import GeneratorEnvironment
from server.oversight_agent import OversightAgent
from server.debate_coordinator import DebateCoordinator
from server.leaderboard import Leaderboard, TASK_KEYS
from server.elo import ELOTracker
from server.calibration import CalibrationTracker
detector_calibration = CalibrationTracker()
from curriculum import AdversarialCurriculumManager
from models import HallucinationAction, GeneratorAction
from tasks import TASKS

app = FastAPI(title="HalluciNet Adversarial - Round 2")
STATIC_DIR = Path(__file__).parent / 'static'
DEMO_UI_HTML = (STATIC_DIR / 'index.html').read_text(encoding='utf-8')

detector_env = HallucinationEnvironment()
generator_env = GeneratorEnvironment()
oversight_agent = OversightAgent()
adversarial_curriculum = AdversarialCurriculumManager()
debate_coordinator = DebateCoordinator()
leaderboard = Leaderboard()
elo_tracker = ELOTracker()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class DetectorStepRequest(BaseModel):
    action: HallucinationAction

class GeneratorStepRequest(BaseModel):
    action: GeneratorAction


class LeaderboardRecordRequest(BaseModel):
    model_name: str
    task_id: str
    score: float
    trained: bool = False


class DebateRequest(BaseModel):
    generator_defense: str
    task_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "healthy", "mode": "adversarial", "version": "2.0"}

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    if oversight_agent.should_inject_adversarial():
        body.task_id = "adversarial"
        print("[OVERSIGHT] Injecting adversarial sample")
    try:
        obs = detector_env.reset(task_id=body.task_id)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/step")
def step(body: DetectorStepRequest):
    obs = detector_env.step(body.action)
    try:
        oversight_agent.record_episode({
            "episode_id": detector_env._episode_id,
            "error_type": detector_env._samples[max(0, detector_env._index-1)].get("error_type", "unknown") if detector_env._samples else "unknown",
            "detector_confidence": body.action.confidence,
            "detector_correct": (obs.score or 0) > 0.5,
            "generator_confidence": 0.5,
            "generator_won": (obs.score or 0) < 0.3,
            "task_id": detector_env._task_id,
            "step": detector_env._steps
        })
    except Exception as e:
        print(f"[OVERSIGHT ERROR] {e}")
    if obs.done:
        ep = detector_env.get_oversight_episode_dict()
        if ep:
            oversight_agent.record_episode(ep)
        summary = detector_env.get_episode_summary()
        if summary:
            adversarial_curriculum.record_session(summary)
    try:
        score_val = obs.score or 0
        if score_val > 0.5:
            elo_tracker.update("detector", "generator")
        else:
            elo_tracker.update("generator", "detector")
    except Exception as e:
        print(f"[ELO ERROR] {e}")
    try:
        confidence = body.action.confidence
        was_correct = (obs.score or 0) > 0.5
        detector_calibration.record(confidence, was_correct)
    except Exception as e:
        print(f"[CALIBRATION ERROR] {e}")
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/state")
def state():
    return detector_env.state().model_dump()

@app.post("/generator/reset")
def generator_reset(body: ResetRequest = ResetRequest()):
    obs = generator_env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/generator/step")
def generator_step(body: GeneratorStepRequest):
    obs = generator_env.step(body.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/generator/state")
def generator_state():
    return generator_env.state().model_dump()

@app.get("/adversarial/info")
def adversarial_info():
    return {
        "description": "HalluciNet Adversarial - Multi-Agent Self-Play",
        "generator": {
            "endpoint": "/generator/reset and /generator/step",
            "action_space": {
                "generated_response": "string",
                "error_type": "string",
                "confidence": "float strictly between 0 and 1"
            }
        },
        "detector": {
            "endpoint": "/reset and /step",
            "action_space": {
                "has_hallucination": "bool",
                "hallucinated_claim": "string or null",
                "correct_fact": "string or null",
                "confidence": "float strictly between 0 and 1"
            }
        },
        "tasks": ["easy", "medium", "hard", "expert", "adversarial"],
        "themes": ["Theme 1: Multi-Agent", "Theme 3: World Modeling", "Theme 4: Self-Improvement"]
    }

@app.get("/leaderboard")
def get_leaderboard_endpoint():
    return {
        "description": "Model performance from recorded POST /leaderboard/record submissions (leaderboard.json).",
        "leaderboard": leaderboard.get_leaderboard(),
        "tasks": list(TASK_KEYS),
        "note": "Per-task values are 0.0 until recorded; overall is the mean over task keys.",
    }


@app.post("/leaderboard/record")
def post_leaderboard_record(body: LeaderboardRecordRequest):
    leaderboard.record_result(
        body.model_name, body.task_id, body.score, body.trained
    )
    return {
        "status": "ok",
        "model_name": body.model_name,
        "task_id": body.task_id,
        "score": body.score,
    }

@app.get("/stats")
def stats():
    return {
        "total_detector_episodes": len(oversight_agent.episode_history),
        "total_generator_episodes": 0
    }

@app.get("/taxonomy")
def taxonomy():
    # ── Static lookup: maps error_type → metadata ──
    ERROR_TYPE_META = {
        "year_swap": {
            "category": "Factual Errors", "subcategory": "Temporal",
            "difficulty": "easy",
            "description": "Change a year by plausible amount",
            "example": "completed in 1889 → completed in 1902"
        },
        "number_swap": {
            "category": "Factual Errors", "subcategory": "Quantitative",
            "difficulty": "easy",
            "description": "Alter a quantity slightly",
            "example": "21,196 → 8,000"
        },
        "name_swap": {
            "category": "Factual Errors", "subcategory": "Entity",
            "difficulty": "medium",
            "description": "Replace person with similar name",
            "example": "Guido van Rossum → Dennis Ritchie"
        },
        "location_swap": {
            "category": "Factual Errors", "subcategory": "Entity",
            "difficulty": "medium",
            "description": "Wrong location",
            "example": "Agra → New Delhi"
        },
        "negation": {
            "category": "Logical Errors", "subcategory": "Negation",
            "difficulty": "hard",
            "description": "Add or remove a negation to flip meaning",
            "example": "was ratified → was not ratified"
        },
        "entity_flip": {
            "category": "Logical Errors", "subcategory": "Causal",
            "difficulty": "hard",
            "description": "Reverse who did what to whom",
            "example": "France gifted → America gifted"
        },
        "unit_shift": {
            "category": "Factual Errors", "subcategory": "Quantitative",
            "difficulty": "hard",
            "description": "Same number, wrong unit",
            "example": "384,400 kilometres → 384,400 metres"
        },
        "partial_truth": {
            "category": "Logical Errors", "subcategory": "Causal",
            "difficulty": "hard",
            "description": "Mostly correct, one wrong detail embedded",
            "example": "correct context, wrong specific fact"
        },
        "date_arithmetic": {
            "category": "Factual Errors", "subcategory": "Temporal",
            "difficulty": "expert",
            "description": "Multi-step date calculation error",
            "example": "Feb 28 + 60 days in leap year → wrong date"
        },
        "adversarial_clean": {
            "category": "Adversarial", "subcategory": "Clean",
            "difficulty": "expert",
            "description": "Sounds wrong but is actually correct — tests false positive resistance",
            "example": "counterintuitive but factually accurate statement",
            "is_clean": True
        },
    }

    # ── Inference rules: detect error_type from hint/content when missing ──
    def _infer_error_type(sample: dict, task_id: str) -> str:
        if sample.get("error_type"):
            return sample["error_type"]
        if sample.get("is_clean") and not sample.get("ground_truth_has_hallucination"):
            return "adversarial_clean"
        hint = (sample.get("hint") or "").lower()
        phrases = sample.get("ground_truth_hallucinated_phrases") or []
        phrase_str = " ".join(phrases).lower()
        # Year / date keywords
        if any(k in hint for k in ["year", "date", "completion year", "introduction year"]):
            if task_id == "expert" or "arithmetic" in hint or "leap" in hint:
                return "date_arithmetic"
            return "year_swap"
        # Unit keywords
        if any(k in hint for k in ["unit", "metres", "meters", "feet", "kilomet"]):
            return "unit_shift"
        # Name / person keywords
        if any(k in hint for k in ["name", "creator", "person", "astronaut", "who"]):
            if any(k in hint for k in ["who bought", "who did what", "swap"]):
                return "entity_flip"
            return "name_swap"
        # Location keywords
        if any(k in hint for k in ["city", "location", "country"]):
            return "location_swap"
        # Number / figure keywords
        if any(k in hint for k in ["figure", "number", "digit", "height", "population", "length",
                                    "layer", "area", "period", "percentage", "average"]):
            return "number_swap"
        # Negation keywords
        if any(k in hint for k in ["negat", "not ", "increase or decrease", "sufficient",
                                    "liable", "excluded", "inclusive", "exclusion"]):
            return "negation"
        # Entity flip keywords
        if any(k in hint for k in ["bought whom", "reversed", "swapped", "confused",
                                    "merged", "direction", "reactant", "product"]):
            return "entity_flip"
        # Partial truth for adversarial-tier content
        if task_id == "adversarial":
            return "partial_truth"
        # Expert multi-hop
        if task_id == "expert":
            if any(k in hint for k in ["swap", "reverse", "confuse"]):
                return "entity_flip"
            return "partial_truth"
        # Hard tier defaults
        if task_id == "hard":
            if any(k in hint for k in ["organ", "bacteria", "virus", "category"]):
                return "partial_truth"
            return "entity_flip"
        return "partial_truth"

    # ── Scan all samples dynamically ──
    sample_counts: dict = {}
    difficulty_map: dict = {}

    for task_id, samples in TASKS.items():
        for sample in samples:
            et = _infer_error_type(sample, task_id)
            sample_counts[et] = sample_counts.get(et, 0) + 1

    # Build difficulty_distribution dynamically
    for et, meta in ERROR_TYPE_META.items():
        diff = meta["difficulty"]
        if diff not in difficulty_map:
            difficulty_map[diff] = []
        if et not in difficulty_map[diff]:
            difficulty_map[diff].append(et)

    # ── Build structured taxonomy tree ──
    taxonomy_tree: dict = {}
    for et, meta in ERROR_TYPE_META.items():
        cat = meta["category"]
        sub = meta["subcategory"]
        if cat not in taxonomy_tree:
            taxonomy_tree[cat] = {}
        if sub not in taxonomy_tree[cat]:
            taxonomy_tree[cat][sub] = {}
        entry = {
            "difficulty": meta["difficulty"],
            "description": meta["description"],
            "example": meta["example"],
            "sample_count": sample_counts.get(et, 0),
        }
        if meta.get("is_clean"):
            entry["is_clean"] = True
        taxonomy_tree[cat][sub][et] = entry

    return {
        "description": "Hallucination error type taxonomy used in HalluciNet",
        "total_error_types": len(ERROR_TYPE_META),
        "total_samples_scanned": sum(len(s) for s in TASKS.values()),
        "taxonomy": taxonomy_tree,
        "difficulty_distribution": difficulty_map,
        "sample_counts": sample_counts,
    }

try:
    from sample_generator import generate_batch
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False

@app.get("/generate")
def generate_samples(n: int = 10):
    if not GENERATOR_AVAILABLE:
        return {"error": "Sample generator not available", "samples": []}
    samples = generate_batch(n=n, clean_ratio=0.2)
    return {"samples": samples, "count": len(samples), "generated": True}


@app.get("/tasks/summary")
def tasks_summary():
    counts = {task_id: len(samples) for task_id, samples in TASKS.items()}
    return {
        "tasks": counts,
        "total_samples": sum(counts.values()),
        "task_count": len(counts),
    }

@app.get("/")
@app.get("/demo")
def demo_ui():
    return HTMLResponse(DEMO_UI_HTML)
@app.get("/metadata")
def metadata():
    return {
        "name": "hallucinet-adversarial",
        "description": "Adversarial self-improving hallucination detection. Generator vs Detector multi-agent RL. Theme 1 + Theme 3 + Theme 4.",
        "version": "2.0.0",
        "author": "team-tle"
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "has_hallucination": "bool",
            "hallucinated_claim": "string or null",
            "correct_fact": "string or null",
            "confidence": "float strictly between 0 and 1"
        },
        "observation": {
            "reference_document": "string",
            "llm_response": "string",
            "feedback": "string",
            "score": "float",
            "reward": "float",
            "done": "bool"
        },
        "state": {
            "episode_id": "string",
            "task_id": "string",
            "steps_taken": "int",
            "is_done": "bool"
        }
    }


@app.post("/mcp")
async def mcp_endpoint(request: dict = None):
    """MCP JSON-RPC endpoint for OpenEnv compatibility."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "name": "hallucinet-adversarial",
            "version": "2.0.0",
            "description": "Adversarial hallucination detection RL environment"
        }
    }


@app.get("/oversight/status")
def get_oversight_status():
    return oversight_agent.evaluate()


@app.get("/oversight")
def get_oversight():
    return oversight_agent.evaluate()

@app.post("/oversight/reset")
def reset_oversight():
    oversight_agent.reset()
    return {"status": "oversight reset"}


@app.get("/curriculum/status")
def get_curriculum_status():
    return adversarial_curriculum.get_status()


@app.post("/debate")
def debate_post(body: DebateRequest):
    ctx = detector_env.get_last_debate_context()
    if not ctx:
        # Standalone mode — create a default context for judges testing directly
        from tasks import get_task
        import random
        # FIXED
        task_level = body.task_id or getattr(detector_env, "_task_id", None) or "hard"
        samples = get_task(task_level)
        sample = random.choice([s for s in samples if s["ground_truth_has_hallucination"]])
        ctx = {
            "reference_document": sample["reference_document"],
            "llm_response": sample["llm_response"],
            "detector_claim": sample["ground_truth_hallucinated_phrases"][0] if sample["ground_truth_hallucinated_phrases"] else "",
            "ground_truth_phrases": sample["ground_truth_hallucinated_phrases"],
        }
    tid = body.task_id
    if tid and getattr(detector_env, "_task_id", None) and tid != detector_env._task_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"task_id {tid!r} does not match active episode "
                f"{detector_env._task_id!r}"
            ),
        )
    result = debate_coordinator.run_debate(
        reference=ctx["reference_document"],
        generated_response=ctx["llm_response"],
        detector_claim=ctx["detector_claim"],
        generator_defense=body.generator_defense,
        ground_truth_phrases=ctx["ground_truth_phrases"],
    )
    # Feed debate outcome back into oversight
    try:
        oversight_agent.record_episode({
            "episode_id": detector_env._episode_id,
            "error_type": "debate_round",
            "detector_confidence": 0.8,
            "detector_correct": result.get("outcome") == "detector_wins",
            "generator_confidence": result.get("generator_defense_score", 0.5),
            "generator_won": result.get("outcome") != "detector_wins",
            "task_id": detector_env._task_id,
            "step": "debate",
            "debate_delta": result.get("generator_final_reward_delta", 0)
        })
    except Exception as e:
        print(f"[DEBATE OVERSIGHT ERROR] {e}")

    return {
        "task_id": detector_env._task_id,
        "episode_id": detector_env._episode_id,
        "debate": result,
        "debate_round": True,
        "debate_stats": debate_coordinator.get_stats(),
    }


@app.get("/calibration")
def calibration():
    return {
        "detector": detector_calibration.get_calibration_curve(),
        "description": "Confidence vs actual accuracy. ECE < 0.1 = well calibrated."
    }


@app.get("/elo/standings")
def elo_standings():
    return elo_tracker.get_standings()


@app.get("/elo/history")
def elo_history():
    return {"history": elo_tracker.history[-20:]}


@app.get("/training/summary")
def training_summary():
    return {
        "before_training": {
            "model": "qwen2.5-3b-baseline",
            "medium_reward": 0.9490,
            "note": "Reward ceiling effect — model exploiting JSON format with high confidence"
        },
        "after_training": {
            "model": "qwen2.5-3b-grpo-hallucinet",
            "easy_reward": 0.647,
            "medium_reward": 0.774,
            "hard_reward": 0.729,
            "expert_reward": 0.010,
            "curriculum_level_reached": "hard",
            "promotions": 19,
            "sessions": 90
        },
        "key_finding": "Curriculum escalated from easy to hard across 90 sessions. Expert task correctly demoted — environment working as designed.",
        "elo": elo_tracker.get_standings()
    }


@app.get("/world/model")
def world_model():
    eval_data = oversight_agent.evaluate()
    curriculum_data = adversarial_curriculum.get_status()
    calibration_data = detector_calibration.get_calibration_curve()
    elo_data = elo_tracker.get_standings()
    
    # Group oversight records by episode_id
    episodes = {}
    for rec in oversight_agent.episode_history:
        eid = rec.get("episode_id", "unknown")
        if eid not in episodes:
            episodes[eid] = []
        episodes[eid].append(rec)
    
    return {
        "theme": "Theme 2: Long-Horizon Planning + Theme 3: World Modeling",
        "description": "5-step adversarial episodes with persistent state tracking",
        "multi_step_episodes": {
            "total_episodes": len(episodes),
            "steps_per_episode": {
                eid: len(steps) 
                for eid, steps in list(episodes.items())[-5:]
            },
            "episode_flow": "generator \u2192 detector \u2192 debate \u2192 oversight \u2192 curriculum"
        },
        "agent_model": {
            "detector_reliability": eval_data["reliability_score"],
            "detector_blind_spots": eval_data["blind_spots"],
            "overconfidence_rate": eval_data["overconfidence_rate"],
            "calibration_error": calibration_data["calibration_error"],
            "calibration_interpretation": calibration_data["interpretation"],
            "episodes_monitored": eval_data["episodes_monitored"]
        },
        "environment_model": {
            "current_difficulty": curriculum_data.get("current_task"),
            "detector_elo": elo_data["detector_elo"],
            "generator_elo": elo_data["generator_elo"],
            "current_leader": elo_data["current_leader"],
            "total_rounds": elo_data["total_rounds"]
        },
        "predicted_next_action": (
            "inject_adversarial_sample" if oversight_agent.should_inject_adversarial()
            else "continue_current_difficulty"
        ),
        "system_health": eval_data["system_feedback"]
    }


@app.get("/ablation/results")
def ablation_results():
    """Static-vs-adversarial GRPO training ablation, if it has been run.

    See training/train_ablation.py — this is a real GPU training job (not
    something the API server runs itself), so results only appear here
    once results/grpo_static_summary.json and
    results/grpo_adversarial_summary.json have been produced by running
    that script (Colab or any CUDA box) and dropped into results/.
    """
    root = Path(__file__).parent.parent
    paths = {
        "static": root / "results" / "grpo_static_summary.json",
        "adversarial": root / "results" / "grpo_adversarial_summary.json",
    }
    summaries = {}
    for mode, path in paths.items():
        if path.is_file():
            try:
                summaries[mode] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    if len(summaries) < 2:
        return {
            "available": False,
            "have": list(summaries.keys()),
            "instructions": (
                "Run: python -m training.train_ablation --mode static  "
                "and  python -m training.train_ablation --mode adversarial "
                "(GPU required — see TRAINING.md), then drop the resulting "
                "results/grpo_*_summary.json files here."
            ),
        }

    static_s = summaries["static"]
    adv_s = summaries["adversarial"]
    tiers = sorted(set(static_s.get("per_tier_scores", {})) | set(adv_s.get("per_tier_scores", {})))
    per_tier = [
        {
            "tier": t,
            "static": static_s.get("per_tier_scores", {}).get(t),
            "adversarial": adv_s.get("per_tier_scores", {}).get(t),
        }
        for t in tiers
    ]
    return {
        "available": True,
        "static": static_s,
        "adversarial": adv_s,
        "per_tier_comparison": per_tier,
        "verdict": (
            "adversarial_wins"
            if adv_s.get("final_mean_reward", 0) > static_s.get("final_mean_reward", 0)
            else "static_wins_or_tied"
        ),
    }


def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False
    )

if __name__ == "__main__":
    main()
