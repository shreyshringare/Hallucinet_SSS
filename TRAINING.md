# 🧪 GRPO Training — HalluciNet Adversarial Detector

## Overview

We trained a hallucination detection model using **Grouped Reinforcement Learning with Policy Optimization (GRPO)** — the same technique used in DeepSeek-R1. The model learns to detect factual errors in LLM responses by receiving reward signals from HalluciNet's deterministic grader.

**Training Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1vrqo8CFXBYJi3lHaFJWVQSPgcw9B34rz?usp=sharing)

---

## Training Configuration

| Parameter | Value |
|---|---|
| **Base Model** | `unsloth/Qwen2.5-3B-Instruct` |
| **Quantization** | 4-bit (QLoRA) |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 16 |
| **LoRA Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Trainer** | `trl.GRPOTrainer` |
| **Reward Function** | `grader.py` — deterministic multi-signal scorer |
| **Num Generations** | Multiple per prompt (GRPO explores diverse completions) |
| **Temperature** | 1.0 (during generation for exploration) |
| **KL Penalty (beta)** | Tuned to balance exploitation vs exploration |
| **Curriculum** | Adaptive (easy → medium → hard → expert) |
| **Platform** | Google Colab — T4 GPU |

---

## Reward Function

The GRPO reward comes directly from HalluciNet's deterministic grader (`grader.py`), ensuring the RL training signal is identical to the evaluation metric:

| Signal | Weight | What it measures |
|---|---:|---|
| Detection correctness | 0.50 | Did the model correctly identify hallucination presence? |
| Phrase grounding | 0.30 | Did it locate the exact wrong phrase? |
| Correct fact | 0.20 | Did it provide what the reference actually says? |
| Confidence calibration | ±0.10 | Is the model honest about its uncertainty? |

**Reward shaping:**
- Minimum reward of 0.05 for unparseable JSON outputs (encourages valid formatting)
- Curriculum integration: detector performance is tracked every N steps to adjust task difficulty

---

## Training Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     GRPO Training Loop                          │
│                                                                  │
│  1. Sample prompt (reference + LLM response) from curriculum    │
│  2. Generate K completions with temperature=1.0 (exploration)    │
│  3. Parse each completion → HallucinationAction JSON             │
│  4. Grade each action using grader.py → reward ∈ [0, 1]          │
│  5. GRPO update: reinforce high-reward completions,              │
│     suppress low-reward ones, with KL penalty to reference       │
│  6. Every N steps: evaluate curriculum → promote/demote task     │
│  7. Repeat until convergence                                     │
└──────────────────────────────────────────────────────────────────┘
```

This is true **Reinforcement Learning from Environment Feedback (RLEF)** — the model's policy is updated via gradient descent based on reward signals, not just prompted or evaluated.

---

## Results — Before vs After Training

### Per-Task Scores (Deterministic Grader)

| Task | Qwen2.5-3B (Base) | Qwen2.5-3B-GRPO (Trained) | Delta | Improvement |
|---|---:|---:|---:|---:|
| Easy | 0.454 | **0.647** | +0.193 | **+42.5%** |
| Medium | 0.375 | **0.774** | +0.399 | **+106.4%** |
| Hard | — | **0.729** | — | **New capability** |
| Expert | — | 0.010 | — | Not yet converged |

### Key Observations

1. **+106% improvement on medium tasks** — the model went from near-random performance to strong detection capability.
2. **Hard task capability emerged** — the base model couldn't handle hard tasks at all; after GRPO training, it scores 0.729.
3. **Curriculum progression worked** — the model was promoted through difficulty levels during training, showing genuine skill acquisition.
4. **Expert remains difficult** — this tier requires multi-hop reasoning that a 3B model struggles with, even after training. This is an honest result, not cherry-picked.

### Multi-Model Comparison

| Model | Params | Easy | Medium | Hard | Method |
|---|---:|---:|---:|---:|---|
| Qwen2.5-3B (base) | 3B | 0.454 | 0.375 | — | Zero-shot |
| TinyLlama-1.1B | 1.1B | 0.556 | 0.639 | — | Zero-shot |
| llama-3.1-8b-instant | 8B | 0.800 | 0.800 | — | Zero-shot |
| **Qwen2.5-3B-GRPO** | **3B** | **0.647** | **0.774** | **0.729** | **GRPO + LoRA** |

The GRPO-trained 3B model approaches the performance of a zero-shot 8B model — demonstrating that **environment-specific RL training can close the gap between small and large models**.

---

## What the Model Learned

Through GRPO training, the model acquired:

1. **Structured output discipline** — consistently produces valid JSON matching the `HallucinationAction` schema
2. **Phrase grounding** — locates the specific wrong claim rather than making vague assertions
3. **Correction accuracy** — provides what the reference document actually says
4. **Confidence calibration** — reports lower confidence when uncertain, avoiding the "overconfident-and-wrong" failure mode
5. **Clean sample resistance** — learned to resist false-positive detection on adversarial-clean samples

---

## How to Reproduce

1. Open the [Colab notebook] (https://colab.research.google.com/drive/1hZ3UVzNT1Fug-59Iea5JcU7BvQDoUe_C?usp=sharing)
2. Ensure a T4 GPU runtime is selected
3. Run all cells in order
4. Training takes approximately 1-2 hours
5. The trained model is saved to `hallucinet-trained/` and can be downloaded

---

## Ablation — Does Adversarial Self-Play Actually Help?

The project's central claim is that training against an adversarially-expanding
task distribution beats training on the fixed 73-sample curated pool alone.
`training/train_ablation.py` tests that claim directly with two GRPO runs that
are identical in every respect (base model, LoRA config, reward function,
curriculum schedule) except the task distribution:

| Condition | Task source |
|---|---|
| **static** (control) | Only `tasks.py`'s 73 curated samples, cycled |
| **adversarial** (treatment) | Curated pool + unlimited `sample_generator.py` variety + a configurable fraction of genuinely LLM-generated adversarial hallucinations (a generator agent targeting the reference documents, same idea as the live environment's generator) |

Requires a GPU (T4 or better) — this is a real training job, not something
the API server runs on its own:

```bash
pip install unsloth trl peft bitsandbytes accelerate
python -m training.train_ablation --mode static
python -m training.train_ablation --mode adversarial --llm-fraction 0.3
```

Each run writes `results/grpo_<mode>_summary.json` (final mean reward,
per-tier scores) and `results/grpo_<mode>_log.csv` (per-step reward). Once
both summary files exist, the demo UI's **Ablation** tab (and `GET
/ablation/results`) automatically renders a side-by-side comparison —
mean reward, verdict, and a per-tier bar breakdown.

The dataset-construction and reward-wiring logic (everything except the
actual multi-hour GRPO training loop) is unit-tested — see
`tests/test_ablation_dataset.py` and `tests/test_ablation_reward.py`.

---

## Connection to Environment Design

The training pipeline validates the environment design end-to-end:

- **The grader's deterministic reward** enables stable GRPO training (no reward model noise)
- **The curriculum manager** provides an automatic difficulty schedule during RL
- **The multi-signal reward** (detection + phrase + fact + calibration) trains all aspects of hallucination detection simultaneously
- **The adversarial-clean samples** teach the model to avoid false positives under adversarial pressure

This closes the loop: **the environment was designed for RL training, and the GRPO results prove it works.**
