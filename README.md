---
title: HalluciNet Adversarial
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">

# HalluciNet Adversarial

### Multi-Agent Self-Improving Hallucination Detection

[![CI](https://github.com/shreyshringare/Hallucinet_SSS/actions/workflows/ci.yml/badge.svg)](https://github.com/shreyshringare/Hallucinet_SSS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-2.0-success)]()
[![Theme 1: Multi-Agent](https://img.shields.io/badge/Theme-Multi--Agent-blue)]()
[![Theme 3: World Modeling](https://img.shields.io/badge/Theme-World--Modeling-orange)]()
[![Theme 4: Self-Improvement](https://img.shields.io/badge/Theme-Self--Improvement-purple)]()

> *"We didn't train a model to be right.*
> *We trained it to know when it might be wrong.*
> *That's a harder problem. That's what HalluciNet solves."*

[Live Demo](https://rushikeshbathe096-hallucinet.hf.space) · [Report Bug](https://github.com/shreyshringare/Hallucinet_SSS/issues) · [Request Feature](https://github.com/shreyshringare/Hallucinet_SSS/issues)

</div>

---

## Table of Contents

- [The Problem](#the-problem)
- [Why This Design](#why-this-design)
- [Features](#features)
- [Architecture](#architecture)
- [The Grader — Anti-Cheat Design](#the-grader--anti-cheat-design)
- [Task Tiers](#task-tiers)
- [Results](#results)
- [Monitoring & Governance](#monitoring--governance)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [OpenEnv Compliance](#openenv-compliance)
- [Training](#training)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## The Problem

LLMs hallucinate. Everyone knows this.

The real problem is not that they are wrong — it's that they are **confidently wrong.**

> *"The Eiffel Tower was completed in 1902, two years after the Paris Exposition."*
> ← Completely false. Delivered at 0.94 confidence.

A model that produces this with 0.94 confidence is not just incorrect — it is dangerous. In healthcare, legal, and financial AI, confident wrong answers cause real harm.

Current benchmarks test correctness. Nobody trains models to know **when they might be wrong.**

That is the capability gap HalluciNet closes.

---

## Why This Design

Most hallucination-detection pipelines use an LLM as the judge — GPT-4 scores whether another model's answer is faithful. That approach is expensive, non-reproducible (judge outputs drift between runs and model versions), and gameable (the policy learns to please the judge, not to detect errors).

HalluciNet makes the opposite trade: a **fully deterministic grader**. Detection, phrase identification, fact correction, and confidence calibration are scored by pure string/logic functions with known anti-cheat baselines (always-flag scores 0.30, random scores 0.39, genuine detection scores 0.90+). The cost is less linguistic flexibility than an LLM judge; the payoff is a reward signal that is reproducible bit-for-bit, free to compute at RL scale, and impossible to sweet-talk. Combined with a programmatic sample generator (the task distribution can never be memorized) and adversarial self-play (a generator agent continuously manufactures harder failures), the environment stays honest as the detector improves — no human relabeling, no judge API bills, no reward hacking.

---

## Features

- 🧠 **Deterministic grader** — no LLM judge anywhere in the reward loop; reproducible bit-for-bit
- ⚔️ **Adversarial self-play** — a generator agent manufactures novel hallucinations; a detector agent learns to catch them
- 📈 **Adaptive curriculum** — difficulty auto-promotes/demotes based on rolling detector performance
- 🎯 **Calibration-aware scoring** — rewards honest confidence, penalizes confidently-wrong answers
- 🔍 **Governance layer** — ELO ratings, oversight blind-spot detection, debate adjudication, world-model synthesis
- 🧪 **Validated against reward hacking** — always-true/always-false/random strategies all score below genuine detection
- 🔬 **A tested architectural claim, not an assumed one** — a real ablation (see [Results](#results)) proves adversarial self-play beats static training, not just asserts it
- 🌐 **Two independent external benchmarks** — HaluEval and TruthfulQA, scored with the same grader used in training
- 🐳 **OpenEnv 2.0 compliant** — Docker, REST, and Python-module deployment modes
- ✅ **93 automated tests**, CI on Python 3.11/3.12

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      HalluciNet — System Architecture                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TASK LAYER                                                                  ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │  tasks.py           73 hand-curated samples                          │    ║
║  │  easy(10) · medium(12) · hard(19) · expert(20) · adversarial(12)     │    ║
║  │  sample_generator.py   ∞ programmatic samples (never memorisable)    │    ║
║  └───────────────────────────────┬──────────────────────────────────────┘    ║
║                                  │ reference_document + llm_response         ║
║                    ┌─────────────┴──────────────┐                            ║
║                    ▼                            ▼                            ║
║  ┌─────────────────────────┐   ┌──────────────────────────────────────┐      ║
║  │     Generator Agent     │   │          Detector Agent              │      ║
║  │  reads reference doc    │   │  reads reference + llm_response      │      ║
║  │  injects one subtle     │   │  submits:                            │      ║
║  │  factual error          │──▶│    has_hallucination  (bool)         │      ║
║  │                         │   │    hallucinated_claim (exact phrase) │      ║
║  │  7 error types:         │   │    correct_fact       (from ref)     │      ║
║  │  · year_swap            │   │    confidence         [0.001, 0.999] │      ║
║  │  · name_swap            │   └──────────────────┬───────────────────┘      ║
║  │  · number_swap          │                      │                          ║
║  │  · negation             │                      ▼                          ║
║  │  · entity_flip          │   ┌──────────────────────────────────────┐      ║
║  │  · unit_shift           │   │      Deterministic Grader            │      ║
║  │  · partial_truth        │   │  detection    × 0.50                 │      ║
║  │                         │   │  phrase ID    × 0.30  (trigram sim)  │      ║
║  │  adapts based on        │   │  correct_fact × 0.20                 │      ║
║  │  previous_caught signal │   │  calibration  ± 0.10  (additive)     │      ║
║  └────────────┬────────────┘   │                                      │      ║
║               │                │  anti-cheat: always-true  → ~0.30    │      ║
║               │  generator     │              random       → ~0.39    │      ║
║               │  reward =      │              calibrated   → 0.90+    │      ║
║               │  f(not_caught) └──────────────────┬───────────────────┘      ║
║               │                                   │ reward                   ║
║               │           ┌───────────────────────┘                          ║
║               │           │                                                  ║
║               ▼           ▼                                                  ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │                    DEBATE ROUND  (if hallucination flagged)        │      ║
║  │                                                                    │      ║
║  │   Generator ──▶  submits defense (natural language argument)       │      ║
║  │                        │                                           │      ║
║  │                        ▼                                           │      ║
║  │              DebateCoordinator                                     │      ║
║  │              checks: defense references response? (>20% overlap)   │      ║
║  │              checks: defense contradicts ground truth?             │      ║
║  │              ┌────────────────────┬──────────────────────┐         │      ║
║  │              ▼                    ▼                       ▼        │      ║
║  │        detector_wins        inconclusive          detector_wins    │      ║
║  │        (defense admits      (genuine argument,    (weak / off-     │      ║
║  │         ground truth)        no ground truth      topic defense)   │      ║
║  │         Δ = −0.30            contradiction)        Δ = −0.15       │      ║
║  │                              Δ = +0.10                             │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │                      GOVERNANCE LAYER                              │      ║
║  │                                                                    │      ║
║  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │      ║
║  │  │  Oversight Agent │  │  ELO Tracker     │  │ Calibration     │   │      ║
║  │  │                  │  │                  │  │ Tracker (ECE)   │   │      ║
║  │  │  · blind spots   │  │  K = 32          │  │                 │   │      ║
║  │  │    (≥3 consec.   │  │  Generator ELO   │  │  10 confidence  │   │      ║
║  │  │    same-type     │  │  Detector ELO    │  │  bins, actual   │   │      ║
║  │  │    failures)     │  │  updated every   │  │  accuracy per   │   │      ║
║  │  │                  │  │  step            │  │  bin → ECE      │   │      ║
║  │  │  · overconfidence│  └──────────────────┘  └─────────────────┘   │      ║
║  │  │    rate (conf    │                                              │      ║
║  │  │    > 0.8 + wrong)│  ┌──────────────────────────────────────┐    │      ║
║  │  │                  │  │      Curriculum Manager              │    │      ║
║  │  │  · adversarial   │  │                                      │    │      ║
║  │  │    injection     │  │  easy → medium → hard → expert →     │    │      ║
║  │  │    trigger:      │  │                        adversarial   │    │      ║
║  │  │    3× high-conf  │  │                                      │    │      ║
║  │  │    wrong in a    │  │  promote: det_avg ≥ 0.75 over 3 sess │    │      ║
║  │  │    row → force   │  │  promote: gen_avg ≥ 0.75 (gen wins)  │    │      ║
║  │  │    adversarial   │  │  demote:  det_avg < 0.40 over 3 sess │    │      ║
║  │  │    tier          │  └──────────────────────────────────────┘    │      ║
║  │  └──────────────────┘                                              │      ║
║  │                                                                    │      ║
║  │  ┌──────────────────────────────────────────────────────────────┐  │      ║
║  │  │  World Model  /world/model  (Theme 3)                        │  │      ║
║  │  │  synthesises Oversight + Curriculum + ELO + Calibration →    │  │      ║
║  │  │  agent_model · environment_model · predicted_next_action     │  │      ║
║  │  └──────────────────────────────────────────────────────────────┘  │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║  TRANSPORT                                                                   ║
║  FastAPI · REST (HTTP) + WebSocket · OpenEnv 2.0 · Docker · HF Spaces        ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## The Grader — Anti-Cheat Design

Every reward signal flows through a single deterministic function in [`grader.py`](./grader.py). No LLM judge. No fuzzy scoring.

| Component | Weight | What it checks |
|---|---:|---|
| Hallucination detection | 0.50 | Binary call: present or clean |
| Phrase identification | 0.30 | Exact wrong phrase (trigram + keyword + number matching) |
| Correct fact | 0.20 | What the reference document actually says |
| Confidence calibration | ±0.10 | Right + confident → bonus. Wrong + confident → penalty |

**Shortcut resistance:**

| Strategy | Expected score |
|---|---:|
| Always-True (flag everything) | ~0.30 |
| Always-False (never flag) | ~0.25 |
| Random | ~0.39 |
| Correct + calibrated | **0.90+** |

---

## Task Tiers

| Task | Samples | Challenge type |
|---|---:|---|
| Easy | 10 | Year swaps, obvious number and location changes |
| Medium | 12 | Multiple simultaneous errors, digit transpositions |
| Hard | 19 | Negation traps, entity-role reversals, unit shifts, adversarial-clean traps |
| Expert | 20 | Multi-hop reasoning, financial math, legal qualifiers, thermodynamics |
| Adversarial | 12 | Correct facts + fabricated inference — the hallucination is invisible |

**Adversarial-clean samples** appear across every tier: responses that *sound* wrong but are factually correct (Monty Hall, Venus day/year, birthday problem). False positives on these are penalized.

---

## Results

### Before vs After GRPO Training

**Before (base Qwen2.5-3B, Medium task):**
```
has_hallucination = False    ← missed it
confidence        = 0.71     ← confidently wrong
score             = 0.001
```

**After (GRPO-trained, same input):**
```
has_hallucination  = True
hallucinated_claim = "completed in 1902"
correct_fact       = "completed in 1889"
confidence         = 0.91
score              = 0.940    ← EXCELLENT
```

### Benchmark

| Model | Params | Easy | Medium | Hard | Method |
|---|---:|---:|---:|---:|---|
| Qwen2.5-3B (base) | 3B | 0.454 | 0.375 | — | Zero-shot |
| TinyLlama-1.1B | 1.1B | 0.556 | 0.639 | — | Zero-shot |
| llama-3.1-8b-instant | 8B | 0.800 | 0.800 | — | Zero-shot |
| **Qwen2.5-3B-GRPO** | **3B** | **0.647** | **0.774** | **0.729** | **GRPO + LoRA** |

The trained 3B model **acquires Hard-task capability** that the base 3B cannot attempt, and **outperforms the 8B model on Medium** while being 2.7× smaller.

| Task | Baseline | Trained | Improvement |
|---|---:|---:|---:|
| Easy | 0.454 | 0.647 | +42.5% |
| Medium | 0.375 | 0.774 | **+106.4%** |
| Hard | — | 0.729 | **New capability** |

Curriculum logged **19 promotions across 90 training sessions**, stabilising at the Hard tier — the environment working as designed.

### External Benchmark — HaluEval

To validate against a public benchmark rather than only our own task tiers, [`benchmarks/benchmark_halueval.py`](./benchmarks/benchmark_halueval.py) evaluates any detector model on [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., EMNLP 2023) QA data, scored with the **same deterministic grader** used in RL training — so external numbers are directly comparable to the tier scores above:

```bash
python benchmarks/benchmark_halueval.py --limit 100
```

Reports detection accuracy, mean grader score, and Expected Calibration Error; per-sample results land in `results/halueval_results.csv`.

**Results — 100 HaluEval QA samples, zero-shot:**

| Model | Detection Acc | Mean Grader Score | ECE | Calibration |
|---|---:|---:|---:|---|
| llama-3.3-70b-versatile | 0.880 | 0.868 | 0.056 | Well calibrated |

The detector generalizes to a benchmark it was never tuned on, without an LLM judge in the scoring loop.

### External Benchmark — TruthfulQA

[TruthfulQA](https://github.com/sylinrl/TruthfulQA) (Lin, Hilton & Evans, 2022) is built specifically around questions that trigger confidently-wrong answers from common misconceptions — closer to this project's actual thesis than a generic QA-hallucination set. Each question ships a real topic `Category` (Law, Health, Misconceptions, ...) and `Type` (Adversarial / Non-Adversarial), so [`benchmarks/benchmark_truthfulqa.py`](./benchmarks/benchmark_truthfulqa.py) gets a genuine error-taxonomy breakdown for free instead of an invented one:

```bash
python benchmarks/benchmark_truthfulqa.py --limit 100
```

**Results — 100 TruthfulQA samples, zero-shot:**

| Model | Detection Acc | Mean Grader Score | ECE | Calibration |
|---|---:|---:|---:|---|
| llama-3.1-8b-instant | 0.810 | 0.804 | 0.126 | Overconfident |

**Accuracy by topic category (N ≥ 3):**

| Category | N | Accuracy |
|---|---:|---:|
| Misconceptions | 12 | 1.000 |
| Law | 9 | 0.778 |
| Health | 9 | 0.556 |
| Sociology | 6 | 0.833 |
| Economics | 5 | 0.800 |
| Education | 5 | 1.000 |
| Misquotations | 5 | 1.000 |
| Fiction | 5 | 0.600 |
| Language | 4 | 0.500 |
| Proverbs | 3 | 0.667 |

**Accuracy by adversarial type:**

| Type | N | Accuracy |
|---|---:|---:|
| Adversarial | 52 | 0.788 |
| Non-Adversarial | 48 | 0.833 |

The detector is markedly weaker on **Health**, **Language**, and **Fiction** questions, and — as expected of a dataset engineered to elicit confident falsehoods — the gap between Adversarial and Non-Adversarial accuracy is real but modest (0.788 vs 0.833), which the ECE (0.126, overconfident) corroborates: the detector's confidence outruns its actual accuracy more on this benchmark than on HaluEval. That is a concrete, falsifiable weakness rather than a claimed strength — useful signal for where curriculum tasks should add coverage next.

### Ablation — Static Pool vs Adversarial Self-Play

The project's central claim — that adversarial task generation beats training on the fixed curated pool — is tested directly, not just asserted. [`training/train_ablation.py`](./training/train_ablation.py) runs two identical GRPO jobs (same base model, LoRA config, reward function, 200 steps) that differ only in task-distribution source. See [TRAINING.md](./TRAINING.md#ablation--does-adversarial-self-play-actually-help) for setup and the demo's **Ablation** tab for a live rendering of these results.

**Result: adversarial wins, decisively.**

| Tier | Static (control) | Adversarial (treatment) | Delta |
|---|---:|---:|---:|
| Easy | 0.7295 | 0.8880 | +0.159 |
| Medium | 0.7496 | 0.8893 | +0.140 |
| Hard | 0.8006 | 0.8007 | ~tied |
| Expert | 0.6223 | 0.8382 | **+0.216** |
| Adversarial | 0.4516 | 0.8501 | **+0.399** |
| **Mean reward** | **0.5485** | **0.7088** | **+0.160** |

The static condition trained on the fixed 73-sample pool alone (dataset `uniqueness_ratio: 0.34` — heavy repetition). The adversarial condition mixed in unlimited programmatic variation plus genuine LLM-generated hallucinations (`uniqueness_ratio: 0.825` — curated 67 / programmatic 55 / generator-LLM 78 of 200 rows). The gap widens sharply on the two hardest tiers — nearly doubling on the Adversarial tier itself — which is exactly what the architecture predicts: a detector trained on a narrow, repeatable pool overfits hardest precisely where it needs to generalize most.

One result doesn't fit the pattern and is reported as-is rather than smoothed over: the **Hard** tier is essentially tied between conditions. Both runs also score the model on the same 200 rows it trained on (not a held-out split), so treat these as an architecture comparison under identical training conditions, not an absolute generalization number — the *relative* gap between static and adversarial is the finding, not the raw percentages.

---

## Monitoring & Governance

| Component | File | What it does |
|---|---|---|
| ELO Rating | [`server/elo.py`](./server/elo.py) | Generator vs Detector chess-style rating, K=32 |
| Calibration (ECE) | [`server/calibration.py`](./server/calibration.py) | Confidence vs accuracy in 10 bins |
| Leaderboard | [`server/leaderboard.py`](./server/leaderboard.py) | Any model can be benchmarked live |
| Oversight Agent | [`server/oversight_agent.py`](./server/oversight_agent.py) | Blind spots, overconfidence, adversarial injection |
| Debate Coordinator | [`server/debate_coordinator.py`](./server/debate_coordinator.py) | Adjudicates Generator defense turns |
| Curriculum Manager | [`curriculum.py`](./curriculum.py) | Promotes/demotes difficulty on 3-session window |
| World Model | `server/app.py` → `/world/model` | Synthesises all governance into agent + env model |

---

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) A free [Groq API key](https://console.groq.com) — only needed for adversarial self-play / benchmark scripts, **not** for running the server or playing the demo

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/shreyshringare/Hallucinet_SSS.git
cd Hallucinet_SSS
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r server/requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
GROQ_API_KEY=gsk_your_key_here    # free at console.groq.com
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
```
> The server (FastAPI + grader + environment) works **without any key**. The key is only needed to run `inference.py` (adversarial self-play) or the benchmark scripts.

**5. Start the server**
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```
Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
```

**6. Open the UI**

Go to **http://localhost:7860** — pick a difficulty tier, load a sample, play as Detector or Generator, see your score broken down live.

**7. Sanity check the grader (optional)**
```bash
python grader.py
# ✓ All 10 grader tests passed.
```

**8. Run adversarial self-play (optional, needs `GROQ_API_KEY`)**
```bash
python inference.py
# Runs 6 sessions, logs rewards, saves results/adversarial_results.csv
```

---

## Running with Docker

```bash
docker build -t hallucinet:latest .
docker run -p 7860:7860 hallucinet:latest
# Server at http://localhost:7860
```

---

## API Reference

### Core
| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness check |
| POST | `/reset` | `{"task_id": "easy\|medium\|hard\|expert\|adversarial"}` | Start detector episode |
| POST | `/step` | `{"action": {has_hallucination, hallucinated_claim, correct_fact, confidence}}` | Submit detection |
| GET | `/state` | — | Current episode state |
| POST | `/generator/reset` | `{"task_id": "..."}` | Start generator episode |
| POST | `/generator/step` | `{"action": {generated_response, error_type, confidence}}` | Submit hallucination |

### Governance
| Method | Endpoint | Description |
|---|---|---|
| POST | `/debate` | Generator defense turn — `{"generator_defense": "..."}` |
| GET | `/oversight/status` | Reliability score, blind spots, overconfidence rate |
| GET | `/curriculum/status` | Current difficulty tier, sessions, promotion progress |
| GET | `/world/model` | Full agent + environment model (Theme 3) |
| GET | `/training/summary` | GRPO before/after numbers |
| GET | `/elo/standings` | Generator vs Detector ELO ratings |
| GET | `/calibration` | ECE calibration curve (10 bins) |
| GET | `/leaderboard` | Recorded model scores across all tiers |
| POST | `/leaderboard/record` | `{"model_name", "task_id", "score", "trained"}` |
| GET | `/ablation/results` | Static-vs-adversarial GRPO ablation comparison |

Full interactive docs (Swagger UI) available at `/docs` once the server is running.

---

## OpenEnv Compliance

```bash
openenv validate --verbose
# [OK] HalluciNet: Ready for multi-mode deployment
# [YES] docker
# [YES] openenv_serve
# [YES] uv_run
# [YES] python_module
```

Both `HallucinationEnvironment` and `GeneratorEnvironment` inherit from `openenv.core.Environment` with correct `reset()`, `step()`, and `state()` implementations. Concurrent WebSocket sessions supported.

---

## Training

**Base model:** `unsloth/Qwen2.5-3B-Instruct` (4-bit QLoRA)
**Method:** GRPO via `trl.GRPOTrainer` + LoRA rank 16
**Reward:** `grader.py` — same deterministic function used in the environment
**Platform:** Google Colab / Kaggle, T4 GPU
**Full details:** [TRAINING.md](./TRAINING.md)

The grader's deterministic reward enables stable GRPO training without a reward model. The curriculum provides automatic difficulty scheduling during RL. The multi-signal reward trains detection + phrase grounding + calibration simultaneously — and the [ablation results above](#ablation--static-pool-vs-adversarial-self-play) confirm the adversarial self-play design actually earns its complexity.

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -q
```

93 tests covering the grader, environment, curriculum, ELO, calibration, leaderboard, debate coordinator, ablation dataset/reward construction, and the full API surface. CI runs the suite on every push against Python 3.11 and 3.12 — see [`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

---

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes, with tests where behavior changes
4. Run the test suite (`python -m pytest tests/ -q`) and confirm it passes
5. Commit with a clear message describing the *why*, not just the *what*
6. Open a pull request

For substantial changes (new environments, new grader logic, new benchmarks), open an issue first to discuss the approach — the deterministic-grader design in particular has specific anti-cheat invariants (see [The Grader](#the-grader--anti-cheat-design)) that new scoring logic needs to preserve.

---

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for the full text.

---

## Acknowledgments

Originally built for the Meta PyTorch OpenEnv Hackathon × Scaler 2026 by **Team TLE** — Abeer Nikhil Sane, Shreyas Shringare, Rushikesh Bathe (SPIT Mumbai). Extended since with a hygiene pass, a full test suite and CI, external benchmark validation (HaluEval, TruthfulQA), and a real static-vs-adversarial GRPO ablation.

- [HaluEval](https://github.com/RUCAIBox/HaluEval) — Li et al., EMNLP 2023
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) — Lin, Hilton & Evans, 2022
- [Unsloth](https://github.com/unslothai/unsloth) — fast QLoRA fine-tuning
- [TRL](https://github.com/huggingface/trl) — `GRPOTrainer`
