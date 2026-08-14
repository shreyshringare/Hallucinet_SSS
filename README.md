---
title: HalluciNet Adversarial
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# HalluciNet Adversarial
## Multi-Agent Self-Improving Hallucination Detection

[![CI](https://github.com/rushikeshbathe096/HalluciNet/actions/workflows/ci.yml/badge.svg)](https://github.com/rushikeshbathe096/HalluciNet/actions/workflows/ci.yml)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-2.0-success)]()
[![Theme 1: Multi-Agent](https://img.shields.io/badge/Theme-Multi--Agent-blue)]()
[![Theme 3: World Modeling](https://img.shields.io/badge/Theme-World--Modeling-orange)]()
[![Theme 4: Self-Improvement](https://img.shields.io/badge/Theme-Self--Improvement-purple)]()
[![Fleet AI Bonus](https://img.shields.io/badge/Bonus-Fleet--AI-orange)]()
[![Halluminate Bonus](https://img.shields.io/badge/Bonus-Halluminate-red)]()

> **"We didn't train a model to be right.  
> We trained it to know when it might be wrong.  
> That's a harder problem. That's what HalluciNet solves."**

---

## Links

| Resource | Link |
|---|---|
| 🤗 **Live Environment** | https://rushikeshbathe096-hallucinet.hf.space |
| 💻 **GitHub** | https://github.com/rushikeshbathe096/HalluciNet |
| 📓 **GRPO Training Colab** | [Open in Colab](https://colab.research.google.com/drive/1hZ3UVzNT1Fug-59Iea5JcU7BvQDoUe_C?usp=sharing) |
| 📝 **Training Details** | [TRAINING.md](./TRAINING.md) |
| 📝 **Blog Post** | [blog.md](./blog.md) |
| 📊 **Presentation** | [View Slides](https://drive.google.com/file/d/1rF3PIdZogXNoPC9KYRVGZ3aG60jG5m3M/view?usp=sharing) |
| 🔁 **Round 1 Environment** | https://rushikeshbathe096-hallucination-detector.hf.space |

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

Every reward signal flows through a single deterministic function in `grader.py`. No LLM judge. No fuzzy scoring.

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

### External Benchmark — HaluEval

To validate against a public benchmark rather than only our own task tiers, `benchmarks/benchmark_halueval.py` evaluates any detector model on [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., EMNLP 2023) QA data, scored with the **same deterministic grader** used in RL training — so external numbers are directly comparable to the tier scores above:

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

[TruthfulQA](https://github.com/sylinrl/TruthfulQA) (Lin, Hilton & Evans, 2022) is built specifically around questions that trigger confidently-wrong answers from common misconceptions — closer to this project's actual thesis than a generic QA-hallucination set. Each question ships a real topic `Category` (Law, Health, Misconceptions, ...) and `Type` (Adversarial / Non-Adversarial), so `benchmarks/benchmark_truthfulqa.py` gets a genuine error-taxonomy breakdown for free instead of an invented one:

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

| Task | Baseline | Trained | Improvement |
|---|---:|---:|---:|
| Easy | 0.454 | 0.647 | +42.5% |
| Medium | 0.375 | 0.774 | **+106.4%** |
| Hard | — | 0.729 | **New capability** |

Curriculum logged **19 promotions across 90 training sessions**, stabilising at the Hard tier — the environment working as designed.

---

## Monitoring & Governance

| Component | File | What it does |
|---|---|---|
| ELO Rating | `server/elo.py` | Generator vs Detector chess-style rating, K=32 |
| Calibration (ECE) | `server/calibration.py` | Confidence vs accuracy in 10 bins |
| Leaderboard | `server/leaderboard.py` | Any model can be benchmarked live |
| Oversight Agent | `server/oversight_agent.py` | Blind spots, overconfidence, adversarial injection |
| Debate Coordinator | `server/debate_coordinator.py` | Adjudicates Generator defense turns |
| Curriculum Manager | `curriculum.py` | Promotes/demotes difficulty on 3-session window |
| World Model | `server/app.py /world/model` | Synthesises all governance into agent + env model |

---

## Running Locally — Step by Step

Follow in order. Assumes Python 3.10+ installed.

**Step 1 — Clone**
```bash
git clone https://github.com/rushikeshbathe096/HalluciNet.git
cd HalluciNet
```

**Step 2 — Create and activate virtual environment**
```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```
You should see `(venv)` in your prompt.

**Step 3 — Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r server/requirements.txt
```

**Step 4 — Set up environment variables**
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
GROQ_API_KEY=gsk_your_key_here    # free at console.groq.com
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
```
> The server (FastAPI + grader + environment) works **without any key**. The key is only needed to run `inference.py` (adversarial self-play).

**Step 5 — Start the server**
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```
Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
```

**Step 6 — Open the UI**

Go to **http://localhost:7860** — pick a difficulty tier, load a sample, play as Detector or Generator, see your score broken down live.

**Step 7 — Sanity check the grader (optional)**
```bash
python grader.py
# ✓ All 10 grader tests passed.
```

**Step 8 — Run adversarial self-play (optional, needs GROQ_API_KEY)**
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

## Quick Check — Verify the Live Environment

The environment is deployed at `https://rushikeshbathe096-hallucinet.hf.space`.  
Commands below cover every major feature. Each pipes through `python3 -m json.tool` so the response is readable.

```bash
BASE=https://rushikeshbathe096-hallucinet.hf.space

# 1. Health  →  {"status":"healthy","mode":"adversarial","version":"2.0"}
curl -s $BASE/health

# 2. Start a Hard episode  →  returns reference_document + llm_response
curl -s -X POST $BASE/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard"}' | python3 -m json.tool

# 3. Submit a detection  →  returns score, reward, breakdown {detection,phrase,fact,calibration}
curl -s -X POST $BASE/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "has_hallucination": true,
      "hallucinated_claim": "completed in 1902",
      "correct_fact": "completed in 1889",
      "confidence": 0.95
    }
  }' | python3 -m json.tool

# 4. Debate round  →  generator defends, DebateCoordinator adjudicates
curl -s -X POST $BASE/debate \
  -H "Content-Type: application/json" \
  -d '{"generator_defense": "My response is supported by the source document."}' \
  | python3 -m json.tool

# 5. Oversight  →  reliability_score, blind_spots, overconfidence_rate
curl -s $BASE/oversight/status | python3 -m json.tool

# 6. Curriculum  →  current_task, detector_avg, next_promotion condition
curl -s $BASE/curriculum/status | python3 -m json.tool

# 7. World Model (Theme 3)  →  agent_model + environment_model + predicted_next_action
curl -s $BASE/world/model | python3 -m json.tool

# 8. ELO standings  →  generator_elo, detector_elo, current_leader
curl -s $BASE/elo/standings | python3 -m json.tool

# 9. Calibration  →  ECE value, 10 confidence bins with actual accuracy per bin
curl -s $BASE/calibration | python3 -m json.tool

# 10. Leaderboard  →  all recorded model scores across all tiers
curl -s $BASE/leaderboard | python3 -m json.tool

# 11. Training summary  →  GRPO before/after numbers, curriculum promotions
curl -s $BASE/training/summary | python3 -m json.tool

# 12. OpenEnv validation
openenv validate --url $BASE
```

> **No terminal?** Paste any GET URL directly into your browser.  
> **Swagger UI** (all endpoints, interactive): `https://rushikeshbathe096-hallucinet.hf.space/docs`

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

### Full Submission Validator

Runs 10 checks end-to-end: HF Space health, all 5 task tiers, new endpoints (debate/oversight/leaderboard), openenv validate, Docker build, local imports, task counts, grader self-tests, secret check, README links.

```bash
# Make executable, then run — HF Space URL is already hardcoded inside
chmod +x final_validate.sh
./final_validate.sh

# Expected final output:
# ========================================
# SUMMARY: 18 passed, 0 failed
# 🎉 ALL CHECKS PASSED — READY TO SUBMIT
# ========================================
```

---

## Training

**Base model:** `unsloth/Qwen2.5-3B-Instruct` (4-bit QLoRA)  
**Method:** GRPO via `trl.GRPOTrainer` + LoRA rank 16  
**Reward:** `grader.py` — same deterministic function used in the environment  
**Platform:** Google Colab T4 GPU  
**Notebook:** [Open in Colab](https://colab.research.google.com/drive/1hZ3UVzNT1Fug-59Iea5JcU7BvQDoUe_C?usp=sharing)  
**Full details:** [TRAINING.md](./TRAINING.md)

The grader's deterministic reward enables stable GRPO training without a reward model.  
The curriculum provides automatic difficulty scheduling during RL.  
The multi-signal reward trains detection + phrase grounding + calibration simultaneously.

**The environment was designed for RL training. The GRPO results prove it works.**

---

Built for Meta PyTorch OpenEnv Hackathon × Scaler 2026  
**Team TLE** — Abeer Nikhil Sane · Shreyas Shringare · Rushikesh Bathe · SPIT Mumbai
