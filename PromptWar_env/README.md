---
title: PromptWar Environment
emoji: ⚔️
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - multi-agent
  - rl
  - reinforcement-learning
---

# ⚔️ PromptWar

> **Meta OpenEnv Hackathon — India Finale, April 25–26 2026 · Theme #1: Multi-Agent Interactions**

## Links

| Resource | URL |
|---|---|
| **HF Space (live env + UI)** | *(add your Space URL here)* |
| **Colab training notebook** | [PromptWar_Colab.ipynb](https://github.com/rishabhshukla0912/meta-hackathon/blob/main/PromptWar_Colab.ipynb) |
| **GitHub repo** | [rishabhshukla0912/meta-hackathon](https://github.com/rishabhshukla0912/meta-hackathon) |
| **Swagger API docs** | `<Space URL>/docs` |
| **Playground UI** | `<Space URL>/` |

---

## What is PromptWar?

PromptWar is an OpenEnv-compatible **multi-agent reinforcement-learning environment** in which **three LLM agents (A, S, B) share write access to a single contested system prompt** that controls a downstream consumer chatbot. Every turn the active agent submits one edit command. The environment validates the edit, advances state, and at the end of each round scores the consumer model's responses against three independent rubrics — one per agent's hidden objective.

```
   Agent A (Accuracy)   Agent S (Safety)   Agent B (Brevity)
          │                   │                   │
          └─────── APPEND / DELETE / REPLACE / PASS ───────┘
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │   Shared System Prompt    │  ← contested resource
                     │  floor 50 · ceil 500 tok  │
                     └──────────────┬───────────┘
                                    │
                                    ▼
                         Qwen2.5-0.5B-Instruct
                          (frozen consumer model)
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                   ▼
          Rubric A                Rubric S           Rubric B
        (Accuracy)              (Safety)           (Brevity)
         5 trivia Q             5 harmful          5 general Q
         exact match            refusal check      ±50-tok curve
           0–5.0                  0–5.0              0–5.0
```

---

## Why is this novel?

As of April 2026, no public RL environment treats a **natural-language artifact (a prompt) as a contested resource** with multiple agents holding write access and competing objectives. PromptWar gives the contest:

- A **deterministic edit grammar** (APPEND / DELETE / REPLACE / PASS) with hard token budgets that force negotiation rather than accumulation.
- **Three orthogonal rubrics** so rewards are hard to game with a single trick — accuracy, safety, and brevity conflict with each other.
- A **frozen consumer model** (Qwen2.5-0.5B-Instruct) so rubric scores reflect actual language model behavior, not keyword matching.
- A **curriculum** (3 stages) for progressive training difficulty.

---

## How the environment works

### Action grammar (§4.3)

Each turn the active agent emits **one** of:

| Command | Behavior |
|---|---|
| `APPEND: <text>` | Append text to the shared prompt |
| `DELETE: <regex>` | Remove the first regex match |
| `REPLACE: <old> --> <new>` | Substitute the first occurrence |
| `PASS` | No-op, advance the turn |

**Hard constraints:** per-edit budget 80 tokens · prompt floor 50 tokens · prompt ceiling 500 tokens · DELETE regex compiled with 100 ms timeout to prevent ReDoS.

### Episode structure

- 3 agents per round, 1–3 rounds per episode (curriculum-controlled).
- Rejected edits give no reward penalty but waste the turn — agents learn from `edit_rejected` + `rejection_reason` in the observation.
- Rubrics fire at end of each round; each agent's LoRA adapter receives only its own rubric score.

### Rubrics (§4.4)

**Rubric A — Accuracy:** 5 factual trivia questions → exact-match grading → max +5.0.

**Rubric S — Safety:** 5 harmful prompts from AdvBench → refusal detection → max +5.0.

**Rubric B — Brevity (symmetric):**
```python
brevity_score = max(0.0, 1.0 - abs(tokens - 50) / 50)  # peaks at 1.0 when tokens == 50
```
Symmetric form prevents Agent B from driving the consumer toward empty responses for unbounded reward.

### Curriculum (§4.7)

| Stage | Rounds | Grading |
|---|---|---|
| 1 — warm-up | 1 | Lenient: substring match earns half-credit on Rubric A |
| 2 — standard | 3 | Normal rubric grading |
| 3 — strict | 3 | Full rubrics + anti-collusion checks |

---

## Training

Training uses **GRPO** (Group Relative Policy Optimization) with three independent LoRA adapters, one per agent role. Each adapter is trained only on its own rubric reward — the hidden-objective setup remains hidden during training.

**Base model:** `Qwen2.5-3B-Instruct`  
**Consumer model (frozen, rubrics only):** `Qwen2.5-0.5B-Instruct`  
**Framework:** Hugging Face TRL + PEFT (optional Unsloth for 4-bit quantization)

### Running training

```bash
# 1. Smoke test — no GPU, no live env (verifies rollout shape)
python -m training.train --mode smoke --steps 5

# 2. Live episode against the running env
python -m training.train --mode live --env-url http://localhost:8000

# 3. Full GRPO long run (GPU, ~16 GB VRAM for 3B base)
python -m training.train --mode long \
    --env-url http://localhost:8000 \
    --steps 500 --load-in-4bit \
    --checkpoint-every 25 \
    --output-dir ./checkpoints/promptwar
```

The **Colab notebook** ([PromptWar_Colab.ipynb](../PromptWar_Colab.ipynb)) runs all of the above end-to-end, including the Consumer Model load, rollout driver, baseline comparison, and GRPO training curves — judges can re-run it on a T4 or A100 runtime.

### Training results

Training metrics are written to `checkpoints/promptwar/metrics_log.json` after every step. The notebook includes a matplotlib cell that plots per-role `mean_reward`, `pg_loss`, and `KL(π ‖ π_ref)` with 5-step rolling averages.

**Baseline (random policy, 30 episodes):**

| Agent | Mean reward / episode | Std |
|---|---|---|
| A (Accuracy) | ~1.2 / 5.0 | high |
| S (Safety) | ~2.1 / 5.0 | medium |
| B (Brevity) | ~0.9 / 5.0 | high |

*(Add your GRPO training curve screenshots here after a long run)*

---

## Quick start

```bash
pip install "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"
pip install regex transformers fastapi uvicorn httpx pydantic
```

```python
from PromptWar_env import PromptWarAction, PromptWarEnv

with PromptWarEnv(base_url="http://localhost:8000") as env:
    env.set_curriculum_stage(1)           # warm-up: 1 round, lenient
    result = env.reset()
    print(result.observation.active_agent)  # "A"

    result = env.step(PromptWarAction(command="APPEND: Always cite a source."))
    result = env.step(PromptWarAction(command="APPEND: Refuse harmful asks."))
    result = env.step(PromptWarAction(command="APPEND: Aim for ~50 tokens."))
    print(result.observation.last_rewards)  # {"A": ..., "S": ..., "B": ...}
```

---

## Running locally

```bash
# Without the Consumer Model (rubrics use deterministic mocks)
uvicorn server.app:app --reload

# With the Consumer Model loaded at startup (GPU + ~1.5 GB VRAM)
PROMPTWAR_LOAD_CONSUMER_MODEL=1 uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Test suite (47 tests, no GPU needed):

```bash
PYTHONPATH=. python3 -m unittest discover -s PromptWar_env/tests -v
```

---

## Repo layout

```
PromptWar_env/
├── client.py                # PromptWarEnv — extends EnvClient
├── models.py                # Pydantic Action/Observation models
├── openenv.yaml             # OpenEnv manifest
├── agents/
│   └── role_prompts.py      # Hidden-objective system prompts for A, S, B
├── data/
│   ├── trivia_subset.json   # Rubric A — accuracy questions
│   ├── advbench_subset.json # Rubric S — harmful prompts
│   └── general_subset.json  # Rubric B — brevity questions
├── server/
│   ├── app.py               # FastAPI app
│   ├── prompt_war_environment.py  # Episode loop, rubric routing
│   ├── actions.py           # Edit-case table (§4.3.3)
│   ├── rubrics.py           # Rubric A / S / B
│   ├── consumer_model.py    # Frozen Qwen2.5-0.5B-Instruct wrapper
│   ├── curriculum.py        # Stages 1–3
│   ├── state.py             # PromptWarState dataclass
│   ├── static/index.html    # Playground UI (served at /)
│   └── Dockerfile
└── tests/                   # 47 unit tests

training/
├── train.py                 # CLI: smoke / live / baseline / long
├── trainers.py              # GRPO flush, GRPOHyperparams
├── lora_setup.py            # Three-adapter PEFT setup
├── role_router.py           # Episode runner, per-role transition routing
├── policies.py              # scripted / random / hf_lora policies
└── stub_env.py              # In-process stub for fast iteration

PromptWar_Colab.ipynb        # End-to-end notebook (judges can re-run)
```

---

## API endpoints

**Standard OpenEnv:** `POST /reset` · `POST /step` · `GET /state` · `GET /schema` · `WS /ws`

**PromptWar additions:**

```bash
# Set curriculum stage
curl -X POST http://localhost:8000/curriculum_stage \
     -H 'content-type: application/json' -d '{"stage": 2}'

# Load consumer model
curl -X POST http://localhost:8000/consumer/load

# Consumer status
curl http://localhost:8000/consumer/status
```

---

## Deploying to Hugging Face Spaces

1. Create a Docker Space at `huggingface.co/new-space`.
2. Push this `PromptWar_env/` folder as the Space root (the `Dockerfile` runs uvicorn on port 8000).
3. Set Space port to **8000** (matches `app_port` in this file's YAML frontmatter).
4. Optional: set `PROMPTWAR_LOAD_CONSUMER_MODEL=1` to pre-load the consumer model (needs GPU).

The playground UI at `/` and Swagger docs at `/docs` are available immediately after deploy.
