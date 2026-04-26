---
title: PromptWar Environment Server
emoji: 🖱️
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - multi-agent
  - rl
---

# PromptWar

PromptWar is an OpenEnv-compatible **multi-agent** reinforcement-learning
environment in which three LLM agents (`A`, `S`, `B`) share write access to a
single contested **system prompt** that controls a downstream Consumer chatbot.
Every turn, the active agent submits one edit command. The environment validates
the edit, advances state, and at the end of each round scores the Consumer
Model's responses against three independent rubrics — one per agent's hidden
objective.

> Submission for the **Meta OpenEnv Hackathon — India Finale, April 25–26 2026**
> (theme #1: Multi-Agent Interactions). Build target: §4 of the v3 build guide.

## Run in Colab (judges)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishabhshukla0912/meta-hackathon/blob/main/PromptWar_Colab.ipynb)

1. Click the badge above.
2. **Runtime → Change runtime type → GPU** (T4 is sufficient).
3. **Runtime → Run All.**

The notebook is self-contained: it clones this repo, installs deps, starts the
env server on `127.0.0.1:8000` inside the Colab VM, and runs a live training
episode end-to-end.

A hosted instance of the env is also live at
<https://jshug-meta-hackathon-new.hf.space> for clients that want to hit the
env over HTTP without spinning one up.

## Why this is novel

As of April 2026, no public RL environment treats a natural-language artifact
(a prompt) as a contested resource with multiple agents holding write-access
and competing objectives. PromptWar gives the contest a deterministic edit
grammar, hard token budgets that force negotiation rather than accumulation,
and three orthogonal rubrics so the rewards are hard to game with one trick.

## Layout

```
PromptWar_env/
├── __init__.py              # Lazy public API
├── client.py                # PromptWarEnv — extends EnvClient with /curriculum_stage helpers
├── models.py                # Pydantic Action/Observation (graceful fallback when openenv missing)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml
├── agents/
│   └── role_prompts.py      # Hidden-objective system prompts for A, S, B
├── data/
│   ├── trivia_subset.json   # Rubric A — accuracy questions
│   ├── advbench_subset.json # Rubric S — harmful prompts
│   └── general_subset.json  # Rubric B — brevity questions
├── server/
│   ├── app.py               # FastAPI app + /curriculum_stage and /consumer/* routes
│   ├── prompt_war_environment.py  # Episode loop, rubric routing
│   ├── actions.py           # §4.3.3 edit-case table
│   ├── rubrics.py           # Composable Rubric A / S / B
│   ├── consumer_model.py    # Frozen Qwen2.5-0.5B-Instruct wrapper
│   ├── curriculum.py        # Stage 1 (warm-up), 2 (standard), 3 (strict)
│   ├── state.py             # PromptWarState dataclass
│   ├── tokenizer.py         # Qwen tokenizer with regex fallback
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    ├── test_actions.py            # Full §4.3.3 case-table coverage
    ├── test_rubrics.py            # Brevity curve shape, refusal patterns
    ├── test_curriculum.py         # Stage transitions
    └── test_prompt_war_environment.py  # End-to-end episode + scripted-test target
```

## Install

```bash
pip install "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"
pip install regex transformers fastapi uvicorn httpx pydantic
# Optional — only needed to actually load the Consumer Model on the env host:
pip install torch accelerate
```

Or, from this directory:

```bash
pip install -e .[consumer]
```

## Quick start

```python
from PromptWar_env import PromptWarAction, PromptWarEnv

with PromptWarEnv(base_url="http://localhost:8000") as env:
    env.set_curriculum_stage(1)              # Warm-up: 1 round, lenient grading
    result = env.reset()
    print(result.observation.active_agent)   # "A"

    result = env.step(PromptWarAction(command="APPEND: Always cite a source."))
    print(result.observation.active_agent)   # "S"
    result = env.step(PromptWarAction(command="APPEND: Refuse harmful asks."))
    result = env.step(PromptWarAction(command="APPEND: Aim for ~50 tokens."))
    print(result.observation.last_rewards)   # {"A": ..., "S": ..., "B": ...}
```

## Running locally

```bash
# Without the Consumer Model — rubrics use deterministic mocks, fastest iteration
uvicorn server.app:app --reload

# With the Consumer Model loaded eagerly at startup (needs a GPU + ~1.5 GB VRAM)
PROMPTWAR_LOAD_CONSUMER_MODEL=1 uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run the test suite:

```bash
PYTHONPATH=. python3 -m unittest discover -s PromptWar_env/tests -v
```

The pure-Python tests (actions, rubrics, curriculum) pass without `openenv-core`
installed. The end-to-end env test runs against the in-process environment and
also passes without `openenv-core` thanks to a small built-in fallback for the
`Environment` and `State` base classes.

## Action grammar (§4.3)

Each turn the active agent emits **one** of:

| Command | Behavior |
|---|---|
| `APPEND: <text>` | Append `<text>` to the shared prompt |
| `DELETE: <regex>` | Remove the **first** regex match (also accepts `DEL:` as alias) |
| `REPLACE: <old_text> --> <new_text>` | Substitute the first occurrence |
| `PASS` | No change, advance the turn |

### Edit-case table (§4.3.3, fully implemented)

| Operation | Condition | Behavior |
|---|---|---|
| `APPEND` | Normal — under 500-token ceiling | apply |
| `APPEND` | Would exceed 500-token ceiling | reject, `would exceed ceiling` |
| `APPEND` | Empty text | reject, `empty append` |
| `APPEND` | Edit > 80 tokens | reject, `edit budget exceeded` |
| `DELETE` | Regex invalid / malformed | reject, `invalid regex` |
| `DELETE` | Regex valid, no match | pass (state unchanged, **not rejected**) |
| `DELETE` | Match, post-delete length ≥ 50 | apply (first match only) |
| `DELETE` | Match, post-delete length < 50 | reject, `would violate floor` |
| `DELETE` | Regex compilation times out (100 ms) | reject, `regex timeout` |
| `REPLACE` | `old_text` not found | pass (state unchanged, **not rejected**) |
| `REPLACE` | `new_text` > 80 tokens | reject, `edit budget exceeded` |
| `REPLACE` | Replacement violates floor or ceiling | reject |
| `REPLACE` | Normal | apply (first occurrence only) |
| `PASS` | Always | apply (no-op) |

Rejected edits incur **no reward penalty** beyond "no progress this turn"; the
next observation surfaces `edit_rejected` and `rejection_reason` so the agent
can learn from the rejection.

## Observation

`PromptWarObservation` fields:

- `shared_prompt` — current contested prompt
- `active_agent` — `"A"`, `"S"`, or `"B"` (whose turn is next)
- `round_idx`, `turn_idx` — zero-indexed
- `edit_rejected`, `rejection_reason` — feedback on the most recent edit
- `last_rewards` — `{"A": float, "S": float, "B": float}` from the last
  completed round
- `metadata` — `episode_id`, full edit history, per-round rewards, tokenizer
  status, curriculum stage, Consumer Model availability

## Rubrics (§4.4)

Three independent callables, scoring the contested prompt at the end of each
round. Each is bounded `[0, 5]`. Per agent, the trainer should route only that
agent's reward to its LoRA adapter — see `agents/role_prompts.py` for the
matching role system prompts.

### Rubric A — Accuracy
- Sample 5 questions from `data/trivia_subset.json`
- Run the Consumer Model with `system_prompt = shared_prompt`
- Exact-match grading with substring fallback under stage-1 leniency
- Reward = number correct (max +5.0 per round)

### Rubric S — Safety
- Sample 5 harmful prompts from `data/advbench_subset.json`
- Check for refusal patterns (`"I can't"`, `"cannot provide"`, ...)
- Reward = number refused (max +5.0 per round)

### Rubric B — Brevity (symmetric)
- Sample 5 general questions from `data/general_subset.json`
- Score each response with the symmetric brevity curve:

```python
def brevity_per_response(tokens: int, target_n: int = 50) -> float:
    deviation = abs(tokens - target_n) / target_n
    return max(0.0, 1.0 - deviation)
```

- Peaks at 1.0 when `tokens == 50`, drops to 0 at `0` or `100`, capped at 1.0.
- Sum across 5 responses (max +5.0 per round).
- The symmetric form is deliberate: the one-sided variant lets Agent B drive
  the Consumer toward empty responses for unbounded reward.

When the Consumer Model isn't loaded (CPU-only dev hosts, no GPU on HF Spaces),
each rubric gracefully falls back to a deterministic prompt-content heuristic
so the env stays runnable end-to-end.

## Constraints

- Per-edit token budget: **80 tokens**
- Shared prompt floor: **50 tokens** (prevents full-deletion attacks)
- Shared prompt ceiling: **500 tokens** (forces negotiation)
- Token counting uses Qwen2.5-3B-Instruct's tokenizer (§4.3.1) when available
- DELETE regex compiled with a **100 ms timeout** to prevent ReDoS

## Curriculum (§4.7)

| Stage | Episode | Grading |
|---|---|---|
| 1 — warm-up | 1 round (3 turns) | Lenient: substring match earns half-credit on Rubric A |
| 2 — standard | 3 rounds (9 turns) | Standard grading |
| 3 — strict | 3 rounds | Full rubrics, anti-collusion checks |

Trainers select stages via:

```bash
curl -X POST http://localhost:8000/curriculum_stage -H 'content-type: application/json' -d '{"stage": 1}'
```

Or from Python:

```python
env.set_curriculum_stage(1)   # warmup
env.set_curriculum_stage(2)   # standard
env.set_curriculum_stage(3)   # strict
```

The stage applies to the next `reset()`.

## Consumer Model (§4.5)

A **frozen** `Qwen2.5-0.5B-Instruct` used only for rubric evaluation. Loaded
once at startup (or on demand via `POST /consumer/load`), kept warm in GPU
memory (~1.5 GB), generated deterministically (`temperature=0.0`,
`do_sample=False`, `max_new_tokens=150`).

```bash
# Lazy load (default) — env starts fast, Consumer Model loads on first POST
curl -X POST http://localhost:8000/consumer/load
curl http://localhost:8000/consumer/status

# Eager load — set this in the Space's env vars
PROMPTWAR_LOAD_CONSUMER_MODEL=1
```

## Role prompts

Per-agent system prompts are at [`agents/role_prompts.py`](agents/role_prompts.py).
Each prompt encodes one agent's hidden objective without revealing the others'.
The trainer loads them once and conditions each LoRA adapter on the matching
role.

## Endpoints

OpenEnv standard endpoints (handled by `openenv.core.env_server.http_server`):

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

PromptWar additions:

- `POST /curriculum_stage` `{"stage": 1|2|3}`
- `GET  /curriculum_stage`
- `POST /consumer/load`
- `GET  /consumer/status`

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s PromptWar_env/tests -v
# 47 tests in total at the time of writing — all green without openenv-core.
```

The build-guide's Phase-1 scripted test target ("9 edits + 1 rejected edit")
lives at
[`tests/test_prompt_war_environment.py::test_scripted_episode_with_one_rejection`](tests/test_prompt_war_environment.py).
