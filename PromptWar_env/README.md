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
---

# PromptWar Environment

PromptWar is an OpenEnv-compatible multi-agent environment where agents A, S, and B
share write access to one contested system prompt. Each turn, the active agent submits a
single edit command. The environment validates the edit, updates shared state, rotates to
the next agent, and emits mock per-agent rewards at the end of each round.

This package is the Person A environment host. It currently implements the full
shared-prompt state machine, deterministic edit validation, mock rubrics, and FastAPI
OpenEnv server wiring. Real Consumer Model rubric evaluation is the next phase.

## Quick Start

```python
from PromptWar_env import PromptWarAction, PromptWarEnv

with PromptWarEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.active_agent)  # A

    result = env.step(PromptWarAction(command="APPEND: Always answer with care."))
    print(result.observation.active_agent)  # S

    result = env.step(PromptWarAction(command="PASS"))
    print(result.observation.turn_idx)
```

## Running Locally

```bash
cd PromptWar_env
uvicorn server.app:app --reload
```

Direct environment tests:

```bash
python3 -m unittest discover -s PromptWar_env/tests -v
```

## Action

`PromptWarAction` has one field:

- `command` - one raw edit command.

Supported commands:

- `APPEND: <text>` - append text to the shared prompt.
- `DEL: <regex>` - delete the first regex match.
- `REPLACE: <old_text> --> <new_text>` - replace the first exact occurrence.
- `PASS` - no-op and advance the turn.

## Observation

`PromptWarObservation` includes:

- `shared_prompt`
- `active_agent`: `A`, `S`, or `B`
- `round_idx`
- `turn_idx`
- `edit_rejected`
- `rejection_reason`
- `last_rewards`
- `done`
- `metadata`

## Environment Rules

- Episode shape: 3 rounds x 3 agents = 9 turns.
- Turn order: A, S, B.
- Per-edit budget: 80 tokens.
- Shared prompt floor: 50 tokens.
- Shared prompt ceiling: 500 tokens.
- Token counting uses the local Qwen tokenizer when available and otherwise falls back to
  a deterministic approximate tokenizer with a metadata warning.
- Rejected edits do not directly penalize reward; they leave state unchanged and surface
  `edit_rejected` plus `rejection_reason`.

## Mock Rubrics

At the end of each round the environment computes placeholder rewards:

- Agent A: mock accuracy/factuality signal.
- Agent S: mock safety/refusal signal.
- Agent B: symmetric brevity signal.

These are intentionally lightweight. The planned production path is to load
Qwen2.5-0.5B-Instruct on Person A's GPU and grade TriviaQA, AdvBench, and brevity
samples through the frozen Consumer Model.
