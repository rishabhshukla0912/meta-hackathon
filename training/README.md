# PromptWar trainer (Person B)

Trainer-side package for the PromptWar OpenEnv environment. This is the
"Person B" half of the build — three LoRA adapters on Qwen2.5-3B, three
`GRPOTrainer` instances, a stub env for trainer-wiring smoke tests, and a
live-episode mode that drives Person A's real env URL.

## Layout

| File | Purpose | §-ref |
|---|---|---|
| `stub_env.py` | Random-reward env mimicking PromptWarEnv shape | §5 H1-4 |
| `role_router.py` | Per-turn rollout, returns per-role transitions | §4.6 |
| `policies.py` | Scripted (no-model) and HF+LoRA sampling policies | §4.6 |
| `lora_setup.py` | Qwen2.5-3B base + 3 named LoRA adapters | §4.6 |
| `trainers.py` | Three `GRPOTrainer`s, one per role | §4.6 |
| `train.py` | CLI: `--mode smoke / live / baseline / long` | §5, §19 |
| `syntactic_validity.py` | Risk #2 quick check (50 edits/role, gate at 50%/30%) | §5 H6-8 |

## Install

CPU-only smoke testing needs nothing beyond the env's deps. For real GRPO
training:

```bash
pip install -r training/requirements.txt
```

The `unsloth` and `bitsandbytes` lines target a CUDA box.

## Run order (matches the v3 build plan)

```bash
# Hour 1-4: trainer-wiring smoke (no GPU, no live env)
python -m training.train --mode smoke --steps 5

# Hour 4-6: connect to A's env, verify rollout shape
python -m training.train --mode live --env-url http://localhost:8000 --episodes 1

# Random-policy baseline (sanity-check the env is not trivially solvable)
python -m training.train --mode baseline --env-url http://localhost:8000 --episodes 30

# Hour 6-8: Risk #2 syntactic validity probe (needs GPU + base model)
python -c "from training.syntactic_validity import run_with_base_model; \
           import json; print(json.dumps({k: v.rate for k,v in run_with_base_model()['reports'].items()}))"

# Hour 8-10: 50-step GRPO with real env + real rubrics
python -m training.train --mode long --env-url http://localhost:8000 --steps 50

# Hour 13+: long run
python -m training.train --mode long --env-url http://localhost:8000 \
    --steps 1500 --use-unsloth --load-in-4bit \
    --checkpoint-every 100 --output-dir ./checkpoints/promptwar
```

## Architecture

```
┌────────────────────────────┐         ┌─────────────────────────────────┐
│  Person A (env host)       │   HTTP  │  Person B (trainer host)        │
│                            │◀───────▶│                                 │
│  PromptWar_env/server/app  │         │  training/train.py --mode live  │
│  + Consumer Model (0.5B)   │         │  └ run_episode (role_router)    │
│                            │         │    ├ A trainer (LoRA "A")       │
│                            │         │    ├ S trainer (LoRA "S")       │
│                            │         │    └ B trainer (LoRA "B")       │
└────────────────────────────┘         │  base = Qwen2.5-3B-Instruct     │
                                       └─────────────────────────────────┘
```

Each turn, `role_router.run_episode` reads the active agent from the
observation, activates that role's LoRA via `lora_setup.activate_adapter`,
samples a completion, sends it as a `PromptWarAction`, and stores the
resulting `(prompt, completion, reward)` under that role. At end-of-episode,
`trainers.flush_episode` submits each role's transitions to its trainer.

Reward routing follows §4.4: at end of round the env emits per-agent
rewards (`last_rewards`); the router credits each role's most recent
transition with its own slice. Agent A only sees Rubric A's reward, etc.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s training/tests -v
```

10 tests, all green without torch/trl/peft installed.
