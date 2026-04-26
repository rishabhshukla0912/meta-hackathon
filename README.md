# PromptWar

OpenEnv-compatible **multi-agent** RL environment in which three LLM agents
(`A`, `S`, `B`) share write access to a single contested **system prompt** that
controls a downstream Consumer chatbot. Each turn the active agent submits one
edit; at the end of each round the Consumer's responses are scored against
three independent rubrics — one per agent's hidden objective.

> Submission for the **Meta OpenEnv Hackathon — India Finale, April 25–26 2026**
> (theme #1: Multi-Agent Interactions).

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

## Repo layout

| Path | What it is |
|---|---|
| `PromptWar_Colab.ipynb` | One-click Colab notebook — env + training, end-to-end |
| `PromptWar_env/` | OpenEnv environment server (FastAPI + Docker, deployed as the HF Space) |
| `training/` | Trainer-side: 3 LoRA adapters on Qwen2.5-3B + 3 GRPO trainers, one per role |

See [`PromptWar_env/README.md`](./PromptWar_env/README.md) for the environment
spec and [`training/README.md`](./training/README.md) for trainer details.
