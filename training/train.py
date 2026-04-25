# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer-side CLI for PromptWar (§5).

Three modes:

  * ``--mode smoke``  Stub env + scripted policy, 5 GRPO steps.
                      Verifies trainer wiring before the env URL is live
                      (Hour 1-4).

  * ``--mode live``   Real env URL + scripted (or model-backed) policy,
                      one episode end-to-end. Used at Hour 4-6 to hit the
                      joint Hour-6 checkpoint.

  * ``--mode long``   Real env + model-backed policy, ``--steps`` GRPO
                      iterations, checkpoint every 100 steps. Used from
                      Hour 13 onward into the overnight run.

Examples
--------

    # Smoke (no GPU, no live env required)
    python -m training.train --mode smoke --steps 5

    # Live episode against Person A's env (still uses scripted policy by default)
    python -m training.train --mode live --env-url http://localhost:8000

    # Real model-backed long run (GPU)
    python -m training.train --mode long \\
        --env-url http://localhost:8000 \\
        --steps 1500 --use-unsloth --load-in-4bit
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .role_router import EpisodeResult, run_episode
from .stub_env import StubPromptWarEnv
from .policies import random_policy, scripted_policy

logger = logging.getLogger("training.train")


# ---------------------------------------------------------------------------
# Mode: smoke
# ---------------------------------------------------------------------------

def run_smoke(steps: int = 5, seed: int = 42) -> Dict[str, Any]:
    """5-step trainer-wiring smoke test (§5 Phase 1, Hour 1-4).

    Drives the stub env with the scripted policy, accumulating per-role
    transitions. Skips the actual GRPO step if torch/trl aren't importable —
    the goal at this phase is just to verify rollout shape is sane.
    """
    env = StubPromptWarEnv(seed=seed)
    policy = scripted_policy()

    aggregated = EpisodeResult()
    for episode_idx in range(steps):
        episode = run_episode(env, policy)
        for role, transitions in episode.transitions_by_role.items():
            aggregated.transitions_by_role[role].extend(transitions)
        logger.info(
            "smoke episode %d: rewards=%s",
            episode_idx,
            episode.per_role_reward,
        )

    flush_metrics: Optional[list] = None
    try:
        flush_metrics = _attempt_grpo_flush(aggregated)
    except _MissingTorchTrl as exc:
        logger.warning("skipping GRPO flush: %s", exc)
    except Exception as exc:
        logger.warning("GRPO flush failed (rollout shape verified, continuing): %s: %s",
                       type(exc).__name__, exc)

    return {
        "mode": "smoke",
        "episodes": steps,
        "transitions_per_role": {r: len(ts) for r, ts in aggregated.transitions_by_role.items()},
        "rewards_per_role": aggregated.per_role_reward,
        "grpo_flush": flush_metrics,
    }


# ---------------------------------------------------------------------------
# Mode: live
# ---------------------------------------------------------------------------

def run_live(env_url: str, *, episodes: int = 1) -> Dict[str, Any]:
    """One end-to-end episode against the real PromptWar env URL (§5 Hour 4-6)."""
    from PromptWar_env.client import PromptWarEnv

    policy = scripted_policy()
    aggregated = EpisodeResult()

    env = PromptWarEnv(base_url=env_url)
    try:
        for episode_idx in range(episodes):
            episode = run_episode(env, policy)
            for role, transitions in episode.transitions_by_role.items():
                aggregated.transitions_by_role[role].extend(transitions)
            logger.info(
                "live episode %d: rewards=%s, final_prompt_tokens=%s",
                episode_idx,
                episode.per_role_reward,
                episode.final_observation.metadata.get("prompt_tokens")
                if episode.final_observation
                else None,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                pass

    return {
        "mode": "live",
        "episodes": episodes,
        "transitions_per_role": {r: len(ts) for r, ts in aggregated.transitions_by_role.items()},
        "rewards_per_role": aggregated.per_role_reward,
        "edit_history": aggregated.edit_history,
    }


# ---------------------------------------------------------------------------
# Mode: baseline (random policy, no model, sanity-checks env difficulty)
# ---------------------------------------------------------------------------

def run_baseline(
    env_url: Optional[str],
    *,
    episodes: int = 30,
    seed: int = 0,
) -> Dict[str, Any]:
    """Random-policy baseline (§19): N episodes, aggregate per-role rewards.

    If ``env_url`` is given we hit Person A's live env. Otherwise we use
    the stub env so the baseline is still runnable on a CPU-only laptop.
    """
    import statistics

    if env_url:
        from PromptWar_env.client import PromptWarEnv

        env: Any = PromptWarEnv(base_url=env_url)
    else:
        env = StubPromptWarEnv(seed=seed)

    policy = random_policy(seed=seed)

    per_role_history: Dict[str, List[float]] = {"A": [], "S": [], "B": []}
    rejections = 0
    applied_edits = 0
    try:
        for ep_idx in range(episodes):
            episode = run_episode(env, policy)
            for role, total in episode.per_role_reward.items():
                per_role_history[role].append(total)
            for ts in episode.transitions_by_role.values():
                for t in ts:
                    if t.edit_rejected:
                        rejections += 1
                    else:
                        applied_edits += 1
            logger.info(
                "baseline ep %d: rewards=%s",
                ep_idx,
                {r: round(v, 2) for r, v in episode.per_role_reward.items()},
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                pass

    def _stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(xs), 3),
            "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
        }

    per_role_stats = {role: _stats(vs) for role, vs in per_role_history.items()}
    aggregate = [a + s + b for a, s, b in zip(*per_role_history.values())]

    return {
        "mode": "baseline",
        "episodes": episodes,
        "env": env_url or "stub",
        "per_role_stats": per_role_stats,
        "aggregate_per_episode": _stats(aggregate),
        "max_possible_per_episode": 45.0,  # 3 rounds × 5.0 × 3 roles = 15 per role
        "applied_edits": applied_edits,
        "rejected_edits": rejections,
        "rejection_rate": round(
            rejections / max(applied_edits + rejections, 1), 3
        ),
    }


# ---------------------------------------------------------------------------
# Mode: long (real model + real env)
# ---------------------------------------------------------------------------

def run_long(
    env_url: str,
    *,
    steps: int = 1500,
    use_unsloth: bool = False,
    load_in_4bit: bool = False,
    checkpoint_every: int = 100,
    output_dir: str = "./checkpoints/promptwar",
) -> Dict[str, Any]:
    """Full GRPO long-run training (§5 Hours 12-22)."""
    from PromptWar_env.client import PromptWarEnv

    from .lora_setup import attach_lora_adapters, build_base_model
    from .policies import hf_lora_policy
    from .trainers import GRPOHyperparams, build_three_trainers, flush_episode

    bundle = build_base_model(use_unsloth=use_unsloth, load_in_4bit=load_in_4bit)
    base_model, tokenizer = bundle["model"], bundle["tokenizer"]
    peft_model = attach_lora_adapters(base_model)
    trainers = build_three_trainers(
        peft_model, tokenizer, hp=GRPOHyperparams(output_dir=output_dir)
    )
    policy = hf_lora_policy(peft_model, tokenizer)

    import json

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_log_path = Path(output_dir) / "metrics_log.json"
    metrics_log: list = []
    with PromptWarEnv(base_url=env_url) as env:
        for step_idx in range(steps):
            episode = run_episode(env, policy)
            metrics = flush_episode(peft_model, trainers, episode)
            metrics_log.append({"step": step_idx, "metrics": metrics})

            # Flush after every step so a crash mid-run still leaves something
            # plottable on disk, and so the plot cell can be run before the
            # full training completes.
            metrics_log_path.write_text(json.dumps(metrics_log, indent=2, default=float))

            if (step_idx + 1) % checkpoint_every == 0:
                ckpt_path = Path(output_dir) / f"step_{step_idx + 1}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                peft_model.save_pretrained(ckpt_path)
                logger.info("checkpointed to %s", ckpt_path)

    return {
        "mode": "long",
        "steps": steps,
        "metrics_log_tail": metrics_log[-5:],
        "metrics_log_path": str(metrics_log_path),
        "checkpoint_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# GRPO flush helper used by smoke mode
# ---------------------------------------------------------------------------

class _MissingTorchTrl(RuntimeError):
    pass


def _attempt_grpo_flush(episode: EpisodeResult) -> Optional[list]:
    """Try one GRPO step using whatever torch/trl + base model is available."""
    try:
        import torch  # type: ignore  # noqa: F401
        from trl import GRPOConfig  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise _MissingTorchTrl(f"torch/trl not installed ({exc}); skipping flush")

    from .lora_setup import attach_lora_adapters, build_base_model
    from .trainers import build_three_trainers, flush_episode

    bundle = build_base_model()
    peft_model = attach_lora_adapters(bundle["model"])
    trainers = build_three_trainers(peft_model, bundle["tokenizer"])
    return flush_episode(peft_model, trainers, episode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="PromptWar trainer driver")
    parser.add_argument(
        "--mode",
        choices=["smoke", "live", "baseline", "long"],
        required=True,
    )
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", default="./checkpoints/promptwar")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    if args.mode == "smoke":
        result = run_smoke(steps=args.steps, seed=args.seed)
    elif args.mode == "live":
        result = run_live(args.env_url, episodes=args.episodes)
    elif args.mode == "baseline":
        # If --episodes is the default 1, bump to 30 for the baseline mode
        # since 1 episode is too noisy to draw any conclusion.
        ep = args.episodes if args.episodes != 1 else 30
        # ``--env-url`` is a string; treat the literal "stub" as opt-out.
        env_url: Optional[str] = None if args.env_url == "stub" else args.env_url
        result = run_baseline(env_url, episodes=ep, seed=args.seed)
    else:
        result = run_long(
            args.env_url,
            steps=args.steps,
            use_unsloth=args.use_unsloth,
            load_in_4bit=args.load_in_4bit,
            checkpoint_every=args.checkpoint_every,
            output_dir=args.output_dir,
        )

    print("--- result ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
