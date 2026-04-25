#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Side-by-side comparison: random baseline under mock rubrics vs real Consumer Model.

Run order:

    # 1) Start env server (without eager-loading the model — we'll load it lazily)
    PYTHONPATH=PromptWar_env:. python3 -m server.app --port 8765 &

    # 2) Run this driver: it asks the server for "mock numbers", loads the
    #    Consumer Model on-the-fly via POST /consumer/load, then re-runs and
    #    prints a side-by-side diff.
    python3 -m training.compare_baselines --env-url http://localhost:8765 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any, Dict, List

import httpx

from PromptWar_env.client import PromptWarEnv  # noqa: E402

from .policies import random_policy
from .role_router import run_episode


def _per_role_stats(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for role in ("A", "S", "B"):
        xs = [row[role] for row in rows]
        out[role] = {
            "mean": round(statistics.mean(xs), 3),
            "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
        }
    return out


def _aggregate_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    sums = [row["A"] + row["S"] + row["B"] for row in rows]
    return {
        "mean": round(statistics.mean(sums), 3),
        "stdev": round(statistics.stdev(sums), 3) if len(sums) > 1 else 0.0,
        "min": round(min(sums), 3),
        "max": round(max(sums), 3),
    }


def _run(env_url: str, episodes: int, seed: int) -> Dict[str, Any]:
    env = PromptWarEnv(base_url=env_url)
    policy = random_policy(seed=seed)
    rows: List[Dict[str, float]] = []
    started = time.time()
    try:
        for ep in range(episodes):
            ep_start = time.time()
            episode = run_episode(env, policy)
            row = {role: round(v, 3) for role, v in episode.per_role_reward.items()}
            rows.append(row)
            print(
                f"  ep {ep + 1}/{episodes}: A={row['A']:.2f}  S={row['S']:.2f}  B={row['B']:.2f}  "
                f"({time.time() - ep_start:.1f}s)",
                flush=True,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    return {
        "episodes": episodes,
        "wall_seconds": round(time.time() - started, 1),
        "per_role": _per_role_stats(rows),
        "aggregate": _aggregate_stats(rows),
        "rows": rows,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:8765")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-mock", action="store_true",
        help="Skip the mock-rubric pass (use when Consumer Model is already loaded).",
    )
    args = parser.parse_args(argv)

    base = args.env_url.rstrip("/")

    # Sanity: server up?
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{base}/consumer/status")
        r.raise_for_status()
        before = r.json()
    print(f"server consumer status before load: {before}")

    mock_result: Dict[str, Any] | None = None
    if not args.skip_mock and not before.get("available"):
        print("\n=== PASS 1: mock rubrics (Consumer Model NOT loaded) ===")
        mock_result = _run(base, args.episodes, args.seed)

    # Load Consumer Model. This may take a few minutes the first time
    # (downloading Qwen2.5-0.5B-Instruct into the HF cache).
    print("\nloading Consumer Model via /consumer/load (may download weights) ...")
    with httpx.Client(timeout=600.0) as c:
        r = c.post(f"{base}/consumer/load")
        r.raise_for_status()
        load_response = r.json()
    print(f"  -> {load_response}")
    if not load_response.get("available"):
        print("Consumer Model failed to load — aborting real-rubric pass.", file=sys.stderr)
        return 1

    print("\n=== PASS 2: real Qwen2.5-0.5B Consumer Model ===")
    real_result = _run(base, args.episodes, args.seed)

    print("\n=== summary ===")
    summary = {
        "mock": mock_result,
        "real": real_result,
        "max_possible_per_episode": 45.0,
    }
    print(json.dumps(summary, indent=2))

    if mock_result is not None:
        print("\n=== diff (mean reward, mock - real) ===")
        for role in ("A", "S", "B"):
            m = mock_result["per_role"][role]["mean"]
            rr = real_result["per_role"][role]["mean"]
            arrow = "↓" if rr < m else "↑"
            print(f"  {role}: mock={m:>6.2f}  real={rr:>6.2f}  delta={rr - m:+.2f} {arrow}")
        ma = mock_result["aggregate"]["mean"]
        ra = real_result["aggregate"]["mean"]
        print(f"  aggregate: mock={ma:>6.2f}  real={ra:>6.2f}  delta={ra - ma:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
