# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export PromptWar long-run rollouts as a :class:`datasets.DatasetDict` (Hugging Face
Datasets) with two splits:

* ``transitions`` — one row per turn (prompt, completion, reward, role, …)
* ``episodes`` — one row per training step (aggregated rewards + serialized GRPO metrics)

Use :func:`save_rollout_dataset` after each checkpoint (or at the end) to write Arrow
data under ``output_dir / "hf_rollout_dataset"`` and optionally push the same object
to the Hub for sharing, analysis, or offline SFT/DPO datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .role_router import EpisodeResult

logger = logging.getLogger(__name__)

_TRANSITION_KEYS = (
    "training_step",
    "role",
    "round_idx",
    "turn_idx",
    "prompt",
    "completion",
    "reward",
    "edit_rejected",
    "rejection_reason",
)
_EPISODE_KEYS = (
    "training_step",
    "reward_A",
    "reward_S",
    "reward_B",
    "total_reward",
    "n_transitions",
    "rewards_by_round",
    "grpo_metrics",
)


def transition_rows_from_episode(episode: EpisodeResult, training_step: int) -> List[Dict[str, Any]]:
    """Flatten :class:`EpisodeResult` into one dict per transition (for an HF ``Dataset``)."""
    rows: List[Dict[str, Any]] = []
    for _role, ts in episode.transitions_by_role.items():
        for t in ts:
            rows.append(
                {
                    "training_step": training_step,
                    "role": t.role,
                    "round_idx": t.round_idx,
                    "turn_idx": t.turn_idx,
                    "prompt": t.prompt,
                    "completion": t.completion,
                    "reward": float(t.reward),
                    "edit_rejected": t.edit_rejected,
                    "rejection_reason": t.rejection_reason,
                }
            )
    return rows


def episode_row_from_step(
    training_step: int,
    episode: EpisodeResult,
    grpo_metrics: Any,
) -> Dict[str, Any]:
    """One summary row per training step (join key: ``training_step``)."""
    pr = episode.per_role_reward
    return {
        "training_step": training_step,
        "reward_A": float(pr.get("A", 0.0)),
        "reward_S": float(pr.get("S", 0.0)),
        "reward_B": float(pr.get("B", 0.0)),
        "total_reward": float(episode.total_reward),
        "n_transitions": sum(len(ts) for ts in episode.transitions_by_role.values()),
        "rewards_by_round": json.dumps(episode.rewards_by_round, default=float),
        "grpo_metrics": json.dumps(grpo_metrics, default=float),
    }


def _dicts_to_dataset(rows: List[Dict[str, Any]], keys: Sequence[str]):
    from datasets import Dataset  # type: ignore

    if not rows:
        return Dataset.from_dict({k: [] for k in keys})
    return Dataset.from_dict({k: [r[k] for r in rows] for k in keys})


def build_dataset_dict(
    transition_rows: List[Dict[str, Any]],
    episode_rows: List[Dict[str, Any]],
) -> "DatasetDict":
    from datasets import DatasetDict  # type: ignore

    return DatasetDict(
        {
            "transitions": _dicts_to_dataset(transition_rows, _TRANSITION_KEYS),
            "episodes": _dicts_to_dataset(episode_rows, _EPISODE_KEYS),
        }
    )


def save_rollout_dataset(
    transition_rows: List[Dict[str, Any]],
    episode_rows: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    *,
    hf_dataset_repo: Optional[str] = None,
    hf_private: bool = False,
    token: Optional[Union[str, bool]] = None,
) -> Dict[str, Any]:
    """Build a :class:`DatasetDict`, save to disk, optionally ``push_to_hub``."""
    dsd = build_dataset_dict(transition_rows, episode_rows)
    root = Path(output_dir)
    ds_path = root / "hf_rollout_dataset"
    dsd.save_to_disk(str(ds_path))
    out: Dict[str, Any] = {
        "dataset_dict_path": str(ds_path),
        "n_transitions": len(transition_rows),
        "n_episode_rows": len(episode_rows),
    }
    if hf_dataset_repo:
        dsd.push_to_hub(
            hf_dataset_repo,
            private=hf_private,
            token=token,
        )
        out["hf_dataset_repo"] = hf_dataset_repo
        logger.info("pushed DatasetDict to Hub: %s", hf_dataset_repo)
    return out


__all__ = [
    "build_dataset_dict",
    "episode_row_from_step",
    "save_rollout_dataset",
    "transition_rows_from_episode",
]
