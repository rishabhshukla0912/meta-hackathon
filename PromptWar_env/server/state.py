# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal episode state for the PromptWar environment (§4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..grammar import NEUTRAL_STARTER_PROMPT  # noqa: F401 — re-export


@dataclass
class PromptWarState:
    """Episode-scoped state for one PromptWar rollout."""

    shared_prompt: str = NEUTRAL_STARTER_PROMPT
    round_idx: int = 0
    turn_idx: int = 0
    edit_history: List[Dict] = field(default_factory=list)
    rewards_by_round: List[Dict[str, float]] = field(default_factory=list)
    last_edit_rejected: bool = False
    last_rejection_reason: str = ""
    done: bool = False


__all__ = ["NEUTRAL_STARTER_PROMPT", "PromptWarState"]
