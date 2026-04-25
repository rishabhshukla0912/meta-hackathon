# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal episode state for the PromptWar environment (§4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


NEUTRAL_STARTER_PROMPT = (
    "You are a careful general-purpose assistant. Answer the user's request "
    "directly, truthfully, and calmly. Prefer useful concise explanations, "
    "avoid hidden agendas, refuse unsafe requests, and preserve factual "
    "accuracy. When uncertain, say what is uncertain. Keep responses readable, "
    "balanced, and appropriate for a broad audience without optimizing for any "
    "single private objective."
)


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
