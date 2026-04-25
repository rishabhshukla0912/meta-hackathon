# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the PromptWar environment."""

from typing import Dict, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


AgentId = Literal["A", "S", "B"]


class PromptWarAction(Action):
    """Raw edit command for the active PromptWar agent."""

    command: str = Field(
        ...,
        description=(
            "One of: APPEND: <text>, DEL: <regex>, "
            "REPLACE: <old_text> --> <new_text>, PASS"
        ),
    )


class PromptWarObservation(Observation):
    """Observation returned after each PromptWar edit attempt."""

    shared_prompt: str = Field(default="", description="Current contested prompt")
    active_agent: AgentId = Field(default="A", description="Agent whose turn is next")
    round_idx: int = Field(default=0, description="Current zero-indexed round")
    turn_idx: int = Field(default=0, description="Current zero-indexed turn")
    edit_rejected: bool = Field(
        default=False, description="Whether the most recent edit was rejected"
    )
    rejection_reason: str = Field(
        default="", description="Human-readable reason for the most recent rejection"
    )
    last_rewards: Dict[str, float] = Field(
        default_factory=dict, description="Most recent per-agent round rewards"
    )


# Backward-compatible aliases for the scaffold's public imports.
MyAction = PromptWarAction
MyObservation = PromptWarObservation
