# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the PromptWar environment.

We try to inherit from ``openenv.core.env_server.types`` so the env wires
into OpenEnv's FastAPI server cleanly. If that package isn't installed
(e.g. on a CPU-only dev box running just the unit tests), we fall back
to bare Pydantic models with the same field shape — enough for in-process
inspection but not for serving over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

try:  # pragma: no cover - dependent on environment
    from openenv.core.env_server.types import Action as _OpenEnvAction
    from openenv.core.env_server.types import Observation as _OpenEnvObservation

    _BaseAction = _OpenEnvAction
    _BaseObservation = _OpenEnvObservation
    _OPENENV_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without openenv-core
    class _BaseAction(BaseModel):  # type: ignore[no-redef]
        """Minimal Action stand-in when openenv-core isn't installed."""

    class _BaseObservation(BaseModel):  # type: ignore[no-redef]
        """Minimal Observation stand-in when openenv-core isn't installed."""

        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    _OPENENV_AVAILABLE = False


AgentId = Literal["A", "S", "B"]


class PromptWarAction(_BaseAction):
    """Raw edit command for the active PromptWar agent."""

    command: str = Field(
        ...,
        description=(
            "One of: APPEND: <text>, DELETE: <regex>, "
            "REPLACE: <old_text> --> <new_text>, PASS"
        ),
    )


class PromptWarObservation(_BaseObservation):
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


__all__ = [
    "AgentId",
    "MyAction",
    "MyObservation",
    "PromptWarAction",
    "PromptWarObservation",
]
