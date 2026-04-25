# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PromptWar environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PromptWarAction, PromptWarObservation


class PromptWarEnv(
    EnvClient[PromptWarAction, PromptWarObservation, State]
):
    """Client for the PromptWar OpenEnv environment."""

    def _step_payload(self, action: PromptWarAction) -> Dict:
        """Convert PromptWarAction to the JSON step payload."""
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[PromptWarObservation]:
        """Parse server response into StepResult[PromptWarObservation]."""
        obs_data = payload.get("observation", {})
        observation = PromptWarObservation(
            shared_prompt=obs_data.get("shared_prompt", ""),
            active_agent=obs_data.get("active_agent", "A"),
            round_idx=obs_data.get("round_idx", 0),
            turn_idx=obs_data.get("turn_idx", 0),
            edit_rejected=obs_data.get("edit_rejected", False),
            rejection_reason=obs_data.get("rejection_reason", ""),
            last_rewards=obs_data.get("last_rewards", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

MyEnv = PromptWarEnv
