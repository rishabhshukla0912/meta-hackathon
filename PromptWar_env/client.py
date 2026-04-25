# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PromptWar environment client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

import httpx

# Try to reuse OpenEnv's StepResult/State as plain types. The EnvClient
# itself is intentionally NOT inherited: in recent openenv-core (≥0.2) it
# became an async WebSocket client, but PromptWar's trainer + role_router
# code is sync-HTTP. We always use the local sync EnvClient defined below.
try:  # pragma: no cover - depends on optional Meta OpenEnv install
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
except Exception:  # pragma: no cover - local fallback path
    @dataclass
    class StepResult(Generic[TypeVar("ObservationT")]):  # type: ignore[no-redef,misc]
        observation: Any
        reward: Optional[float] = None
        done: bool = False

    @dataclass
    class State:  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0

ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")
StateT = TypeVar("StateT")


class EnvClient(Generic[ActionT, ObservationT, StateT]):
    """Sync HTTP EnvClient for PromptWar.

    Replaces openenv-core's async WebSocket EnvClient (which arrived in
    openenv-core ≥0.2). The trainer + role_router stack is sync-only, so
    we keep a small purpose-built sync client here. Subclasses must
    implement ``_step_payload``, ``_parse_result``, and ``_parse_state``.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    def _step_payload(self, action: ActionT) -> Dict[str, Any]:
        raise NotImplementedError

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObservationT]:
        raise NotImplementedError

    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        raise NotImplementedError

    def reset(self) -> StepResult[ObservationT]:
        # OpenEnv ≥0.2 ResetRequest accepts {} — both seed and episode_id are optional.
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self.base_url}/reset", json={})
            response.raise_for_status()
            return self._parse_result(response.json())

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        # OpenEnv StepRequest expects {"action": {...}}, not the action dict at the top level.
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self.base_url}/step",
                json={"action": self._step_payload(action)},
            )
            response.raise_for_status()
            return self._parse_result(response.json())

    def state(self) -> StateT:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self.base_url}/state")
            response.raise_for_status()
            return self._parse_state(response.json())

    def close(self) -> None:
        """No-op — sessions are short-lived and per-request."""
        return None

    def __enter__(self) -> "EnvClient[ActionT, ObservationT, StateT]":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

from .models import PromptWarAction, PromptWarObservation


class PromptWarEnv(EnvClient[PromptWarAction, PromptWarObservation, State]):
    """Client for the PromptWar OpenEnv environment.

    Adds two PromptWar-specific helpers on top of the standard EnvClient API:

    * :meth:`set_curriculum_stage` — switch the env between warm-up,
      standard, and strict grading (§4.7).
    * :meth:`load_consumer_model` / :meth:`consumer_status` — control the
      Consumer Model lifecycle (§4.5) so trainers can defer the heavy
      Qwen2.5-0.5B load until the first real training step.
    """

    # ------------------------------------------------------------------
    # Required EnvClient hooks
    # ------------------------------------------------------------------

    def _step_payload(self, action: PromptWarAction) -> Dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PromptWarObservation]:
        obs_data = payload.get("observation", payload)
        observation = PromptWarObservation(
            shared_prompt=obs_data.get("shared_prompt", ""),
            active_agent=obs_data.get("active_agent", "A"),
            round_idx=obs_data.get("round_idx", 0),
            turn_idx=obs_data.get("turn_idx", 0),
            edit_rejected=obs_data.get("edit_rejected", False),
            rejection_reason=obs_data.get("rejection_reason", ""),
            last_rewards=obs_data.get("last_rewards", {}),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # ------------------------------------------------------------------
    # PromptWar-specific extensions
    # ------------------------------------------------------------------

    def set_curriculum_stage(self, stage: int) -> Dict[str, Any]:
        """Switch curriculum stage (1=warmup, 2=standard, 3=strict)."""
        url = f"{self.base_url.rstrip('/')}/curriculum_stage"
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json={"stage": stage})
            r.raise_for_status()
            return r.json()

    def get_curriculum_stage(self) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/curriculum_stage"
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()

    def load_consumer_model(self) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/consumer/load"
        with httpx.Client(timeout=600.0) as client:
            r = client.post(url)
            r.raise_for_status()
            return r.json()

    def consumer_status(self) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/consumer/status"
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()


MyEnv = PromptWarEnv

__all__ = ["MyEnv", "PromptWarEnv"]
