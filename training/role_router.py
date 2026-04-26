# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-turn rollout router for PromptWar (§4.6 / §5).

The trainer drives the env one turn at a time. Each turn:
  1. Read the current observation.
  2. Look up the active agent's role (A/S/B) and its adapter.
  3. Build the role-conditioned prompt (system role prompt + observation).
  4. Sample a completion from the adapter.
  5. Send the completion to the env as a ``PromptWarAction``.
  6. After the env returns, store the (role, prompt, completion, reward,
     ``edit_rejected``) transition under the role that produced it.

End-of-episode the router yields per-role transition lists, ready to be
fed to the matching :class:`GRPOTrainer`. The model-side parts (sampling,
adapter swapping) are pluggable callables so this module stays free of
torch/trl imports — unit tests can exercise the routing logic directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from PromptWar_env.agents.role_prompts import ROLE_PROMPTS, build_observation_prompt
from PromptWar_env.models import PromptWarAction, PromptWarObservation


SamplePolicy = Callable[[str, str, str], str]
"""Signature: ``policy(role, system_prompt, user_message) -> completion``.

The routed system prompt + per-turn observation are passed in. The policy
is responsible for activating the correct LoRA adapter, calling the model,
and returning a single completion string (the raw edit command).
"""


class EnvLike(Protocol):
    def reset(self) -> PromptWarObservation: ...
    def step(self, action: PromptWarAction) -> PromptWarObservation: ...


@dataclass
class Transition:
    """One per-turn datum, routed to its role's trainer."""

    role: str
    round_idx: int
    turn_idx: int
    prompt: str
    completion: str
    reward: float
    edit_rejected: bool
    rejection_reason: str


@dataclass
class EpisodeResult:
    """Outcome of a single PromptWar episode."""

    transitions_by_role: Dict[str, List[Transition]] = field(
        default_factory=lambda: {"A": [], "S": [], "B": []}
    )
    final_observation: Optional[PromptWarObservation] = None
    rewards_by_round: List[Dict[str, float]] = field(default_factory=list)
    edit_history: List[Dict] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(t.reward for ts in self.transitions_by_role.values() for t in ts)

    @property
    def per_role_reward(self) -> Dict[str, float]:
        return {role: sum(t.reward for t in ts) for role, ts in self.transitions_by_role.items()}


def _unwrap_observation(value: object) -> PromptWarObservation:
    """Accept either a bare observation or an EnvClient ``StepResult``."""
    obs = getattr(value, "observation", value)
    return obs  # type: ignore[return-value]


def run_episode(
    env: EnvLike,
    policy: SamplePolicy,
    *,
    max_turns: Optional[int] = None,
) -> EpisodeResult:
    """Drive ``env`` to completion using ``policy`` to choose each edit.

    Parameters
    ----------
    env
        Anything with ``reset()`` / ``step(PromptWarAction)`` — the live
        :class:`PromptWarEnv` HTTP client or the :class:`StubPromptWarEnv`.
    policy
        Sampling callable. Receives the role-conditioned system prompt and
        the per-turn user message; must return a single edit command.
    max_turns
        Optional safety cap. Rollouts already terminate via ``done=True``;
        this guards against bugs in custom envs.
    """
    result = EpisodeResult()
    obs = _unwrap_observation(env.reset())
    result.final_observation = obs

    step_idx = 0
    last_seen_round_idx = obs.round_idx
    while not obs.done:
        if max_turns is not None and step_idx >= max_turns:
            break

        role = obs.active_agent
        system_prompt = ROLE_PROMPTS[role]
        user_message = build_observation_prompt(
            role=role,
            shared_prompt=obs.shared_prompt,
            round_idx=obs.round_idx,
            turn_idx=obs.turn_idx,
            last_edit_rejected=obs.edit_rejected,
            last_rejection_reason=obs.rejection_reason,
        )

        completion = policy(role, system_prompt, user_message)
        action = PromptWarAction(command=completion)
        next_obs = _unwrap_observation(env.step(action))

        # Append the transition with reward 0; the env only fills its scalar
        # ``reward`` field for the agent who *closed* the round (per §4.4
        # "Rewards emitted per-agent"), so we credit per-role rewards from
        # ``last_rewards`` once the round actually closes (below).
        result.transitions_by_role[role].append(
            Transition(
                role=role,
                round_idx=obs.round_idx,
                turn_idx=obs.turn_idx,
                prompt=user_message,
                completion=completion,
                reward=0.0,
                edit_rejected=next_obs.edit_rejected,
                rejection_reason=next_obs.rejection_reason,
            )
        )

        round_just_closed = (
            next_obs.round_idx > last_seen_round_idx or next_obs.done
        )
        if round_just_closed and next_obs.last_rewards:
            result.rewards_by_round.append(dict(next_obs.last_rewards))
            for r, ts in result.transitions_by_role.items():
                if not ts:
                    continue
                base = float(next_obs.last_rewards.get(r, 0.0))
                # Penalize PASS — gives GRPO a gradient even when rubric reward is 0
                pass_penalty = -0.3 if ts[-1].completion.strip().upper() == "PASS" else 0.0
                ts[-1].reward = base + pass_penalty
            last_seen_round_idx = next_obs.round_idx

        obs = next_obs
        result.final_observation = obs
        step_idx += 1

    if obs.metadata:
        result.rewards_by_round = list(obs.metadata.get("rewards_by_round", []))
        result.edit_history = list(obs.metadata.get("edit_history", []))

    return result


__all__ = ["EnvLike", "EpisodeResult", "SamplePolicy", "Transition", "run_episode"]
