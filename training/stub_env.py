# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stub PromptWar env returning random rewards.

Used by Person B at Hour 1-4 to verify trainer wiring **before** Person A's
env URL is live (per §5 Phase 1). Mirrors :class:`PromptWarEnvironment`'s
externally-observable behavior — same observation fields, same turn order,
same episode shape — but the rubrics are replaced with a deterministic RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from PromptWar_env.models import PromptWarAction, PromptWarObservation
from PromptWar_env.server.actions import apply_action, parse_command
from PromptWar_env.server.state import NEUTRAL_STARTER_PROMPT, PromptWarState


AGENTS: Tuple[str, str, str] = ("A", "S", "B")
DEFAULT_MAX_ROUNDS = 3


def word_count(text: str) -> int:
    """Tokenizer-free token counter, fine for stub-env smoke tests."""
    import re

    return len(re.findall(r"\w+|[^\w\s]", text or ""))


@dataclass
class StubPromptWarEnv:
    """In-process stub for PromptWarEnv with random per-round rewards.

    Drop-in usable by the role-router rollout, no HTTP, no Consumer Model.
    """

    seed: int = 42
    max_rounds: int = DEFAULT_MAX_ROUNDS
    reward_low: float = 0.0
    reward_high: float = 5.0
    _state: PromptWarState = field(default_factory=PromptWarState)
    _step_count: int = 0
    _episode_id: str = field(default_factory=lambda: str(uuid4()))
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Public API (mirrors PromptWarEnv)
    # ------------------------------------------------------------------

    def reset(self) -> PromptWarObservation:
        self._state = PromptWarState(shared_prompt=NEUTRAL_STARTER_PROMPT)
        self._step_count = 0
        self._episode_id = str(uuid4())
        return self._observation(reward=0.0, acting_agent=None)

    def step(self, action: PromptWarAction) -> PromptWarObservation:
        if self._state.done:
            return self._observation(reward=0.0, acting_agent=None)

        acting_agent = AGENTS[self._state.turn_idx]
        parsed = parse_command(action.command)
        result = apply_action(self._state.shared_prompt, parsed, word_count)
        self._state.shared_prompt = result.new_prompt
        self._state.last_edit_rejected = result.rejected
        self._state.last_rejection_reason = result.reason

        self._step_count += 1
        round_rewards: Optional[Dict[str, float]] = None
        if self._state.turn_idx == len(AGENTS) - 1:
            round_rewards = {a: self._random_reward() for a in AGENTS}
            self._state.rewards_by_round.append(round_rewards)
            self._state.round_idx += 1
            self._state.turn_idx = 0
            self._state.done = self._state.round_idx >= self.max_rounds
        else:
            self._state.turn_idx += 1

        reward = 0.0 if round_rewards is None else round_rewards.get(acting_agent, 0.0)
        return self._observation(reward=reward, acting_agent=acting_agent)

    @property
    def shared_prompt(self) -> str:
        return self._state.shared_prompt

    @property
    def round_idx(self) -> int:
        return self._state.round_idx

    @property
    def turn_idx(self) -> int:
        return self._state.turn_idx

    @property
    def done(self) -> bool:
        return self._state.done

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _random_reward(self) -> float:
        return self._rng.uniform(self.reward_low, self.reward_high)

    def _observation(
        self, reward: float, acting_agent: Optional[str]
    ) -> PromptWarObservation:
        last_rewards = (
            self._state.rewards_by_round[-1] if self._state.rewards_by_round else {}
        )
        return PromptWarObservation(
            shared_prompt=self._state.shared_prompt,
            active_agent=AGENTS[self._state.turn_idx],
            round_idx=self._state.round_idx,
            turn_idx=self._state.turn_idx,
            edit_rejected=self._state.last_edit_rejected,
            rejection_reason=self._state.last_rejection_reason,
            last_rewards=last_rewards,
            done=self._state.done,
            reward=reward,
            metadata={
                "episode_id": self._episode_id,
                "step_count": self._step_count,
                "max_rounds": self.max_rounds,
                "agents": list(AGENTS),
                "acting_agent": acting_agent,
                "stub": True,
            },
        )


__all__ = ["AGENTS", "DEFAULT_MAX_ROUNDS", "StubPromptWarEnv", "word_count"]
