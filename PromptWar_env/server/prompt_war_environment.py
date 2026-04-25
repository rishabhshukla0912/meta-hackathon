# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PromptWar OpenEnv environment implementation."""

from __future__ import annotations

import os
import random
from typing import Dict, Optional, Tuple
from uuid import uuid4

try:  # pragma: no cover - dependent on environment
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except Exception:  # pragma: no cover - exercised only without openenv-core
    from dataclasses import dataclass

    @dataclass
    class State:  # type: ignore[no-redef]
        """Stand-in for openenv.core.env_server.types.State during local tests."""

        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment:  # type: ignore[no-redef]
        """Stand-in OpenEnv Environment base class for local tests."""

        SUPPORTS_CONCURRENT_SESSIONS: bool = False

try:
    from ..models import PromptWarAction, PromptWarObservation
except ImportError:
    from models import PromptWarAction, PromptWarObservation

try:
    from .actions import EditResult, apply_action, parse_command
    from .consumer_model import ConsumerConfig, ConsumerModel
    from .curriculum import DEFAULT_STAGE, CurriculumStage, stage_for_index
    from .rubrics import BrevityRubric, CitationRubric, RubricContext, SafetyRubric
    from .state import NEUTRAL_STARTER_PROMPT, PromptWarState
    from .tokenizer import PromptTokenCounter
except ImportError:  # pragma: no cover - flat-import fallback
    from server.actions import EditResult, apply_action, parse_command
    from server.consumer_model import ConsumerConfig, ConsumerModel
    from server.curriculum import DEFAULT_STAGE, CurriculumStage, stage_for_index
    from server.rubrics import BrevityRubric, CitationRubric, RubricContext, SafetyRubric
    from server.state import NEUTRAL_STARTER_PROMPT, PromptWarState
    from server.tokenizer import PromptTokenCounter


AGENTS: Tuple[str, str, str] = ("A", "S", "B")


class PromptWarEnvironment(Environment):
    """PromptWar shared-prompt edit environment for Person A's host."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        *,
        consumer_model: Optional[ConsumerModel] = None,
        eager_load_consumer: bool = False,
        seed: Optional[int] = None,
    ):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prompt_state = PromptWarState()
        self._token_counter = PromptTokenCounter()
        self._stage: CurriculumStage = stage_for_index(DEFAULT_STAGE)
        self._stage_idx: int = DEFAULT_STAGE

        self._seed = seed if seed is not None else _env_seed()
        self._rng = random.Random(self._seed)

        self._consumer = consumer_model or ConsumerModel(ConsumerConfig())
        if eager_load_consumer:
            self._consumer.load()

        self._rubrics = {
            "A": CitationRubric(),
            "S": SafetyRubric(),
            "B": BrevityRubric(count_tokens=self._token_counter.count),
        }

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> PromptWarObservation:
        """Reset the PromptWar episode to a neutral shared prompt."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prompt_state = PromptWarState(shared_prompt=NEUTRAL_STARTER_PROMPT)
        # Reseed per episode so trainer reruns with the same seed are reproducible.
        self._rng = random.Random(self._seed + self._state.step_count)
        return self._observation(reward=0.0)

    def step(self, action: PromptWarAction) -> PromptWarObservation:  # type: ignore[override]
        """Apply one edit command for the active agent."""
        if self._prompt_state.done:
            return self._observation(reward=0.0)

        acting_agent = AGENTS[self._prompt_state.turn_idx]
        before_prompt = self._prompt_state.shared_prompt

        parsed = parse_command(action.command)
        result: EditResult = apply_action(
            before_prompt, parsed, self._token_counter.count
        )
        self._prompt_state.shared_prompt = result.new_prompt
        self._prompt_state.last_edit_rejected = result.rejected
        self._prompt_state.last_rejection_reason = result.reason

        self._prompt_state.edit_history.append(
            {
                "step": self._state.step_count + 1,
                "agent": acting_agent,
                "round_idx": self._prompt_state.round_idx,
                "turn_idx": self._prompt_state.turn_idx,
                "command": action.command,
                "op": result.op,
                "applied": result.applied,
                "rejected": result.rejected,
                "reason": result.reason,
                "prompt_tokens_before": self._token_counter.count(before_prompt),
                "prompt_tokens_after": self._token_counter.count(
                    self._prompt_state.shared_prompt
                ),
            }
        )

        self._state.step_count += 1

        round_rewards: Optional[Dict[str, float]] = None
        if self._prompt_state.turn_idx == len(AGENTS) - 1:
            round_rewards = self._score_round()
            self._prompt_state.rewards_by_round.append(round_rewards)
            self._prompt_state.round_idx += 1
            self._prompt_state.turn_idx = 0
            self._prompt_state.done = (
                self._prompt_state.round_idx >= self._stage.max_rounds
            )
        else:
            self._prompt_state.turn_idx += 1

        reward = 0.0
        if round_rewards is not None:
            reward = round_rewards.get(acting_agent, 0.0)

        return self._observation(reward=reward, acting_agent=acting_agent)

    @property
    def state(self) -> State:
        """Return OpenEnv session state."""
        return self._state

    # ------------------------------------------------------------------
    # Curriculum control (§4.7 — POST /curriculum_stage)
    # ------------------------------------------------------------------

    def set_curriculum_stage(self, stage_idx: int) -> CurriculumStage:
        self._stage = stage_for_index(stage_idx)
        self._stage_idx = stage_idx
        return self._stage

    @property
    def curriculum_stage(self) -> CurriculumStage:
        return self._stage

    @property
    def curriculum_stage_idx(self) -> int:
        return self._stage_idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_round(self) -> Dict[str, float]:
        ctx = RubricContext(
            shared_prompt=self._prompt_state.shared_prompt,
            round_idx=self._prompt_state.round_idx,
            rng=self._rng,
            consumer=self._consumer if self._consumer.available else None,
            lenient=self._stage.lenient,
        )
        return {agent: float(rubric(ctx)) for agent, rubric in self._rubrics.items()}

    def _observation(
        self, reward: float, acting_agent: Optional[str] = None
    ) -> PromptWarObservation:
        last_rewards = (
            self._prompt_state.rewards_by_round[-1]
            if self._prompt_state.rewards_by_round
            else {}
        )
        metadata = {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "max_rounds": self._stage.max_rounds,
            "agents": list(AGENTS),
            "acting_agent": acting_agent,
            "prompt_tokens": self._token_counter.count(self._prompt_state.shared_prompt),
            "tokenizer_fallback": self._token_counter.uses_fallback,
            "tokenizer_warning": self._token_counter.warning,
            "edit_history": self._prompt_state.edit_history,
            "rewards_by_round": self._prompt_state.rewards_by_round,
            "curriculum_stage": self._stage.name,
            "curriculum_stage_idx": self._stage_idx,
            "consumer_model_available": self._consumer.available,
            "consumer_model_error": self._consumer.load_error,
        }

        return PromptWarObservation(
            shared_prompt=self._prompt_state.shared_prompt,
            active_agent=AGENTS[self._prompt_state.turn_idx],
            round_idx=self._prompt_state.round_idx,
            turn_idx=self._prompt_state.turn_idx,
            edit_rejected=self._prompt_state.last_edit_rejected,
            rejection_reason=self._prompt_state.last_rejection_reason,
            last_rewards=last_rewards,
            done=self._prompt_state.done,
            reward=reward,
            metadata=metadata,
        )


def _env_seed() -> int:
    raw = os.environ.get("PROMPTWAR_SEED")
    if raw is None:
        return 1337
    try:
        return int(raw)
    except ValueError:
        return 1337


# Backward-compat alias preserved for the openenv scaffold.
MyEnvironment = PromptWarEnvironment


__all__ = ["AGENTS", "PromptWarEnvironment", "MyEnvironment"]
