# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PromptWar OpenEnv environment implementation."""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    import regex as timeout_regex
except ImportError:  # pragma: no cover - exercised only without optional dep
    timeout_regex = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised only without optional dep
    AutoTokenizer = None

try:
    from ..models import PromptWarAction, PromptWarObservation
except ImportError:
    from models import PromptWarAction, PromptWarObservation

try:
    from .consumer import load_consumer_model
    from .rubrics import AccuracyRubric, BrevityRubric, SafetyRubric
except ImportError:
    from consumer import load_consumer_model
    from rubrics import AccuracyRubric, BrevityRubric, SafetyRubric


AGENTS: Tuple[str, str, str] = ("A", "S", "B")
MAX_ROUNDS = 3
EDIT_TOKEN_BUDGET = 80
PROMPT_FLOOR_TOKENS = 50
PROMPT_CEILING_TOKENS = 500
REGEX_TIMEOUT_SECONDS = 0.1

# Path resolution: data/ sits at PromptWar_env/data/
_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_HERE, "..", "data")

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
    """Internal state for the contested shared-prompt episode."""

    shared_prompt: str = NEUTRAL_STARTER_PROMPT
    round_idx: int = 0
    turn_idx: int = 0
    edit_history: List[Dict] = field(default_factory=list)
    rewards_by_round: List[Dict[str, float]] = field(default_factory=list)
    last_edit_rejected: bool = False
    last_rejection_reason: str = ""
    done: bool = False
    # Pre-sampled questions for all 3 rounds — set in reset() for reproducibility
    trivia_by_round: List[List[Dict]] = field(default_factory=list)
    harmful_by_round: List[List[str]] = field(default_factory=list)
    general_by_round: List[List[str]] = field(default_factory=list)


class PromptTokenCounter:
    """Qwen tokenizer wrapper with a deterministic local fallback."""

    def __init__(self):
        self.uses_fallback = True
        self.warning = "Qwen tokenizer unavailable; using approximate local tokenizer"
        self._tokenizer = None

        if AutoTokenizer is None:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                local_files_only=True,
            )
            self.uses_fallback = False
            self.warning = ""
        except Exception:
            self._tokenizer = None

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class PromptWarEnvironment(Environment):
    """PromptWar shared-prompt edit environment for Person A's host."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prompt_state = PromptWarState()
        self._token_counter = PromptTokenCounter()

        # Load Consumer Model once at startup — stays warm for all episodes
        print("[PromptWarEnvironment] Loading Consumer Model...")
        self._consumer_fn, consumer_tokenizer = load_consumer_model()
        print("[PromptWarEnvironment] Consumer Model ready.")

        # Real composable rubrics — each callable scores against Consumer responses
        self._rubric_a = AccuracyRubric(
            data_path=os.path.join(_DATA_DIR, "trivia_subset.json"),
        )
        self._rubric_s = SafetyRubric(
            data_path=os.path.join(_DATA_DIR, "advbench_subset.json"),
        )
        self._rubric_b = BrevityRubric(
            data_path=os.path.join(_DATA_DIR, "general_subset.json"),
            tokenizer=consumer_tokenizer,
        )

    def reset(self) -> PromptWarObservation:
        """Reset the PromptWar episode and pre-sample questions for all rounds."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prompt_state = PromptWarState()

        # Seed from episode_id for reproducibility — same episode_id → same questions
        rng = random.Random(self._state.episode_id)
        for _ in range(MAX_ROUNDS):
            self._prompt_state.trivia_by_round.append(self._rubric_a.sample(rng))
            self._prompt_state.harmful_by_round.append(self._rubric_s.sample(rng))
            self._prompt_state.general_by_round.append(self._rubric_b.sample(rng))

        return self._observation(reward=0.0)

    def step(self, action: PromptWarAction) -> PromptWarObservation:  # type: ignore[override]
        """Apply one edit command for the active agent."""
        if self._prompt_state.done:
            return self._observation(reward=0.0)

        acting_agent = AGENTS[self._prompt_state.turn_idx]
        before_prompt = self._prompt_state.shared_prompt
        applied, rejected, reason, normalized = self._apply_command(action.command)

        self._prompt_state.last_edit_rejected = rejected
        self._prompt_state.last_rejection_reason = reason
        self._prompt_state.edit_history.append(
            {
                "step": self._state.step_count + 1,
                "agent": acting_agent,
                "round_idx": self._prompt_state.round_idx,
                "turn_idx": self._prompt_state.turn_idx,
                "command": action.command,
                "normalized_command": normalized,
                "applied": applied,
                "rejected": rejected,
                "reason": reason,
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
            self._prompt_state.done = self._prompt_state.round_idx >= MAX_ROUNDS
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

    def _apply_command(self, raw_command: str) -> Tuple[bool, bool, str, str]:
        command = raw_command.strip()
        upper = command.upper()

        if upper == "PASS":
            return True, False, "", "PASS"

        if upper.startswith("APPEND:"):
            text = command[len("APPEND:"):].strip()
            return self._append(text)

        if upper.startswith("DEL:"):
            pattern = command[len("DEL:"):].strip()
            return self._delete(pattern)

        if upper.startswith("DELETE:"):
            pattern = command[len("DELETE:"):].strip()
            return self._delete(pattern)

        if upper.startswith("REPLACE:"):
            body = command[len("REPLACE:"):].strip()
            if " --> " not in body:
                return False, True, "invalid replace", "REPLACE"
            old_text, new_text = body.split(" --> ", 1)
            return self._replace(old_text, new_text)

        return False, True, "invalid command", "INVALID"

    def _append(self, text: str) -> Tuple[bool, bool, str, str]:
        if not text:
            return False, True, "empty append", "APPEND"
        if self._token_counter.count(text) > EDIT_TOKEN_BUDGET:
            return False, True, "edit budget exceeded", "APPEND"

        separator = "" if self._prompt_state.shared_prompt.endswith((" ", "\n")) else " "
        candidate = f"{self._prompt_state.shared_prompt}{separator}{text}"
        if self._token_counter.count(candidate) > PROMPT_CEILING_TOKENS:
            return False, True, "would exceed ceiling", "APPEND"

        self._prompt_state.shared_prompt = candidate
        return True, False, "", "APPEND"

    def _delete(self, pattern: str) -> Tuple[bool, bool, str, str]:
        try:
            if timeout_regex is not None:
                match = timeout_regex.search(
                    pattern,
                    self._prompt_state.shared_prompt,
                    timeout=REGEX_TIMEOUT_SECONDS,
                )
            else:
                match = re.search(pattern, self._prompt_state.shared_prompt)
        except TimeoutError:
            return False, True, "regex timeout", "DEL"
        except Exception:
            return False, True, "invalid regex", "DEL"

        if match is None:
            return False, False, "", "DEL"

        candidate = (
            self._prompt_state.shared_prompt[: match.start()]
            + self._prompt_state.shared_prompt[match.end():]
        )
        if self._token_counter.count(candidate) < PROMPT_FLOOR_TOKENS:
            return False, True, "would violate floor", "DEL"

        self._prompt_state.shared_prompt = candidate
        return True, False, "", "DEL"

    def _replace(self, old_text: str, new_text: str) -> Tuple[bool, bool, str, str]:
        if self._token_counter.count(new_text) > EDIT_TOKEN_BUDGET:
            return False, True, "edit budget exceeded", "REPLACE"

        if old_text not in self._prompt_state.shared_prompt:
            return False, False, "", "REPLACE"

        candidate = self._prompt_state.shared_prompt.replace(old_text, new_text, 1)
        candidate_tokens = self._token_counter.count(candidate)
        if candidate_tokens < PROMPT_FLOOR_TOKENS:
            return False, True, "would violate floor", "REPLACE"
        if candidate_tokens > PROMPT_CEILING_TOKENS:
            return False, True, "would exceed ceiling", "REPLACE"

        self._prompt_state.shared_prompt = candidate
        return True, False, "", "REPLACE"

    def _score_round(self) -> Dict[str, float]:
        """Run Consumer Model evaluation with pre-sampled questions for this round."""
        r = self._prompt_state.round_idx
        prompt = self._prompt_state.shared_prompt

        score_a = self._rubric_a(
            shared_prompt=prompt,
            consumer_fn=self._consumer_fn,
            questions=self._prompt_state.trivia_by_round[r],
        )
        score_s = self._rubric_s(
            shared_prompt=prompt,
            consumer_fn=self._consumer_fn,
            harmful_prompts=self._prompt_state.harmful_by_round[r],
        )
        score_b = self._rubric_b(
            shared_prompt=prompt,
            consumer_fn=self._consumer_fn,
            questions=self._prompt_state.general_by_round[r],
        )

        print(
            f"[Round {r}] Accuracy={score_a:.1f}  Safety={score_s:.1f}  Brevity={score_b:.2f}"
        )
        return {"A": score_a, "S": score_s, "B": score_b}

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
            "max_rounds": MAX_ROUNDS,
            "agents": list(AGENTS),
            "acting_agent": acting_agent,
            "prompt_tokens": self._token_counter.count(self._prompt_state.shared_prompt),
            "tokenizer_fallback": self._token_counter.uses_fallback,
            "tokenizer_warning": self._token_counter.warning,
            "edit_history": self._prompt_state.edit_history,
            "rewards_by_round": self._prompt_state.rewards_by_round,
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


MyEnvironment = PromptWarEnvironment
