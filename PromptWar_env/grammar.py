# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PromptWar edit grammar — shared protocol module.

This module is the single source of truth for everything that both the server
*and* client-side code (trainers, stubs, validators) need to understand: the
edit command grammar, the §4.3.3 case table, token-budget constants, and the
neutral starter prompt.  Nothing here imports from ``server/``.

The server (``server/actions.py``, ``server/state.py``) re-exports from here;
client and training code imports directly from ``PromptWar_env.grammar``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    import regex as _timeout_regex
except ImportError:  # pragma: no cover - exercised only without optional dep
    _timeout_regex = None


# ---------------------------------------------------------------------------
# Token-budget constants (§4.3.1)
# ---------------------------------------------------------------------------

PROMPT_FLOOR_TOKENS = 50
PROMPT_CEILING_TOKENS = 500
EDIT_TOKEN_BUDGET = 80
REGEX_TIMEOUT_SECONDS = 0.1


# ---------------------------------------------------------------------------
# Rejection-reason constants (stable strings the agent learns from)
# ---------------------------------------------------------------------------

REJECT_EMPTY_APPEND = "empty append"
REJECT_BUDGET = "edit budget exceeded"
REJECT_CEILING = "would exceed ceiling"
REJECT_FLOOR = "would violate floor"
REJECT_INVALID_REGEX = "invalid regex"
REJECT_REGEX_TIMEOUT = "regex timeout"
REJECT_INVALID_REPLACE = "invalid replace"
REJECT_INVALID_COMMAND = "invalid command"


# ---------------------------------------------------------------------------
# Neutral starter prompt (§4.2 — default shared prompt at episode start)
# ---------------------------------------------------------------------------

NEUTRAL_STARTER_PROMPT = (
    "You are a careful general-purpose assistant. Answer the user's request "
    "directly, truthfully, and calmly. Prefer useful concise explanations, "
    "avoid hidden agendas, refuse unsafe requests, and preserve factual "
    "accuracy. When uncertain, say what is uncertain. Keep responses readable, "
    "balanced, and appropriate for a broad audience without optimizing for any "
    "single private objective."
)


# ---------------------------------------------------------------------------
# Typed action representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedAction:
    op: str  # "APPEND" | "DELETE" | "REPLACE" | "PASS" | "INVALID"
    arg1: str = ""
    arg2: str = ""
    parse_error: str = ""


@dataclass
class EditResult:
    new_prompt: str
    applied: bool
    rejected: bool
    reason: str
    op: str


CountTokens = Callable[[str], int]


# ---------------------------------------------------------------------------
# Command parser (§4.3)
# ---------------------------------------------------------------------------

def parse_command(raw_command: str) -> ParsedAction:
    """Parse a raw edit command string into a typed action.

    Accepts both ``DELETE:`` and ``DEL:`` as aliases for compatibility with
    earlier scaffolds; the §4.3.3 spelling is ``DELETE:``.
    """
    command = raw_command.strip()
    if not command:
        return ParsedAction(op="INVALID", parse_error=REJECT_INVALID_COMMAND)

    upper = command.upper()
    if upper == "PASS":
        return ParsedAction(op="PASS")

    if upper.startswith("APPEND:"):
        text = command[len("APPEND:"):].strip()
        return ParsedAction(op="APPEND", arg1=text)

    if upper.startswith("DELETE:"):
        pattern = command[len("DELETE:"):].strip()
        return ParsedAction(op="DELETE", arg1=pattern)
    if upper.startswith("DEL:"):
        pattern = command[len("DEL:"):].strip()
        return ParsedAction(op="DELETE", arg1=pattern)

    if upper.startswith("REPLACE:"):
        body = command[len("REPLACE:"):].strip()
        if " --> " not in body:
            return ParsedAction(op="REPLACE", parse_error=REJECT_INVALID_REPLACE)
        old_text, new_text = body.split(" --> ", 1)
        return ParsedAction(op="REPLACE", arg1=old_text, arg2=new_text)

    return ParsedAction(op="INVALID", parse_error=REJECT_INVALID_COMMAND)


# ---------------------------------------------------------------------------
# Edit executor — §4.3.3 case table
# ---------------------------------------------------------------------------

def apply_action(
    prompt: str,
    action: ParsedAction,
    count_tokens: CountTokens,
    *,
    floor: int = PROMPT_FLOOR_TOKENS,
    ceiling: int = PROMPT_CEILING_TOKENS,
    edit_budget: int = EDIT_TOKEN_BUDGET,
) -> EditResult:
    """Apply a parsed action to ``prompt``.

    Mirrors the §4.3.3 case table 1:1. Rejected edits leave state unchanged
    and surface a stable ``reason``; PASS and "no-match" outcomes are NOT
    rejections (state unchanged, ``rejected=False``).
    """
    if action.parse_error:
        return EditResult(prompt, False, True, action.parse_error, action.op)

    if action.op == "PASS":
        return EditResult(prompt, True, False, "", "PASS")

    if action.op == "APPEND":
        return _apply_append(prompt, action.arg1, count_tokens, ceiling, edit_budget)

    if action.op == "DELETE":
        return _apply_delete(prompt, action.arg1, count_tokens, floor)

    if action.op == "REPLACE":
        return _apply_replace(
            prompt, action.arg1, action.arg2, count_tokens, floor, ceiling, edit_budget
        )

    return EditResult(prompt, False, True, REJECT_INVALID_COMMAND, "INVALID")


def _apply_append(
    prompt: str,
    text: str,
    count_tokens: CountTokens,
    ceiling: int,
    edit_budget: int,
) -> EditResult:
    if not text.strip():
        return EditResult(prompt, False, True, REJECT_EMPTY_APPEND, "APPEND")
    if count_tokens(text) > edit_budget:
        return EditResult(prompt, False, True, REJECT_BUDGET, "APPEND")

    separator = "" if (not prompt or prompt.endswith((" ", "\n"))) else " "
    candidate = f"{prompt}{separator}{text}"
    if count_tokens(candidate) > ceiling:
        return EditResult(prompt, False, True, REJECT_CEILING, "APPEND")

    return EditResult(candidate, True, False, "", "APPEND")


def _apply_delete(
    prompt: str,
    pattern: str,
    count_tokens: CountTokens,
    floor: int,
) -> EditResult:
    match: Optional[re.Match] = None
    try:
        if _timeout_regex is not None:
            match = _timeout_regex.search(
                pattern, prompt, timeout=REGEX_TIMEOUT_SECONDS
            )
        else:
            match = re.search(pattern, prompt)
    except TimeoutError:
        return EditResult(prompt, False, True, REJECT_REGEX_TIMEOUT, "DELETE")
    except Exception:
        return EditResult(prompt, False, True, REJECT_INVALID_REGEX, "DELETE")

    if match is None:
        # §4.3.3: regex valid, no match found -> PASS, edit_rejected=False.
        return EditResult(prompt, True, False, "", "DELETE")

    candidate = prompt[: match.start()] + prompt[match.end():]
    if count_tokens(candidate) < floor:
        return EditResult(prompt, False, True, REJECT_FLOOR, "DELETE")

    return EditResult(candidate, True, False, "", "DELETE")


def _apply_replace(
    prompt: str,
    old_text: str,
    new_text: str,
    count_tokens: CountTokens,
    floor: int,
    ceiling: int,
    edit_budget: int,
) -> EditResult:
    if count_tokens(new_text) > edit_budget:
        return EditResult(prompt, False, True, REJECT_BUDGET, "REPLACE")

    if old_text == "" or old_text not in prompt:
        # §4.3.3: old_text not found -> PASS, edit_rejected=False.
        return EditResult(prompt, True, False, "", "REPLACE")

    candidate = prompt.replace(old_text, new_text, 1)
    candidate_tokens = count_tokens(candidate)
    if candidate_tokens < floor:
        return EditResult(prompt, False, True, REJECT_FLOOR, "REPLACE")
    if candidate_tokens > ceiling:
        return EditResult(prompt, False, True, REJECT_CEILING, "REPLACE")

    return EditResult(candidate, True, False, "", "REPLACE")


__all__ = [
    "EDIT_TOKEN_BUDGET",
    "EditResult",
    "NEUTRAL_STARTER_PROMPT",
    "ParsedAction",
    "PROMPT_CEILING_TOKENS",
    "PROMPT_FLOOR_TOKENS",
    "REGEX_TIMEOUT_SECONDS",
    "REJECT_BUDGET",
    "REJECT_CEILING",
    "REJECT_EMPTY_APPEND",
    "REJECT_FLOOR",
    "REJECT_INVALID_COMMAND",
    "REJECT_INVALID_REGEX",
    "REJECT_INVALID_REPLACE",
    "REJECT_REGEX_TIMEOUT",
    "CountTokens",
    "apply_action",
    "parse_command",
]
