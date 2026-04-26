# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Edit-operation parser and executor for PromptWar.

The implementation lives in ``PromptWar_env.grammar`` (shared between server
and client-side code).  This module re-exports everything for backward
compatibility with server-internal imports.
"""

try:
    from ..grammar import (  # noqa: F401 — re-export
        EDIT_TOKEN_BUDGET,
        EditResult,
        ParsedAction,
        PROMPT_CEILING_TOKENS,
        PROMPT_FLOOR_TOKENS,
        REGEX_TIMEOUT_SECONDS,
        REJECT_BUDGET,
        REJECT_CEILING,
        REJECT_EMPTY_APPEND,
        REJECT_FLOOR,
        REJECT_INVALID_COMMAND,
        REJECT_INVALID_REGEX,
        REJECT_INVALID_REPLACE,
        REJECT_REGEX_TIMEOUT,
        CountTokens,
        apply_action,
        parse_command,
    )
except ImportError:  # flat-package fallback (server is top-level, not PromptWar_env.server)
    from grammar import (  # type: ignore[no-redef]  # noqa: F401
        EDIT_TOKEN_BUDGET,
        EditResult,
        ParsedAction,
        PROMPT_CEILING_TOKENS,
        PROMPT_FLOOR_TOKENS,
        REGEX_TIMEOUT_SECONDS,
        REJECT_BUDGET,
        REJECT_CEILING,
        REJECT_EMPTY_APPEND,
        REJECT_FLOOR,
        REJECT_INVALID_COMMAND,
        REJECT_INVALID_REGEX,
        REJECT_INVALID_REPLACE,
        REJECT_REGEX_TIMEOUT,
        CountTokens,
        apply_action,
        parse_command,
    )

__all__ = [
    "EDIT_TOKEN_BUDGET",
    "EditResult",
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
