# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Token counter (§4.3.1).

All token counts use the agent model's tokenizer (Qwen2.5 family). The
tokenizer is loaded once at server startup. If unavailable (offline dev,
no HF cache, missing transformers), we fall back to a deterministic
word/punct regex counter — clearly signaled via ``uses_fallback``.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover - exercised only without optional dep
    AutoTokenizer = None  # type: ignore


_FALLBACK_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class PromptTokenCounter:
    """Qwen tokenizer wrapper with a deterministic local fallback."""

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.uses_fallback = True
        self.warning = "Qwen tokenizer unavailable; using approximate local tokenizer"
        self._tokenizer = None
        self._model_id = model_id

        if AutoTokenizer is None:
            return

        attempts = [{"local_files_only": True}]
        if os.environ.get("PROMPTWAR_ALLOW_TOKENIZER_DOWNLOAD", "0") == "1":
            attempts.append({})

        for attempt_kwargs in attempts:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, **attempt_kwargs)
                self.uses_fallback = False
                self.warning = ""
                return
            except Exception as exc:  # pragma: no cover - env-dependent
                logger.debug("tokenizer load attempt failed (%s): %s", attempt_kwargs, exc)
                self._tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return len(_FALLBACK_PATTERN.findall(text))


__all__ = ["PromptTokenCounter"]
