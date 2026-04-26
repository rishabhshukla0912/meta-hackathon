# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Consumer Model wrapper (§4.5).

A frozen Qwen2.5-0.5B-Instruct used only for rubric evaluation. Loaded
lazily — if the environment cannot find weights or torch/transformers
aren't available, the wrapper reports ``available=False`` and rubrics
silently fall back to deterministic mocks (still scoring, never crashing).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _default_max_new_tokens() -> int:
    return _env_int("PROMPTWAR_CONSUMER_MAX_NEW_TOKENS", 150)


@dataclass
class ConsumerConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = field(default_factory=_default_max_new_tokens)
    temperature: float = 0.0
    do_sample: bool = False
    device: Optional[str] = None  # "cuda" | "cpu" | None (auto)


class ConsumerModel:
    """Frozen Qwen2.5-0.5B-Instruct wrapper with deterministic decoding."""

    def __init__(self, config: Optional[ConsumerConfig] = None):
        self.config = config or ConsumerConfig()
        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None
        self.available = False
        self.load_error: str = ""

    def load(self) -> bool:
        """Attempt to load the underlying model. Idempotent."""
        if self.available:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - env-dependent
            self.load_error = f"transformers/torch unavailable: {exc}"
            logger.info("ConsumerModel: %s", self.load_error)
            return False

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype="auto"
            )
        except Exception as exc:  # pragma: no cover - env-dependent
            self.load_error = f"failed to load {self.config.model_id}: {exc}"
            logger.info("ConsumerModel: %s", self.load_error)
            return False

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.to(device)
        except Exception as exc:  # pragma: no cover - env-dependent
            self.load_error = f"failed to move to {device}: {exc}"
            return False

        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self.available = True
        self.load_error = ""
        logger.info("ConsumerModel loaded on %s", device)
        return True

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Run a single deterministic completion. Returns ``""`` if unavailable."""
        if not self.available:
            return ""

        import torch  # local import to keep optional dep optional

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            chat_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_text = f"{system_prompt}\n\n{user_message}\n"

        inputs = self._tokenizer(chat_text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


__all__ = ["ConsumerConfig", "ConsumerModel"]
