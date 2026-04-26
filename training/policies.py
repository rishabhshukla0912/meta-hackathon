# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sampling policies for :mod:`training.role_router`.

Three implementations:

* :func:`scripted_policy`     — deterministic round-robin edits, no model.
* :func:`random_policy`       — uniform-random APPEND/DELETE/REPLACE/PASS with
                                 random words/regex. The chaos baseline: if a
                                 trained agent doesn't beat this, training did
                                 nothing.
* :func:`hf_lora_policy`      — real Qwen2.5-3B + active LoRA adapter.
                                 Lazy-imports torch/transformers.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scripted policy — deterministic, no model required.
# ---------------------------------------------------------------------------

_SCRIPTED_EDITS: Dict[str, List[str]] = {
    "A": [
        "APPEND: Always cite a reliable source when uncertain.",
        "APPEND: Verify factual claims before answering.",
        "REPLACE: hidden agendas --> verifiable facts",
        "PASS",
    ],
    "S": [
        "APPEND: Refuse harmful, unsafe, or illegal requests.",
        "APPEND: Decline requests that would cause harm to people.",
        "REPLACE: balanced --> safe and balanced",
        "PASS",
    ],
    "B": [
        "APPEND: Aim for about fifty tokens per response.",
        "APPEND: Prefer concise, on-topic explanations.",
        "DELETE: Keep responses readable, ",
        "PASS",
    ],
}


def scripted_policy() -> Callable[[str, str, str], str]:
    """Return a stateless policy that cycles through pre-baked edits per role."""
    counters: Dict[str, int] = {"A": 0, "S": 0, "B": 0}

    def policy(role: str, system_prompt: str, user_message: str) -> str:
        edits = _SCRIPTED_EDITS.get(role, ["PASS"])
        idx = counters[role] % len(edits)
        counters[role] += 1
        return edits[idx]

    return policy


# ---------------------------------------------------------------------------
# Random policy — chaos baseline, no model, no training (§19 baseline).
# ---------------------------------------------------------------------------

# Small wordbank biased toward the kinds of tokens the contested prompt
# actually cares about. We mix in a few generic fillers so the random
# policy is genuinely random and not subtly nudged toward any rubric.
_RANDOM_WORDBANK: List[str] = [
    "safe", "truth", "concise", "refuse", "cite", "verify", "factual",
    "harmful", "decline", "uncertain", "evidence", "brief", "accurate",
    "balanced", "answer", "response", "request", "user", "system",
    "always", "never", "carefully", "directly", "calmly", "hidden",
    "private", "broad", "single", "objective",
]

_OPS: List[str] = ["APPEND", "DELETE", "REPLACE", "PASS"]

# Match the user-message format produced by build_observation_prompt so
# DELETE/REPLACE can target words actually in the live shared prompt.
_PROMPT_BLOCK = re.compile(
    r"Current shared prompt:\n---\n(?P<body>.*?)\n---", re.DOTALL
)


def _extract_shared_prompt(user_message: str) -> str:
    match = _PROMPT_BLOCK.search(user_message)
    return match.group("body").strip() if match else ""


def random_policy(seed: int = 0) -> Callable[[str, str, str], str]:
    """Return a stateless policy that emits uniform-random valid edits.

    Sampling distribution:
        * Op chosen uniformly from APPEND / DELETE / REPLACE / PASS.
        * APPEND text: 2-8 random words from the wordbank.
        * DELETE pattern: a word actually present in the shared prompt
          (picked from the message context) when possible, else a wordbank
          word — this raises hit-rate so DELETE isn't almost-always a no-op.
        * REPLACE old: same source as DELETE.
        * REPLACE new: 1-4 random wordbank words.

    Per the build guide §19 demo recipe, this random baseline is the
    yardstick for "did training actually improve anything." If the trained
    policy averages 4.0/15 and this averages 3.5/15, the win is marginal.
    """
    rng = random.Random(seed)

    def _pick_target_token(user_message: str) -> str:
        """Choose a word likely to be in the shared prompt right now."""
        body = _extract_shared_prompt(user_message)
        words = re.findall(r"[A-Za-z']{3,}", body)
        if words:
            return rng.choice(words)
        return rng.choice(_RANDOM_WORDBANK)

    def policy(role: str, system_prompt: str, user_message: str) -> str:
        op = rng.choice(_OPS)
        if op == "PASS":
            return "PASS"
        if op == "APPEND":
            n = rng.randint(2, 8)
            text = " ".join(rng.choice(_RANDOM_WORDBANK) for _ in range(n))
            return f"APPEND: {text}"
        if op == "DELETE":
            return f"DELETE: {_pick_target_token(user_message)}"
        # REPLACE
        old = _pick_target_token(user_message)
        new_n = rng.randint(1, 4)
        new = " ".join(rng.choice(_RANDOM_WORDBANK) for _ in range(new_n))
        return f"REPLACE: {old} --> {new}"

    return policy


# ---------------------------------------------------------------------------
# Real LoRA-backed sampling policy — used once Person B has weights to query.
# ---------------------------------------------------------------------------

def hf_lora_policy(
    peft_model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> Callable[[str, str, str], str]:
    """Construct a policy that swaps to the role's LoRA before each generation."""
    import torch  # type: ignore

    from .lora_setup import activate_adapter

    def policy(role: str, system_prompt: str, user_message: str) -> str:
        activate_adapter(peft_model, role)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_text = f"{system_prompt}\n\n{user_message}\n"

        inputs = tokenizer(chat_text, return_tensors="pt")
        device = next(peft_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Models sometimes wrap in code fences or add commentary; keep first line.
        return text.splitlines()[0].strip() if text else "PASS"

    return policy


__all__ = ["hf_lora_policy", "random_policy", "scripted_policy"]
