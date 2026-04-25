# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Risk #2 quick check (§5 Hour 6-8).

Generate ``n_per_role`` edit actions per role with the *base* Qwen2.5-3B
(no training, no LoRA), parse each through
:func:`PromptWar_env.server.actions.parse_command`, and report the rate of
syntactically valid edits per role. The build guide gates at:

  * ≥ 50 % syntactic validity → ship as is.
  * 30-50 %                    → add few-shot examples and retry.
  * < 30 % after retry         → PIVOT TO TIMEWARP (§8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PromptWar_env.agents.role_prompts import ROLE_PROMPTS, build_observation_prompt
from PromptWar_env.server.actions import parse_command
from PromptWar_env.server.state import NEUTRAL_STARTER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ValidityReport:
    role: str
    samples: List[str]
    valid_count: int

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def rate(self) -> float:
        return self.valid_count / self.n if self.n else 0.0


def _is_syntactically_valid(raw: str) -> bool:
    parsed = parse_command(raw)
    return parsed.op in {"APPEND", "DELETE", "REPLACE", "PASS"} and not parsed.parse_error


def measure_role_validity(
    sampler: Callable[[str, str, str], str],
    *,
    n_per_role: int = 50,
    shared_prompt: str = NEUTRAL_STARTER_PROMPT,
) -> Dict[str, ValidityReport]:
    """Run the validity probe for all three roles."""
    reports: Dict[str, ValidityReport] = {}
    for role, system_prompt in ROLE_PROMPTS.items():
        samples: List[str] = []
        valid = 0
        for i in range(n_per_role):
            user_msg = build_observation_prompt(
                role=role,
                shared_prompt=shared_prompt,
                round_idx=0,
                turn_idx=i % 3,
            )
            raw = sampler(role, system_prompt, user_msg)
            samples.append(raw)
            if _is_syntactically_valid(raw):
                valid += 1
        reports[role] = ValidityReport(role=role, samples=samples, valid_count=valid)
    return reports


def gate_decision(reports: Dict[str, ValidityReport]) -> Dict[str, Any]:
    """Apply the build-guide gate to a set of validity reports."""
    rates = {role: r.rate for role, r in reports.items()}
    weakest_role, weakest_rate = min(rates.items(), key=lambda kv: kv[1])
    decision: str
    if weakest_rate >= 0.5:
        decision = "SHIP"
    elif weakest_rate >= 0.3:
        decision = "RETRY_WITH_FEWSHOT"
    else:
        decision = "PIVOT_TO_TIMEWARP"
    return {
        "rates": rates,
        "weakest_role": weakest_role,
        "weakest_rate": weakest_rate,
        "decision": decision,
    }


def run_with_base_model(
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    n_per_role: int = 50,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """End-to-end probe: build/load a base model and run the validity check.

    If ``model`` and ``tokenizer`` are supplied, they're used as is. Otherwise
    we lazily build :data:`BASE_MODEL_ID` via :func:`build_base_model`.
    """
    import torch  # type: ignore

    from .lora_setup import build_base_model

    if model is None or tokenizer is None:
        bundle = build_base_model()
        model, tokenizer = bundle["model"], bundle["tokenizer"]

    torch.manual_seed(seed)

    def sampler(role: str, system_prompt: str, user_message: str) -> str:
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
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text.splitlines()[0].strip() if text else "PASS"

    reports = measure_role_validity(sampler, n_per_role=n_per_role)
    gate = gate_decision(reports)
    return {"reports": reports, "gate": gate}


__all__ = [
    "ValidityReport",
    "gate_decision",
    "measure_role_validity",
    "run_with_base_model",
]
