# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Three-LoRA configuration on Qwen2.5-3B-Instruct (§4.6).

Loads a single base model and attaches three independently-trainable LoRA
adapters keyed ``"A"``, ``"S"``, ``"B"`` via ``peft``. The trainer swaps
adapters per turn through :func:`activate_adapter` so each role's gradients
flow only into its own weights.

This module imports torch/peft only when actually called — the rest of the
training package stays importable on CPU-only dev hosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ROLES: tuple = ("A", "S", "B")


@dataclass
class LoRAConfigSpec:
    """Per-§4.6 LoRA spec. Same hyperparams across all three adapters."""

    r: int = 16
    alpha: int = 32
    target_modules: str = "all-linear"
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    roles: tuple = field(default_factory=lambda: ROLES)


def build_base_model(
    model_id: str = BASE_MODEL_ID,
    *,
    use_unsloth: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> Dict[str, Any]:
    """Load the base Qwen2.5-3B model + tokenizer.

    Returns a dict with keys ``model``, ``tokenizer``, ``device_map``.

    ``use_unsloth=True`` swaps in :class:`unsloth.FastLanguageModel` (per
    the participant guide §10 — Unsloth for memory efficiency). Falls back
    to plain ``transformers`` if Unsloth isn't installed.
    """
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError:
            use_unsloth = False

    if use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=load_in_4bit,
        )
        return {"model": model, "tokenizer": tokenizer, "device_map": "unsloth"}

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = _base_model_kwargs(load_in_4bit=load_in_4bit, device_map=device_map)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return {"model": model, "tokenizer": tokenizer, "device_map": device_map}


def _base_model_kwargs(*, load_in_4bit: bool, device_map: str) -> Dict[str, Any]:
    """Build kwargs for ``AutoModelForCausalLM.from_pretrained``."""
    model_kwargs: Dict[str, Any] = {"torch_dtype": "auto", "device_map": device_map}
    if not load_in_4bit:
        return model_kwargs

    # Pass 4-bit loading through quantization_config so it is consumed by
    # transformers instead of leaking into Qwen2ForCausalLM.__init__.
    try:
        import torch  # type: ignore
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError as exc:  # pragma: no cover - GPU-only path
        raise RuntimeError(
            "load_in_4bit=True requires transformers + bitsandbytes installed"
        ) from exc

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return model_kwargs


def attach_lora_adapters(
    model: Any,
    spec: LoRAConfigSpec | None = None,
) -> Any:
    """Attach a TRL-compatible default adapter plus one adapter per role.

    Returns the resulting ``PeftModel``. TRL's ``GRPOTrainer`` expects a
    ``"default"`` PEFT config so it can create its reference adapter; the
    training loop still activates only the role adapters.
    """
    from peft import LoraConfig, get_peft_model  # type: ignore

    spec = spec or LoRAConfigSpec()
    lora_config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        target_modules=spec.target_modules,
        lora_dropout=spec.dropout,
        bias=spec.bias,
        task_type=spec.task_type,
    )

    roles: List[str] = list(spec.roles)
    if not roles:
        raise ValueError("LoRAConfigSpec.roles must be non-empty")

    peft_model = get_peft_model(model, lora_config)
    for role in roles:
        peft_model.add_adapter(adapter_name=role, peft_config=lora_config)
    peft_model.set_adapter(roles[0])
    return peft_model


def activate_adapter(peft_model: Any, role: str) -> None:
    """Swap the active LoRA adapter to the named role."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    peft_model.set_adapter(role)


def trainable_param_summary(peft_model: Any) -> Dict[str, int]:
    """Quick sanity check — counts trainable params per adapter."""
    counts: Dict[str, int] = {}
    for name, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        for role in ROLES:
            if f".{role}." in name or name.endswith(f".{role}.weight"):
                counts[role] = counts.get(role, 0) + p.numel()
                break
    return counts


__all__ = [
    "BASE_MODEL_ID",
    "LoRAConfigSpec",
    "ROLES",
    "activate_adapter",
    "attach_lora_adapters",
    "build_base_model",
    "trainable_param_summary",
]
