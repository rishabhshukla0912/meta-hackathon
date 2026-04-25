# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Three GRPOTrainer instances, one per role (§4.6).

Each trainer holds the same base model object but a **different active
LoRA adapter**. The rollout function from :mod:`role_router` produces
per-role transitions; this module turns those transitions into
``(prompt, completion, reward)`` tuples and feeds them to the trainer
matching the active role.

We deliberately import ``trl`` lazily so the training package stays
importable for unit tests on machines without GPUs / TRL installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .lora_setup import ROLES, activate_adapter
from .role_router import EpisodeResult, Transition

logger = logging.getLogger(__name__)


@dataclass
class GRPOHyperparams:
    """§5 Hour 10-12 starting points — `beta=0.004`, `num_generations=4`, `lr=5e-6`."""

    beta: float = 0.004
    num_generations: int = 4
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 96
    output_dir: str = "./checkpoints/promptwar"


@dataclass
class RoleGRPOTrainer:
    """One role's GRPOTrainer + the bookkeeping to flush transitions to it."""

    role: str
    trainer: Any
    pending_transitions: List[Transition] = field(default_factory=list)

    def submit(self, transition: Transition) -> None:
        if transition.role != self.role:
            raise ValueError(
                f"transition role {transition.role!r} does not match trainer role {self.role!r}"
            )
        self.pending_transitions.append(transition)

    def flush(self, peft_model: Any) -> Dict[str, Any]:
        """Run one optimizer step using the buffered transitions.

        TRL's ``GRPOTrainer.training_step`` expects a batch with
        ``prompts`` and a custom reward model; we adapt by feeding
        precomputed rewards via TRL's ``reward_funcs`` callback list.
        """
        if not self.pending_transitions:
            return {"role": self.role, "steps": 0, "n_transitions": 0}

        activate_adapter(peft_model, self.role)

        prompts = [t.prompt for t in self.pending_transitions]
        completions = [t.completion for t in self.pending_transitions]
        rewards = [t.reward for t in self.pending_transitions]

        # TRL >= 0.11 exposes ``training_step_with_rollouts``; fall back to a
        # generic ``step`` for older versions.
        result: Dict[str, Any]
        if hasattr(self.trainer, "training_step_with_rollouts"):
            metrics = self.trainer.training_step_with_rollouts(
                prompts=prompts, completions=completions, rewards=rewards
            )
            result = {"metrics": metrics}
        else:
            metrics = _generic_grpo_step(self.trainer, prompts, completions, rewards)
            result = {"metrics": metrics}

        result["role"] = self.role
        result["n_transitions"] = len(self.pending_transitions)
        result["steps"] = 1
        self.pending_transitions.clear()
        return result


def build_three_trainers(
    peft_model: Any,
    tokenizer: Any,
    *,
    hp: GRPOHyperparams | None = None,
) -> Dict[str, RoleGRPOTrainer]:
    """Construct one ``GRPOTrainer`` per role and return as a role-indexed dict."""
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    hp = hp or GRPOHyperparams()
    # Filter to fields the installed TRL actually accepts — the GRPOConfig
    # signature has churned across versions (e.g. max_prompt_length was
    # dropped/renamed in newer trl). This keeps us compatible without pinning.
    import inspect
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    candidate_kwargs = {
        "learning_rate": hp.learning_rate,
        "beta": hp.beta,
        "num_generations": hp.num_generations,
        "per_device_train_batch_size": hp.per_device_train_batch_size,
        "gradient_accumulation_steps": hp.gradient_accumulation_steps,
        "max_prompt_length": hp.max_prompt_length,
        "max_completion_length": hp.max_completion_length,
        "output_dir": hp.output_dir,
    }
    grpo_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    dropped = sorted(set(candidate_kwargs) - set(grpo_kwargs))
    if dropped:
        logger.info("GRPOConfig: dropping kwargs not supported by installed trl: %s", dropped)
    grpo_config = GRPOConfig(**grpo_kwargs)

    # TRL ≥0.13 requires a train_dataset at construction time even when we
    # feed prompts/completions directly via training_step_with_rollouts. A
    # one-row placeholder satisfies the constructor; the real prompts come
    # from the env at every step.
    placeholder_dataset = _placeholder_grpo_dataset()

    trainers: Dict[str, RoleGRPOTrainer] = {}
    for role in ROLES:
        # ``reward_funcs`` is a TRL hook; we'll override scores per-batch via
        # ``training_step_with_rollouts`` so the placeholder is fine.
        trainer = GRPOTrainer(
            model=peft_model,
            processing_class=tokenizer,
            args=grpo_config,
            reward_funcs=[lambda *args, **kwargs: 0.0],
            train_dataset=placeholder_dataset,
        )
        trainers[role] = RoleGRPOTrainer(role=role, trainer=trainer)
    return trainers


def _placeholder_grpo_dataset() -> Any:
    """One-row HF dataset whose only role is to satisfy GRPOTrainer's constructor."""
    from datasets import Dataset  # type: ignore
    return Dataset.from_dict({"prompt": ["placeholder — overridden by rollouts"]})


def flush_episode(
    peft_model: Any,
    trainers: Dict[str, RoleGRPOTrainer],
    episode: EpisodeResult,
) -> List[Dict[str, Any]]:
    """Submit every transition to its trainer and run one step per role."""
    for role, transitions in episode.transitions_by_role.items():
        if role not in trainers:
            logger.warning("no trainer for role %r; %d transitions dropped", role, len(transitions))
            continue
        for t in transitions:
            trainers[role].submit(t)

    return [trainers[role].flush(peft_model) for role in ROLES if role in trainers]


def _generic_grpo_step(
    trainer: Any,
    prompts: List[str],
    completions: List[str],
    rewards: List[float],
) -> Dict[str, Any]:
    """Best-effort fallback for older TRL versions.

    Builds a minimal batch and invokes ``trainer.compute_loss`` then steps
    the optimizer. Real production setups should use TRL's first-class
    rollout API, but this keeps the smoke test runnable.
    """
    import torch  # type: ignore

    tokenizer = trainer.processing_class
    full_texts = [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    device = next(trainer.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_attention_mask = prompt_inputs["attention_mask"].to(device)

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    for row_idx, prompt_len in enumerate(prompt_attention_mask.sum(dim=1).tolist()):
        labels[row_idx, :prompt_len] = -100

    with torch.enable_grad():
        outputs = trainer.model(**inputs, labels=labels)
        # Scale CE by reward as a crude REINFORCE proxy. NOT real GRPO —
        # only a fallback used when TRL doesn't expose the rollout API.
        scaled = outputs.loss * float(sum(rewards) / max(len(rewards), 1))
        scaled.backward()
        if trainer.optimizer is None:
            trainer.create_optimizer()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    return {
        "loss": float(outputs.loss.detach()),
        "scaled_loss": float(scaled.detach()),
        "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
    }


__all__ = [
    "GRPOHyperparams",
    "RoleGRPOTrainer",
    "build_three_trainers",
    "flush_episode",
]
