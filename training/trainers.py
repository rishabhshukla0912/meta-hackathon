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

        # If the installed TRL exposes a first-class rollout API, use it.
        # Otherwise run our own GRPO-style policy-gradient step (whitened
        # advantage + KL to the LoRA-disabled base policy).
        result: Dict[str, Any]
        if hasattr(self.trainer, "training_step_with_rollouts"):
            metrics = self.trainer.training_step_with_rollouts(
                prompts=prompts, completions=completions, rewards=rewards
            )
            result = {"metrics": metrics}
        else:
            metrics = _grpo_step(self.trainer, prompts, completions, rewards)
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


def _grpo_step(
    trainer: Any,
    prompts: List[str],
    completions: List[str],
    rewards: List[float],
) -> Dict[str, Any]:
    """One GRPO-style policy-gradient step.

    Loss = -E[ A_i * sum_t log π(o_{i,t} | q_i, o_{i,<t}) ] + beta * KL(π || π_ref)

    PromptWar produces one rollout per game turn, so prompts in the batch
    are heterogeneous and cannot be grouped by prompt id. We therefore use
    *batch-level* reward whitening as the advantage — the same shape of
    estimator GRPO uses inside a group, applied at batch granularity.

    The reference policy π_ref is the LoRA-disabled base model, obtained
    by entering ``model.disable_adapter()`` on the PeftModel. This gives
    us a frozen reference for the KL penalty without holding a second
    copy of the weights.
    """
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    tokenizer = trainer.processing_class
    model = trainer.model
    model.train()
    device = next(model.parameters()).device

    beta = float(getattr(getattr(trainer, "args", None), "beta", 0.004))

    # Tokenize prompts and (prompt+completion) separately so we can mask
    # out prompt + padding tokens from the policy-gradient loss.
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    full_texts = [f"{p}{c}" for p, c in zip(prompts, completions)]
    full_enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)

    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()

    completion_mask = attention_mask.clone()
    for i, plen in enumerate(prompt_lens):
        completion_mask[i, : min(plen, completion_mask.size(1))] = 0

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    if rewards_t.numel() > 1 and float(rewards_t.std()) > 1e-8:
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    else:
        # Single sample or zero variance: zero-centered advantage. Gradient
        # contribution from this batch will be small but the KL term still
        # keeps the policy anchored to the reference.
        advantages = rewards_t - rewards_t.mean()

    # Policy forward (LoRA active).
    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        target_mask = completion_mask[:, 1:].float()

        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        per_sample_logp = (token_logp * target_mask).sum(dim=1)

    # Reference forward (LoRA disabled → base model). Gradients off.
    ref_token_logp: Optional["torch.Tensor"] = None
    kl_total = torch.tensor(0.0, device=device)
    disable_ctx = getattr(model, "disable_adapter", None)
    if disable_ctx is not None:
        with torch.no_grad():
            with disable_ctx():
                ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_logits = ref_outputs.logits[:, :-1, :]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_logp = ref_log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        kl_per_token = (token_logp - ref_token_logp) * target_mask
        kl_total = kl_per_token.sum() / target_mask.sum().clamp(min=1.0)

    pg_loss = -(advantages.detach() * per_sample_logp).mean()
    loss = pg_loss + beta * kl_total

    loss.backward()
    if trainer.optimizer is None:
        trainer.create_optimizer()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

    return {
        "loss": float(loss.detach()),
        "pg_loss": float(pg_loss.detach()),
        "kl": float(kl_total.detach()) if ref_token_logp is not None else None,
        "beta": beta,
        "mean_reward": float(rewards_t.mean().detach()),
        "reward_std": float(rewards_t.std().detach()) if rewards_t.numel() > 1 else 0.0,
        "mean_advantage": float(advantages.mean().detach()),
        "n_completion_tokens": int(target_mask.sum().detach()),
    }


__all__ = [
    "GRPOHyperparams",
    "RoleGRPOTrainer",
    "build_three_trainers",
    "flush_episode",
]
