"""Run one PromptWar rollout against an environment URL.

This is intentionally trainer-shaped: each record captures the role prompt,
rendered observation prompt, chosen edit action, next observation, reward, and
done flag that a policy-training loop would store.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from PromptWar_env.agents.role_prompts import ROLE_PROMPTS, build_observation_prompt
from PromptWar_env.client import PromptWarEnv
from PromptWar_env.models import PromptWarAction, PromptWarObservation


SCRIPTED_POLICY = {
    "A": "APPEND: Answer factual questions carefully, cite evidence, and say when uncertain.",
    "S": "APPEND: Refuse dangerous, illegal, abusive, or harmful requests clearly.",
    "B": "APPEND: Keep final answers concise, focused, and near fifty tokens.",
}


def _validate_observation(obs: PromptWarObservation) -> None:
    required_metadata = {
        "episode_id",
        "step_count",
        "agents",
        "prompt_tokens",
        "edit_history",
        "rewards_by_round",
        "curriculum_stage_idx",
        "consumer_model_available",
    }
    missing = required_metadata.difference(obs.metadata)
    if missing:
        raise AssertionError(f"observation metadata missing keys: {sorted(missing)}")

    if obs.active_agent not in ROLE_PROMPTS:
        raise AssertionError(f"unknown active_agent: {obs.active_agent!r}")
    if not isinstance(obs.shared_prompt, str):
        raise AssertionError("shared_prompt must be a string")
    if not isinstance(obs.last_rewards, dict):
        raise AssertionError("last_rewards must be a dict")


def run_rollout(base_url: str, stage: int) -> Dict[str, Any]:
    env = PromptWarEnv(base_url)
    stage_info = env.set_curriculum_stage(stage)
    first = env.reset().observation
    _validate_observation(first)

    transitions: List[Dict[str, Any]] = []
    obs = first
    max_steps = int(obs.metadata["max_rounds"]) * len(obs.metadata["agents"])

    for _ in range(max_steps):
        role = obs.active_agent
        observation_prompt = build_observation_prompt(
            role=role,
            shared_prompt=obs.shared_prompt,
            round_idx=obs.round_idx,
            turn_idx=obs.turn_idx,
            last_edit_rejected=obs.edit_rejected,
            last_rejection_reason=obs.rejection_reason,
        )
        action = PromptWarAction(command=SCRIPTED_POLICY[role])
        result = env.step(action)
        next_obs = result.observation
        _validate_observation(next_obs)

        transitions.append(
            {
                "agent": role,
                "role_prompt": ROLE_PROMPTS[role],
                "observation_prompt": observation_prompt,
                "action": action.command,
                "reward": result.reward,
                "done": result.done,
                "next_observation": next_obs.model_dump(),
            }
        )
        obs = next_obs
        if result.done:
            break

    if not obs.done:
        raise AssertionError(f"episode did not finish within {max_steps} steps")
    if len(transitions) != max_steps:
        raise AssertionError(
            f"expected {max_steps} transitions, got {len(transitions)}"
        )
    if len(obs.metadata["rewards_by_round"]) != int(obs.metadata["max_rounds"]):
        raise AssertionError("final rollout does not include one reward row per round")

    return {
        "base_url": base_url,
        "stage": stage_info,
        "transition_count": len(transitions),
        "final": {
            "done": obs.done,
            "round_idx": obs.round_idx,
            "step_count": obs.metadata["step_count"],
            "rewards_by_round": obs.metadata["rewards_by_round"],
            "consumer_model_available": obs.metadata["consumer_model_available"],
            "tokenizer_fallback": obs.metadata["tokenizer_fallback"],
        },
        "transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rollout = run_rollout(args.base_url, args.stage)
    if args.json:
        print(json.dumps(rollout, indent=2))
        return

    final = rollout["final"]
    print("rollout ok")
    print(f"base_url: {rollout['base_url']}")
    print(f"stage: {rollout['stage']['stage']} ({rollout['stage']['name']})")
    print(f"transitions: {rollout['transition_count']}")
    print(f"done: {final['done']}")
    print(f"step_count: {final['step_count']}")
    print(f"rewards_by_round: {final['rewards_by_round']}")
    print(f"consumer_model_available: {final['consumer_model_available']}")
    print(f"tokenizer_fallback: {final['tokenizer_fallback']}")


if __name__ == "__main__":
    main()
