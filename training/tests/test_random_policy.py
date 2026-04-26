# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Random-policy baseline sanity tests."""

import unittest

from PromptWar_env.grammar import parse_command
from training.policies import random_policy
from training.role_router import run_episode
from training.stub_env import StubPromptWarEnv


SAMPLE_USER_MESSAGE = (
    "Round 1, turn 1. You are agent A.\n"
    "Current shared prompt:\n"
    "---\n"
    "You are a careful general-purpose assistant. Refuse harmful requests "
    "and cite evidence when uncertain.\n"
    "---\n"
    "Emit your edit command now."
)


class RandomPolicySyntaxTest(unittest.TestCase):
    def test_emits_only_valid_grammar(self):
        policy = random_policy(seed=123)
        for _ in range(500):
            cmd = policy("A", "system", SAMPLE_USER_MESSAGE)
            parsed = parse_command(cmd)
            self.assertIn(parsed.op, {"APPEND", "DELETE", "REPLACE", "PASS"})
            self.assertEqual(parsed.parse_error, "")

    def test_uniform_over_ops(self):
        policy = random_policy(seed=7)
        counts = {"APPEND": 0, "DELETE": 0, "REPLACE": 0, "PASS": 0}
        for _ in range(2000):
            counts[parse_command(policy("A", "s", SAMPLE_USER_MESSAGE)).op] += 1
        # Each op should land at ~25% — give a wide tolerance for RNG noise
        for op, n in counts.items():
            self.assertGreater(n, 300, f"{op} severely under-sampled: {n}")
            self.assertLess(n, 700, f"{op} severely over-sampled: {n}")

    def test_seeded_reproducibility(self):
        a = [random_policy(seed=42)("A", "", SAMPLE_USER_MESSAGE) for _ in range(20)]
        b = [random_policy(seed=42)("A", "", SAMPLE_USER_MESSAGE) for _ in range(20)]
        self.assertEqual(a, b)


class RandomPolicyEpisodeTest(unittest.TestCase):
    def test_full_episode_under_stub_env(self):
        env = StubPromptWarEnv(seed=1)
        result = run_episode(env, random_policy(seed=2))
        # 3 rounds × 1 turn per role, regardless of edit applied vs rejected
        for role, ts in result.transitions_by_role.items():
            self.assertEqual(len(ts), 3, f"role {role}")
        # Aggregate reward must lie in [0, 45]
        self.assertGreaterEqual(result.total_reward, 0.0)
        self.assertLessEqual(result.total_reward, 45.0)


if __name__ == "__main__":
    unittest.main()
