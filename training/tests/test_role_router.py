# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from training.policies import scripted_policy
from training.role_router import run_episode
from training.stub_env import StubPromptWarEnv


class RoleRouterTest(unittest.TestCase):
    def test_episode_produces_per_role_transitions(self):
        env = StubPromptWarEnv(seed=42)
        result = run_episode(env, scripted_policy())

        self.assertEqual(set(result.transitions_by_role.keys()), {"A", "S", "B"})
        # 3 rounds × 1 turn per role = 3 transitions per role
        for role, transitions in result.transitions_by_role.items():
            self.assertEqual(len(transitions), 3, f"role {role}")
            for t in transitions:
                self.assertEqual(t.role, role)
                self.assertGreaterEqual(t.reward, 0.0)
                self.assertLessEqual(t.reward, 5.0)

    def test_per_role_reward_sums(self):
        env = StubPromptWarEnv(seed=7)
        result = run_episode(env, scripted_policy())
        per_role = result.per_role_reward
        # Sum of per-role equals total
        self.assertAlmostEqual(
            sum(per_role.values()),
            result.total_reward,
            places=6,
        )

    def test_transition_round_turn_indices(self):
        env = StubPromptWarEnv(seed=1)
        result = run_episode(env, scripted_policy())
        for role, transitions in result.transitions_by_role.items():
            for i, t in enumerate(transitions):
                self.assertEqual(t.round_idx, i)
                # Each role acts at the same turn slot every round
                self.assertEqual(t.turn_idx, {"A": 0, "S": 1, "B": 2}[role])


if __name__ == "__main__":
    unittest.main()
