# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from PromptWar_env.models import PromptWarAction
from training.stub_env import AGENTS, StubPromptWarEnv


class StubEnvTest(unittest.TestCase):
    def test_reset_returns_neutral_state(self):
        env = StubPromptWarEnv()
        obs = env.reset()
        self.assertEqual(obs.active_agent, "A")
        self.assertEqual(obs.round_idx, 0)
        self.assertFalse(obs.done)

    def test_three_rounds_complete_episode(self):
        env = StubPromptWarEnv()
        env.reset()
        for _ in range(3 * len(AGENTS)):
            obs = env.step(PromptWarAction(command="PASS"))
        self.assertTrue(obs.done)
        self.assertEqual(len(obs.last_rewards), 3)

    def test_reward_routed_to_acting_agent(self):
        env = StubPromptWarEnv(seed=0)
        env.reset()
        # First two turns are mid-round and get reward 0.
        first = env.step(PromptWarAction(command="PASS"))
        self.assertEqual(first.reward, 0.0)
        env.step(PromptWarAction(command="PASS"))
        # Third turn closes the round; the acting agent's reward is the slice.
        third = env.step(PromptWarAction(command="PASS"))
        self.assertGreater(third.reward, 0.0)
        self.assertIn("B", third.last_rewards)

    def test_random_rewards_are_seeded(self):
        a = StubPromptWarEnv(seed=123)
        b = StubPromptWarEnv(seed=123)
        a.reset(); b.reset()
        for _ in range(9):
            obs_a = a.step(PromptWarAction(command="PASS"))
            obs_b = b.step(PromptWarAction(command="PASS"))
            self.assertEqual(obs_a.last_rewards, obs_b.last_rewards)


if __name__ == "__main__":
    unittest.main()
