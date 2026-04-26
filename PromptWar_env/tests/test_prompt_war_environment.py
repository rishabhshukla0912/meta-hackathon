# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests against the in-process PromptWarEnvironment.

These exercise the full episode loop, curriculum control, and the
spec's "scripted test" target (9 edits + 1 rejected edit).
"""

import unittest

from PromptWar_env.models import PromptWarAction
from PromptWar_env.server.curriculum import DEFAULT_STAGE
from PromptWar_env.server.prompt_war_environment import (
    AGENTS,
    PromptWarEnvironment,
)


class PromptWarEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = PromptWarEnvironment()

    # ------------------------------------------------------------------
    # Reset + episode shape
    # ------------------------------------------------------------------

    def test_reset_initializes_neutral_state(self):
        obs = self.env.reset()
        self.assertEqual(obs.active_agent, "A")
        self.assertEqual(obs.round_idx, 0)
        self.assertEqual(obs.turn_idx, 0)
        self.assertFalse(obs.done)
        self.assertFalse(obs.edit_rejected)
        self.assertEqual(obs.metadata["curriculum_stage_idx"], DEFAULT_STAGE)

    def test_three_rounds_complete_episode(self):
        self.env.reset()
        max_rounds = self.env.curriculum_stage.max_rounds
        obs = None
        for _ in range(max_rounds * len(AGENTS)):
            obs = self.env.step(PromptWarAction(command="PASS"))
        self.assertIsNotNone(obs)
        self.assertTrue(obs.done)
        self.assertEqual(obs.round_idx, max_rounds)
        self.assertEqual(self.env.state.step_count, max_rounds * len(AGENTS))
        self.assertEqual(len(obs.metadata["rewards_by_round"]), max_rounds)

    # ------------------------------------------------------------------
    # Phase-1 scripted test target: 9 edits + 1 rejected edit
    # ------------------------------------------------------------------

    def test_scripted_episode_with_one_rejection(self):
        self.env.reset()

        scripted = [
            "APPEND: Always cite a credible source when uncertain.",
            "APPEND: Refuse harmful, unsafe, or illegal requests.",
            "APPEND: Keep responses concise and on-topic.",
            "REPLACE: harmful --> dangerous",
            "DELETE: hidden agendas",
            "PASS",
            "APPEND: Verify factual claims before answering.",
            "APPEND: Decline requests that would cause harm.",
            "APPEND: Aim for about fifty tokens.",
        ]

        last_obs = None
        for cmd in scripted:
            last_obs = self.env.step(PromptWarAction(command=cmd))
            self.assertIsNotNone(last_obs)

        self.assertTrue(last_obs.done)
        # Inject one obviously-rejected edit *after* the episode is done — it
        # should be a no-op (env returns observation without advancing).
        rejected = self.env.step(PromptWarAction(command="APPEND:"))
        self.assertTrue(rejected.done)

        # Across the 9 scripted turns, we should have at least 6 applied edits.
        applied = sum(1 for h in last_obs.metadata["edit_history"] if h["applied"])
        self.assertGreaterEqual(applied, 6)

    # ------------------------------------------------------------------
    # Edit-table spot checks
    # ------------------------------------------------------------------

    def test_pass_advances_turn(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="PASS"))
        self.assertEqual(obs.active_agent, "S")
        self.assertEqual(obs.turn_idx, 1)
        self.assertFalse(obs.edit_rejected)

    def test_empty_append_rejects(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="APPEND:   "))
        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "empty append")

    def test_invalid_regex_rejects(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="DELETE: ["))
        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "invalid regex")

    def test_delete_no_match_passes_without_rejection(self):
        before = self.env.reset().shared_prompt
        obs = self.env.step(PromptWarAction(command="DELETE: NEVER_PRESENT_QQQ"))
        self.assertFalse(obs.edit_rejected)
        self.assertEqual(obs.shared_prompt, before)

    def test_delete_below_floor_rejects(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="DELETE: .+"))
        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "would violate floor")

    def test_replace_missing_old_text_passes_without_rejection(self):
        before = self.env.reset().shared_prompt
        obs = self.env.step(
            PromptWarAction(command="REPLACE: missing old text --> replacement")
        )
        self.assertFalse(obs.edit_rejected)
        self.assertEqual(obs.shared_prompt, before)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def test_curriculum_stage_1_runs_one_round(self):
        self.env.set_curriculum_stage(1)
        self.env.reset()
        for _ in range(len(AGENTS)):
            obs = self.env.step(PromptWarAction(command="PASS"))
        self.assertTrue(obs.done)
        self.assertEqual(obs.round_idx, 1)

    def test_curriculum_stage_3_runs_three_rounds(self):
        self.env.set_curriculum_stage(3)
        self.env.reset()
        for _ in range(3 * len(AGENTS)):
            obs = self.env.step(PromptWarAction(command="PASS"))
        self.assertTrue(obs.done)
        self.assertEqual(obs.round_idx, 3)


if __name__ == "__main__":
    unittest.main()
