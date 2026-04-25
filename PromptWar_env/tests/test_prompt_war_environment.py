import unittest

from PromptWar_env.models import PromptWarAction
from PromptWar_env.server.prompt_war_environment import (
    MAX_ROUNDS,
    PROMPT_CEILING_TOKENS,
    PROMPT_FLOOR_TOKENS,
    PromptWarEnvironment,
)


class PromptWarEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = PromptWarEnvironment()

    def test_reset_initializes_neutral_state(self):
        obs = self.env.reset()

        self.assertEqual(obs.active_agent, "A")
        self.assertEqual(obs.round_idx, 0)
        self.assertEqual(obs.turn_idx, 0)
        self.assertFalse(obs.done)
        self.assertFalse(obs.edit_rejected)
        self.assertGreaterEqual(
            obs.metadata["prompt_tokens"],
            PROMPT_FLOOR_TOKENS,
        )

    def test_nine_valid_turns_complete_episode(self):
        obs = self.env.reset()

        for _ in range(MAX_ROUNDS * 3):
            obs = self.env.step(PromptWarAction(command="PASS"))

        self.assertTrue(obs.done)
        self.assertEqual(obs.round_idx, MAX_ROUNDS)
        self.assertEqual(self.env.state.step_count, MAX_ROUNDS * 3)
        self.assertEqual(len(obs.metadata["rewards_by_round"]), MAX_ROUNDS)

    def test_pass_advances_turn(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="PASS"))

        self.assertEqual(obs.active_agent, "S")
        self.assertEqual(obs.turn_idx, 1)
        self.assertFalse(obs.edit_rejected)

    def test_append_empty_rejects(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="APPEND:   "))

        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "empty append")

    def test_append_over_ceiling_rejects(self):
        self.env.reset()
        self.env._prompt_state.shared_prompt = " ".join(
            f"base{i}" for i in range(PROMPT_CEILING_TOKENS - 10)
        )
        obs = self.env.step(
            PromptWarAction(command="APPEND: " + " ".join(f"extra{i}" for i in range(20)))
        )

        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "would exceed ceiling")

    def test_invalid_regex_rejects(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="DEL: ["))

        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "invalid regex")

    def test_delete_no_match_passes_without_rejection(self):
        before = self.env.reset().shared_prompt
        obs = self.env.step(PromptWarAction(command="DEL: DOES_NOT_EXIST_IN_PROMPT"))

        self.assertFalse(obs.edit_rejected)
        self.assertEqual(obs.shared_prompt, before)

    def test_delete_below_floor_rejects(self):
        self.env.reset()
        obs = self.env.step(PromptWarAction(command="DEL: .+"))

        self.assertTrue(obs.edit_rejected)
        self.assertEqual(obs.rejection_reason, "would violate floor")

    def test_replace_missing_old_text_passes_without_rejection(self):
        before = self.env.reset().shared_prompt
        obs = self.env.step(
            PromptWarAction(command="REPLACE: missing old text --> replacement")
        )

        self.assertFalse(obs.edit_rejected)
        self.assertEqual(obs.shared_prompt, before)


if __name__ == "__main__":
    unittest.main()
