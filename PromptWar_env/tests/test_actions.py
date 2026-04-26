# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cover every row of the §4.3.3 edit-case table.

Uses a simple word-count tokenizer so the case table can be exercised
without loading the real Qwen tokenizer.
"""

import re
import unittest

from PromptWar_env.server.actions import (
    EDIT_TOKEN_BUDGET,
    PROMPT_CEILING_TOKENS,
    PROMPT_FLOOR_TOKENS,
    REJECT_BUDGET,
    REJECT_CEILING,
    REJECT_EMPTY_APPEND,
    REJECT_FLOOR,
    REJECT_INVALID_COMMAND,
    REJECT_INVALID_REGEX,
    apply_action,
    parse_command,
)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def make_prompt(words: int) -> str:
    # Each "wi" counts as one token under our test counter.
    return " ".join(f"w{i}" for i in range(words))


class ParseCommandTest(unittest.TestCase):
    def test_parses_pass(self):
        action = parse_command("PASS")
        self.assertEqual(action.op, "PASS")

    def test_parses_append(self):
        action = parse_command("APPEND: hello world")
        self.assertEqual(action.op, "APPEND")
        self.assertEqual(action.arg1, "hello world")

    def test_parses_delete(self):
        action = parse_command("DELETE: foo")
        self.assertEqual(action.op, "DELETE")
        self.assertEqual(action.arg1, "foo")

    def test_parses_del_alias(self):
        action = parse_command("DEL: foo")
        self.assertEqual(action.op, "DELETE")

    def test_parses_replace(self):
        action = parse_command("REPLACE: old --> new")
        self.assertEqual(action.op, "REPLACE")
        self.assertEqual(action.arg1, "old")
        self.assertEqual(action.arg2, "new")

    def test_replace_missing_arrow_is_parse_error(self):
        action = parse_command("REPLACE: old new")
        self.assertEqual(action.op, "REPLACE")
        self.assertTrue(action.parse_error)

    def test_unknown_command_is_invalid(self):
        action = parse_command("FROBNICATE: stuff")
        self.assertEqual(action.op, "INVALID")
        self.assertEqual(action.parse_error, REJECT_INVALID_COMMAND)


class AppendCaseTableTest(unittest.TestCase):
    def setUp(self):
        self.prompt = make_prompt(PROMPT_FLOOR_TOKENS + 5)

    def test_normal_append_applies(self):
        result = apply_action(self.prompt, parse_command("APPEND: extra"), word_count)
        self.assertTrue(result.applied)
        self.assertFalse(result.rejected)
        self.assertIn("extra", result.new_prompt)

    def test_empty_append_rejects(self):
        result = apply_action(self.prompt, parse_command("APPEND:    "), word_count)
        self.assertTrue(result.rejected)
        self.assertEqual(result.reason, REJECT_EMPTY_APPEND)

    def test_over_budget_append_rejects(self):
        big = " ".join(f"x{i}" for i in range(EDIT_TOKEN_BUDGET + 5))
        result = apply_action(self.prompt, parse_command(f"APPEND: {big}"), word_count)
        self.assertTrue(result.rejected)
        self.assertEqual(result.reason, REJECT_BUDGET)

    def test_over_ceiling_append_rejects(self):
        near_ceiling = make_prompt(PROMPT_CEILING_TOKENS - 5)
        addition = " ".join(f"y{i}" for i in range(20))
        result = apply_action(
            near_ceiling, parse_command(f"APPEND: {addition}"), word_count
        )
        self.assertTrue(result.rejected)
        self.assertEqual(result.reason, REJECT_CEILING)


class DeleteCaseTableTest(unittest.TestCase):
    def setUp(self):
        # Has a "ZZZ" anchor so we can target it precisely.
        self.prompt = make_prompt(PROMPT_FLOOR_TOKENS + 10) + " ZZZ"

    def test_invalid_regex_rejects(self):
        result = apply_action(self.prompt, parse_command("DELETE: ["), word_count)
        self.assertTrue(result.rejected)
        self.assertEqual(result.reason, REJECT_INVALID_REGEX)

    def test_no_match_passes_without_rejection(self):
        result = apply_action(
            self.prompt, parse_command("DELETE: NEVER_PRESENT_QQQ"), word_count
        )
        self.assertFalse(result.rejected)
        self.assertEqual(result.new_prompt, self.prompt)

    def test_match_above_floor_applies(self):
        result = apply_action(self.prompt, parse_command("DELETE: ZZZ"), word_count)
        self.assertTrue(result.applied)
        self.assertNotIn("ZZZ", result.new_prompt)

    def test_match_below_floor_rejects(self):
        # Prompt has just enough tokens that deleting almost-all goes under floor.
        prompt = make_prompt(PROMPT_FLOOR_TOKENS + 2)
        result = apply_action(prompt, parse_command("DELETE: w[0-9]+"), word_count)
        # Only first match (w0) is removed; tokens drop from 52 to 51 — still >= 50.
        # So choose a heavier pattern that strips a lot.
        result_big = apply_action(prompt, parse_command("DELETE: .+"), word_count)
        self.assertTrue(result_big.rejected)
        self.assertEqual(result_big.reason, REJECT_FLOOR)
        # And the small-delete sanity branch should still apply cleanly.
        self.assertTrue(result.applied)


class ReplaceCaseTableTest(unittest.TestCase):
    def setUp(self):
        self.prompt = make_prompt(PROMPT_FLOOR_TOKENS + 5) + " TARGET"

    def test_old_text_not_found_passes(self):
        before = self.prompt
        result = apply_action(
            self.prompt,
            parse_command("REPLACE: NOPE --> something"),
            word_count,
        )
        self.assertFalse(result.rejected)
        self.assertEqual(result.new_prompt, before)

    def test_normal_replace_applies(self):
        result = apply_action(
            self.prompt,
            parse_command("REPLACE: TARGET --> replaced"),
            word_count,
        )
        self.assertTrue(result.applied)
        self.assertIn("replaced", result.new_prompt)
        self.assertNotIn("TARGET", result.new_prompt)

    def test_replace_over_budget_rejects(self):
        big = " ".join(f"z{i}" for i in range(EDIT_TOKEN_BUDGET + 5))
        result = apply_action(
            self.prompt,
            parse_command(f"REPLACE: TARGET --> {big}"),
            word_count,
        )
        self.assertTrue(result.rejected)
        self.assertEqual(result.reason, REJECT_BUDGET)

    def test_replace_below_floor_rejects(self):
        prompt = make_prompt(PROMPT_FLOOR_TOKENS + 1) + " HUNK"
        # Replace HUNK with a single-token "x" — drops below floor since the prompt
        # was just barely above.
        # First strip enough to bring tokens close to floor.
        long_old = " ".join(f"w{i}" for i in range(PROMPT_FLOOR_TOKENS - 5))
        prompt_big = long_old + " HUNK"
        # Replacing the long_old prefix with "x" pushes total tokens to 2 < 50.
        result = apply_action(
            prompt_big,
            parse_command(f"REPLACE: {long_old} --> x"),
            word_count,
        )
        self.assertTrue(result.rejected)
        self.assertEqual(result.reason, REJECT_FLOOR)


class PassTest(unittest.TestCase):
    def test_pass_is_no_op(self):
        prompt = make_prompt(PROMPT_FLOOR_TOKENS + 1)
        result = apply_action(prompt, parse_command("PASS"), word_count)
        self.assertTrue(result.applied)
        self.assertFalse(result.rejected)
        self.assertEqual(result.new_prompt, prompt)


if __name__ == "__main__":
    unittest.main()
