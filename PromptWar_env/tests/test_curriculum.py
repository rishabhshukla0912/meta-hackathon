# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from PromptWar_env.server.curriculum import (
    DEFAULT_STAGE,
    STAGES,
    stage_for_index,
)


class CurriculumTest(unittest.TestCase):
    def test_default_stage_is_standard(self):
        self.assertEqual(DEFAULT_STAGE, 2)

    def test_stage_1_is_lenient_one_round(self):
        s = stage_for_index(1)
        self.assertEqual(s.max_rounds, 1)
        self.assertTrue(s.lenient)
        self.assertFalse(s.strict)

    def test_stage_2_is_standard_three_rounds(self):
        s = stage_for_index(2)
        self.assertEqual(s.max_rounds, 3)
        self.assertFalse(s.lenient)
        self.assertFalse(s.strict)

    def test_stage_3_is_strict(self):
        s = stage_for_index(3)
        self.assertEqual(s.max_rounds, 3)
        self.assertTrue(s.strict)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            stage_for_index(99)

    def test_all_stages_present(self):
        self.assertEqual(set(STAGES.keys()), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
