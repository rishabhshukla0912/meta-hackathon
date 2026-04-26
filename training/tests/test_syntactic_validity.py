# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from training.syntactic_validity import (
    gate_decision,
    measure_role_validity,
)


def all_valid_sampler(role, system_prompt, user_message):
    return "APPEND: be careful and helpful"


def all_invalid_sampler(role, system_prompt, user_message):
    return "this is not a real edit command"


def per_role_mixed_sampler(role, system_prompt, user_message):
    if role == "A":
        return "APPEND: cite sources"
    if role == "S":
        return "REPLACE: balanced --> safe"
    return "garbage output from base model"


class SyntacticValidityTest(unittest.TestCase):
    def test_all_valid_ships(self):
        reports = measure_role_validity(all_valid_sampler, n_per_role=5)
        gate = gate_decision(reports)
        self.assertEqual(gate["decision"], "SHIP")
        self.assertEqual(gate["weakest_rate"], 1.0)

    def test_all_invalid_pivots(self):
        reports = measure_role_validity(all_invalid_sampler, n_per_role=5)
        gate = gate_decision(reports)
        self.assertEqual(gate["decision"], "PIVOT_TO_TIMEWARP")
        self.assertEqual(gate["weakest_rate"], 0.0)

    def test_partial_returns_per_role_breakdown(self):
        reports = measure_role_validity(per_role_mixed_sampler, n_per_role=10)
        self.assertEqual(reports["A"].rate, 1.0)
        self.assertEqual(reports["S"].rate, 1.0)
        self.assertEqual(reports["B"].rate, 0.0)
        gate = gate_decision(reports)
        self.assertEqual(gate["weakest_role"], "B")


if __name__ == "__main__":
    unittest.main()
