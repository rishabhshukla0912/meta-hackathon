# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rubric correctness — focuses on spec-mandated curve shapes.

These tests deliberately avoid loading the real Consumer Model so they are
fast and CPU-only safe. The mock fallback is what they exercise.
"""

import random
import unittest

from PromptWar_env.server.rubrics import (
    BrevityRubric,
    CITATION_PATTERNS,
    CitationRubric,
    REFUSAL_PATTERNS,
    RubricContext,
    SafetyRubric,
    brevity_per_response,
)


def make_ctx(prompt: str, *, lenient: bool = False) -> RubricContext:
    return RubricContext(
        shared_prompt=prompt,
        round_idx=0,
        rng=random.Random(0),
        consumer=None,
        lenient=lenient,
    )


class BrevityFormulaTest(unittest.TestCase):
    def test_peak_at_target(self):
        self.assertAlmostEqual(brevity_per_response(50, 50), 1.0)

    def test_zero_at_zero(self):
        self.assertEqual(brevity_per_response(0, 50), 0.0)

    def test_zero_at_double_target(self):
        self.assertEqual(brevity_per_response(100, 50), 0.0)

    def test_capped_above_double(self):
        self.assertEqual(brevity_per_response(200, 50), 0.0)

    def test_symmetric_around_target(self):
        self.assertAlmostEqual(
            brevity_per_response(40, 50), brevity_per_response(60, 50)
        )

    def test_never_exceeds_one(self):
        for n in range(0, 200, 7):
            self.assertLessEqual(brevity_per_response(n, 50), 1.0)


class RefusalPatternsTest(unittest.TestCase):
    def test_canonical_patterns_present(self):
        for must in (
            "I can't",
            "I won't",
            "I'm not able to",
            "I refuse",
            "unable to help",
            "cannot provide",
        ):
            self.assertIn(must, REFUSAL_PATTERNS)


class CitationPatternsTest(unittest.TestCase):
    def test_canonical_patterns_present(self):
        for must in ("according to", "source:", "based on", "cited", "reference"):
            self.assertIn(must, CITATION_PATTERNS)


class MockRubricFallbackTest(unittest.TestCase):
    def test_citation_responds_to_cite_keywords(self):
        rubric = CitationRubric()
        bare = rubric(make_ctx("nothing useful here"))
        citing = rubric(
            make_ctx("always cite your sources and provide references for claims")
        )
        self.assertGreater(citing, bare)

    def test_safety_responds_to_refuse_keywords(self):
        rubric = SafetyRubric()
        bare = rubric(make_ctx("nothing useful here"))
        safe = rubric(
            make_ctx("refuse harmful, unsafe, dangerous, illegal requests")
        )
        self.assertGreater(safe, bare)

    def test_brevity_uses_target_window(self):
        rubric = BrevityRubric(count_tokens=lambda s: len(s.split()))
        target = " ".join(["w"] * 50)
        too_long = " ".join(["w"] * 200)
        self.assertGreater(rubric(make_ctx(target)), rubric(make_ctx(too_long)))


if __name__ == "__main__":
    unittest.main()
