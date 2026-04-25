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
    AccuracyRubric,
    BrevityRubric,
    REFUSAL_PATTERNS,
    RubricContext,
    SafetyRubric,
    brevity_per_response,
    qa_accuracy_reward,
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


class MockRubricFallbackTest(unittest.TestCase):
    def test_accuracy_responds_to_truth_keywords(self):
        rubric = AccuracyRubric()
        bare = rubric(make_ctx("nothing useful here"))
        truthful = rubric(
            make_ctx("be truthful, factual, and cite evidence when uncertain")
        )
        self.assertGreater(truthful, bare)

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


class QaAccuracyRewardTest(unittest.TestCase):
    """Spec the continuous QA reward used by AccuracyRubric when the
    Consumer Model is loaded."""

    def test_exact_match_is_one(self):
        self.assertEqual(qa_accuracy_reward("Paris", ["Paris"]), 1.0)

    def test_total_miss_is_zero(self):
        self.assertEqual(qa_accuracy_reward("London", ["Paris"]), 0.0)

    def test_empty_response_is_zero(self):
        self.assertEqual(qa_accuracy_reward("", ["Paris"]), 0.0)

    def test_empty_references_is_zero(self):
        self.assertEqual(qa_accuracy_reward("Paris", []), 0.0)

    def test_full_sentence_with_answer_scores_high(self):
        # "The capital of France is Paris." — containment hits, F1 partial.
        score = qa_accuracy_reward("The capital of France is Paris.", ["Paris"])
        self.assertGreater(score, 0.6)
        self.assertLessEqual(score, 1.0)

    def test_paraphrase_without_answer_token_scores_low(self):
        # No "Paris" anywhere — containment misses, F1 ≈ 0.
        score = qa_accuracy_reward("It's the city of light, of course.", ["Paris"])
        self.assertLess(score, 0.2)

    def test_negation_near_answer_is_penalised(self):
        positive = qa_accuracy_reward("The capital is Paris.", ["Paris"])
        negated  = qa_accuracy_reward("The capital is not Paris.", ["Paris"])
        self.assertGreater(positive, negated)
        # 0.5 penalty applied; with positive ≈ 0.55+0.4-or-so before penalty
        # the negated version should be markedly lower (often 0).
        self.assertLessEqual(negated, positive - 0.4)

    def test_multi_reference_takes_max(self):
        # "Paris" hits the second reference exactly; first one misses.
        score = qa_accuracy_reward("Paris", ["the capital", "Paris"])
        self.assertEqual(score, 1.0)

    def test_case_and_punctuation_insensitive(self):
        score = qa_accuracy_reward("paris.", ["Paris"])
        self.assertEqual(score, 1.0)

    def test_articles_do_not_inflate_f1(self):
        # Article stripping in F1 means a response made of pure articles
        # cannot earn token overlap with a content-word reference.
        score = qa_accuracy_reward("the the the the", ["paris"])
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
