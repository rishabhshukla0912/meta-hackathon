# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Curriculum stages (§4.7).

Stage 1 (warm-up): 1 round (3 turns), lenient grading.
Stage 2 (standard): 3 rounds (9 turns), standard grading.
Stage 3 (strict):   3 rounds, strict grading + anti-collusion checks.

The trainer signals the environment via POST /curriculum_stage. The env
applies the new stage to subsequent ``reset()`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_rounds: int
    lenient: bool
    strict: bool


STAGE_1 = CurriculumStage(name="warmup", max_rounds=1, lenient=True, strict=False)
STAGE_2 = CurriculumStage(name="standard", max_rounds=3, lenient=False, strict=False)
STAGE_3 = CurriculumStage(name="strict", max_rounds=3, lenient=False, strict=True)

STAGES = {1: STAGE_1, 2: STAGE_2, 3: STAGE_3}
DEFAULT_STAGE = 2


def stage_for_index(idx: int) -> CurriculumStage:
    if idx not in STAGES:
        raise ValueError(f"unknown curriculum stage {idx}; expected one of {sorted(STAGES)}")
    return STAGES[idx]


__all__ = [
    "CurriculumStage",
    "DEFAULT_STAGE",
    "STAGES",
    "STAGE_1",
    "STAGE_2",
    "STAGE_3",
    "stage_for_index",
]
