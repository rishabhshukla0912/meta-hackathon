# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-agent role prompts for PromptWar trainer rollouts."""

from .role_prompts import ROLE_PROMPTS, build_observation_prompt

__all__ = ["ROLE_PROMPTS", "build_observation_prompt"]
