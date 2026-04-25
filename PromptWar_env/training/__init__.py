# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PromptWar trainer-side package (Person B).

Modules:
- ``stub_env``       Random-reward env mimicking PromptWarEnv shape (smoke tests)
- ``role_router``    Per-turn rollout router (A/S/B)
- ``lora_setup``     Three LoRA adapters on Qwen2.5-3B (peft)
- ``trainers``       Three GRPOTrainer instances, one per role
- ``train``          CLI entrypoint (smoke / live-episode / long-run)
- ``syntactic_validity``  Risk #2 quick check — 50 edits/role, parse rate
"""
