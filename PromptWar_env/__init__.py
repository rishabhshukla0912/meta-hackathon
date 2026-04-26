# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PromptWar OpenEnv environment.

The :mod:`PromptWar_env.client` module pulls in ``openenv``; we resolve it
lazily so unit tests of the pure-Python modules (actions, rubrics,
curriculum, role prompts) can run on machines that don't yet have
``openenv-core`` installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import MyAction, MyObservation, PromptWarAction, PromptWarObservation

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .client import MyEnv, PromptWarEnv


def __getattr__(name: str):
    if name in {"MyEnv", "PromptWarEnv"}:
        from .client import MyEnv, PromptWarEnv  # noqa: F401

        return {"MyEnv": MyEnv, "PromptWarEnv": PromptWarEnv}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MyAction",
    "MyObservation",
    "MyEnv",
    "PromptWarAction",
    "PromptWarObservation",
    "PromptWarEnv",
]
