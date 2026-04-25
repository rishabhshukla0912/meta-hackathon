# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the PromptWar environment.

This module creates an HTTP server that exposes the PromptWar environment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints (handled by ``openenv.core.env_server.http_server.create_app``):
    - POST /reset
    - POST /step
    - GET  /state
    - GET  /schema
    - WS   /ws

Endpoints layered on top by this module:
    - POST /curriculum_stage   — set the active stage (1, 2, or 3)
    - GET  /curriculum_stage   — read the active stage
    - POST /consumer/load      — eagerly load the Consumer Model on demand
    - GET  /consumer/status    — report whether the Consumer Model is loaded

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Or:
    python -m server.app --port 8000
"""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover - dep missing
    _OPENENV_IMPORT_ERROR = e
    create_app = None
else:
    _OPENENV_IMPORT_ERROR = None

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from ..models import PromptWarAction, PromptWarObservation
    from .curriculum import STAGES
    from .prompt_war_environment import PromptWarEnvironment
except ImportError:  # pragma: no cover - flat-import fallback
    from models import PromptWarAction, PromptWarObservation
    from server.curriculum import STAGES
    from server.prompt_war_environment import PromptWarEnvironment


# ---------------------------------------------------------------------------
# OpenEnv app + a singleton env handle for the curriculum/consumer endpoints.
# ---------------------------------------------------------------------------

_EAGER_LOAD = os.environ.get("PROMPTWAR_LOAD_CONSUMER_MODEL", "0") == "1"

_environment_singleton = PromptWarEnvironment(eager_load_consumer=_EAGER_LOAD)


def _environment_factory() -> PromptWarEnvironment:
    """Reuse the singleton so /curriculum_stage and /consumer/* see the same env.

    OpenEnv's ``create_app`` accepts a class or factory for environment
    construction; passing a callable here keeps each session sharing one
    Consumer Model instance (which is expensive to load) while still
    isolating per-episode state via ``reset()``.
    """
    return _environment_singleton


def _dump_model(value: Any) -> Any:
    """Serialize Pydantic v1/v2 models and simple stand-ins."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _create_fallback_app() -> FastAPI:
    """Small local server used when Meta OpenEnv is not installed.

    This keeps hackathon/local development usable. When ``openenv-core`` is
    installed, the official OpenEnv HTTP server is used instead.
    """
    fallback_app = FastAPI(title="PromptWar_env")

    @fallback_app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "name": "PromptWar_env",
            "mode": "local-fallback",
            "openenv_error": str(_OPENENV_IMPORT_ERROR),
        }

    @fallback_app.post("/reset")
    def reset() -> Dict[str, Any]:
        return _dump_model(_environment_singleton.reset())

    @fallback_app.post("/step")
    def step(action: PromptWarAction) -> Dict[str, Any]:
        return _dump_model(_environment_singleton.step(action))

    @fallback_app.get("/state")
    def state() -> Dict[str, Any]:
        return _dump_model(_environment_singleton.state)

    @fallback_app.get("/schema")
    def schema() -> Dict[str, Any]:
        return {
            "action": PromptWarAction.model_json_schema(),
            "observation": PromptWarObservation.model_json_schema(),
        }

    return fallback_app


if create_app is None:
    app = _create_fallback_app()
else:
    app = create_app(
        _environment_factory,
        PromptWarAction,
        PromptWarObservation,
        env_name="PromptWar_env",
        max_concurrent_envs=1,
    )


# ---------------------------------------------------------------------------
# Curriculum stage endpoint (§4.7)
# ---------------------------------------------------------------------------

class CurriculumStageRequest(BaseModel):
    stage: int = Field(..., description="Curriculum stage index (1, 2, or 3)")


class CurriculumStageResponse(BaseModel):
    stage: int
    name: str
    max_rounds: int
    lenient: bool
    strict: bool


@app.post("/curriculum_stage", response_model=CurriculumStageResponse)
def set_curriculum_stage(req: CurriculumStageRequest) -> CurriculumStageResponse:
    if req.stage not in STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown stage {req.stage}; expected one of {sorted(STAGES)}",
        )
    stage = _environment_singleton.set_curriculum_stage(req.stage)
    return CurriculumStageResponse(
        stage=req.stage,
        name=stage.name,
        max_rounds=stage.max_rounds,
        lenient=stage.lenient,
        strict=stage.strict,
    )


@app.get("/curriculum_stage", response_model=CurriculumStageResponse)
def get_curriculum_stage() -> CurriculumStageResponse:
    stage = _environment_singleton.curriculum_stage
    return CurriculumStageResponse(
        stage=_environment_singleton.curriculum_stage_idx,
        name=stage.name,
        max_rounds=stage.max_rounds,
        lenient=stage.lenient,
        strict=stage.strict,
    )


# ---------------------------------------------------------------------------
# Consumer Model lifecycle endpoints (§4.5)
# ---------------------------------------------------------------------------

@app.post("/consumer/load")
def load_consumer() -> Dict[str, Any]:
    consumer = _environment_singleton._consumer  # noqa: SLF001 - intentional
    ok = consumer.load()
    return {
        "available": ok,
        "load_error": consumer.load_error,
    }


@app.get("/consumer/status")
def consumer_status() -> Dict[str, Any]:
    consumer = _environment_singleton._consumer  # noqa: SLF001 - intentional
    return {
        "available": consumer.available,
        "load_error": consumer.load_error,
        "model_id": consumer.config.model_id,
    }


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app via uvicorn (no Docker)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
