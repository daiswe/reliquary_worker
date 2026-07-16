"""Subset of miner→validator submission schemas for the worker."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from worker.constants import M_ROLLOUTS


class UnsignedRolloutSubmission(BaseModel):
    """Rollout payload without commit signature — orchestrator signs server-side."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=1)
    reward: FiniteFloat
    commit: dict[str, Any]
    env_name: str


class WorkerSubmitRequest(BaseModel):
    """Worker → orchestrator batch after one full generation pass."""

    model_config = ConfigDict(extra="forbid")

    prompt_idx: int = Field(..., ge=0)
    window_n: int = Field(..., ge=0)
    env_name: str
    merkle_root: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    rollouts: list[UnsignedRolloutSubmission]

    @field_validator("rollouts")
    @classmethod
    def _rollout_count_is_m(cls, v: list) -> list:
        if len(v) != M_ROLLOUTS:
            raise ValueError(
                f"rollouts must have exactly {M_ROLLOUTS} entries, got {len(v)}"
            )
        return v


class WorkerNextResponse(BaseModel):
    """Orchestrator → worker assignment for one generation pass."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_n: int = Field(..., ge=0)
    checkpoint_hash: str = ""
    window_n: int = Field(..., ge=0)
    randomness: str
    drand_round: int = Field(default=0, ge=0)
    prompt_idx: int = Field(..., ge=0)
    env_name: str
    miner_hotkey: str
    protocol_version: int = 1
