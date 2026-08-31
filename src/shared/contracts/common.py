from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


CONTRACT_VERSION = "2.0.0"
ContractVersion = Literal["2.0.0"]
NonEmptyStr = Annotated[str, Field(min_length=1)]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionScope(ContractModel):
    """Immutable identifiers shared by runtime, messages, traces, memory, and RAG."""

    session_id: NonEmptyStr
    thread_id: NonEmptyStr
    snapshot_id: NonEmptyStr | None = None
    round_index: int = Field(default=0, ge=0)
    scenario_id: NonEmptyStr | None = None
    scenario_tags: list[NonEmptyStr] = Field(default_factory=list)
    supi: NonEmptyStr | None = None

    @field_validator("scenario_tags")
    @classmethod
    def validate_unique_tags(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("scenario_tags must be unique")
        return values


__all__ = [
    "CONTRACT_VERSION",
    "ContractModel",
    "ContractVersion",
    "ExecutionScope",
    "JsonValue",
    "NonEmptyStr",
    "utc_now",
]
