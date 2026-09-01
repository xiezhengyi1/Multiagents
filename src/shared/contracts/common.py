from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator


CONTRACT_VERSION = "2.0.0"
ContractVersion = Literal["2.0.0"]
NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, strict=True),
]


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


def validate_scope_lineage(upstream: ExecutionScope, downstream: ExecutionScope) -> None:
    lineage_fields = (
        "session_id",
        "thread_id",
        "snapshot_id",
        "round_index",
        "scenario_id",
        "scenario_tags",
    )
    changed = [
        field_name
        for field_name in lineage_fields
        if getattr(upstream, field_name) != getattr(downstream, field_name)
    ]
    if changed:
        raise ValueError("artifact response changed scope lineage fields: " + ", ".join(changed))
    if upstream.supi is not None and downstream.supi != upstream.supi:
        raise ValueError("artifact response changed an established scope.supi")


__all__ = [
    "CONTRACT_VERSION",
    "ContractModel",
    "ContractVersion",
    "ExecutionScope",
    "JsonValue",
    "NonEmptyStr",
    "utc_now",
    "validate_scope_lineage",
]
