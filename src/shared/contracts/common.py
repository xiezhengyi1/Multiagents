from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, strict=True),
]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventScope(ContractModel):
    episode_id: NonEmptyStr
    round_index: int = Field(ge=0)


__all__ = [
    "ContractModel",
    "EventScope",
    "JsonValue",
    "NonEmptyStr",
    "utc_now",
]
