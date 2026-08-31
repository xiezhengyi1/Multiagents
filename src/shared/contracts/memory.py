from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import ContractModel, ExecutionScope, NonEmptyStr, utc_now


MemoryKind = Literal["event", "summary"]


class MemoryRecord(ContractModel):
    memory_id: NonEmptyStr
    scope: ExecutionScope
    kind: MemoryKind
    source_agent: NonEmptyStr
    role: NonEmptyStr
    content: NonEmptyStr
    source_trace_ids: list[NonEmptyStr] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryQuery(ContractModel):
    scope: ExecutionScope
    text: NonEmptyStr
    top_k: int = Field(default=3, ge=1, le=20)


class MemoryBundle(ContractModel):
    query: MemoryQuery
    short_term: list[MemoryRecord]
    long_term: list[MemoryRecord]


__all__ = ["MemoryBundle", "MemoryKind", "MemoryQuery", "MemoryRecord"]
