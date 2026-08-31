from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

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
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @field_validator("source_trace_ids")
    @classmethod
    def validate_unique_trace_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("source_trace_ids must be unique")
        return values


class MemoryQuery(ContractModel):
    scope: ExecutionScope
    text: NonEmptyStr
    top_k: Annotated[int, Field(strict=True, ge=1, le=20)] = 3


class MemoryBundle(ContractModel):
    query: MemoryQuery
    short_term: list[MemoryRecord]
    long_term: list[MemoryRecord]

    @model_validator(mode="after")
    def validate_namespaces(self) -> "MemoryBundle":
        short_ids = [record.memory_id for record in self.short_term]
        long_ids = [record.memory_id for record in self.long_term]
        if len(short_ids) != len(set(short_ids)) or len(long_ids) != len(set(long_ids)):
            raise ValueError("memory bundle cannot contain duplicate memory_id values")
        if set(short_ids) & set(long_ids):
            raise ValueError("memory_id cannot appear in both short-term and long-term memory")

        for record in self.short_term:
            if record.kind != "event":
                raise ValueError("short_term memory must contain only event records")
            if record.scope.thread_id != self.query.scope.thread_id:
                raise ValueError("short_term memory must match the query thread_id")

        query_identity = _long_term_identity(self.query.scope)
        for record in self.long_term:
            if record.kind != "summary":
                raise ValueError("long_term memory must contain only summary records")
            if _long_term_identity(record.scope) != query_identity:
                raise ValueError("long_term memory must match the query SUPI or session namespace")
        return self


def _long_term_identity(scope: ExecutionScope) -> tuple[str, str]:
    if scope.supi is not None:
        return ("supi", scope.supi)
    return ("session", scope.session_id)


__all__ = ["MemoryBundle", "MemoryKind", "MemoryQuery", "MemoryRecord"]
