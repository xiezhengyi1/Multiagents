from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .common import ContractModel, EventScope, NonEmptyStr

MessageRole = Literal["system", "user", "assistant", "tool"]
TraceStatus = Literal["success", "error"]


class ModelToolCall(ContractModel):
    tool_call_id: NonEmptyStr | None = None
    name: NonEmptyStr | None = None
    arguments: JsonValue


class ModelMessage(ContractModel):
    role: MessageRole
    content: JsonValue = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    tool_call_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> ModelMessage:
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool_calls")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may contain tool_call_id")
        call_ids = [
            call.tool_call_id
            for call in self.tool_calls
            if call.tool_call_id is not None
        ]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "tool_call ids must be unique within one assistant message"
            )
        return self


class ModelToolSpec(ContractModel):
    name: NonEmptyStr
    description: str
    input_schema: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_input_schema(self) -> ModelToolSpec:
        if self.input_schema.get("type") != "object":
            raise ValueError(f"tool {self.name} input_schema.type must be object")
        return self


class TokenUsage(ContractModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class AgentTrace(ContractModel):
    """One trace per actual model request, never per workflow or tool loop."""

    trace_id: NonEmptyStr
    trigger_event_id: NonEmptyStr
    call_index: int = Field(ge=0)
    scope: EventScope
    agent_name: NonEmptyStr
    model_name: NonEmptyStr
    started_at: datetime
    ended_at: datetime
    input_messages: list[ModelMessage] = Field(min_length=1)
    output_message: ModelMessage | None
    tools: list[ModelToolSpec] = Field(default_factory=list)
    status: TraceStatus
    error: NonEmptyStr | None = None
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def validate_trace(self) -> AgentTrace:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.status == "success":
            if self.error is not None:
                raise ValueError("successful trace cannot contain error")
            if self.output_message is None:
                raise ValueError("successful trace requires output_message")
        elif self.error is None:
            raise ValueError("error trace requires error")
        if self.output_message is not None and self.output_message.role != "assistant":
            raise ValueError("model output_message must have role=assistant")
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("trace tool names must be unique")
        exposed = set(tool_names)
        called = {
            call.name
            for message in [
                *self.input_messages,
                *([self.output_message] if self.output_message else []),
            ]
            for call in message.tool_calls
        }
        undeclared = sorted(name for name in called - exposed if name is not None)
        if undeclared:
            raise ValueError(
                "trace contains calls to undeclared tools: " + ", ".join(undeclared)
            )
        return self


__all__ = [
    "AgentTrace",
    "MessageRole",
    "ModelMessage",
    "ModelToolCall",
    "ModelToolSpec",
    "TokenUsage",
    "TraceStatus",
]
