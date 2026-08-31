from __future__ import annotations

from typing import Literal, Sequence

from pydantic import Field, JsonValue, model_validator

from .common import ContractModel, NonEmptyStr


MessageRole = Literal["system", "user", "assistant", "tool"]
ToolCallStatus = Literal["success", "error"]


class ToolCall(ContractModel):
    id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class AgentMessage(ContractModel):
    """Model-independent conversational message with native tool-call fields."""

    role: MessageRole
    content: str = ""
    name: NonEmptyStr | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: NonEmptyStr | None = None
    tool_status: ToolCallStatus | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> "AgentMessage":
        if self.role in {"system", "user"}:
            if self.tool_calls or self.tool_call_id is not None or self.tool_status is not None:
                raise ValueError(f"{self.role} messages cannot contain tool protocol fields")
            if not self.content.strip():
                raise ValueError(f"{self.role} message content must not be empty")
        elif self.role == "assistant":
            if self.tool_call_id is not None or self.tool_status is not None or self.name is not None:
                raise ValueError("assistant messages cannot contain tool-result fields")
            if not self.content.strip() and not self.tool_calls:
                raise ValueError("assistant messages require content or tool_calls")
            call_ids = [call.id for call in self.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("assistant tool_call ids must be unique within a message")
        else:
            if self.tool_calls:
                raise ValueError("tool messages cannot contain tool_calls")
            if self.tool_call_id is None or self.name is None or self.tool_status is None:
                raise ValueError("tool messages require tool_call_id, name, and tool_status")
        return self


def validate_message_sequence(
    messages: Sequence[AgentMessage],
    *,
    require_final_assistant: bool,
) -> None:
    if not messages:
        raise ValueError("message sequence must not be empty")
    if messages[0].role != "system":
        raise ValueError("message sequence must start with system")
    if not any(message.role == "user" for message in messages):
        raise ValueError("message sequence must contain a user message")

    declared: dict[str, str] = {}
    resolved: set[str] = set()
    for message in messages:
        for call in message.tool_calls:
            if call.id in declared:
                raise ValueError(f"duplicate tool_call id across conversation: {call.id}")
            declared[call.id] = call.name
        if message.role != "tool":
            continue
        call_id = str(message.tool_call_id)
        if call_id not in declared:
            raise ValueError(f"tool result references unknown call id: {call_id}")
        if call_id in resolved:
            raise ValueError(f"tool call has multiple results: {call_id}")
        if declared[call_id] != message.name:
            raise ValueError(
                f"tool result name mismatch for {call_id}: expected {declared[call_id]}, got {message.name}"
            )
        resolved.add(call_id)

    if require_final_assistant:
        unresolved = sorted(set(declared) - resolved)
        if unresolved:
            raise ValueError("successful conversation has unresolved tool calls: " + ", ".join(unresolved))
        if messages[-1].role != "assistant" or not messages[-1].content.strip():
            raise ValueError("successful conversation must end with a textual assistant response")


__all__ = [
    "AgentMessage",
    "MessageRole",
    "ToolCall",
    "ToolCallStatus",
    "validate_message_sequence",
]
