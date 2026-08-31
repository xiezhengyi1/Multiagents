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
    allow_terminal_unresolved_tool_calls: bool = False,
) -> None:
    if not messages:
        raise ValueError("message sequence must not be empty")
    if messages[0].role != "system":
        raise ValueError("message sequence must start with system")
    if not any(message.role == "user" for message in messages):
        raise ValueError("message sequence must contain a user message")

    pending: dict[str, str] = {}
    declared_ids: set[str] = set()
    for index, message in enumerate(messages):
        if index > 0 and message.role == "system":
            raise ValueError("system message may only appear at the start")

        if pending:
            if message.role != "tool":
                raise ValueError(
                    "all pending tool calls must be resolved before the conversation continues"
                )
            call_id = str(message.tool_call_id)
            if call_id not in pending:
                raise ValueError(f"tool result references a non-pending call id: {call_id}")
            expected_name = pending.pop(call_id)
            if expected_name != message.name:
                raise ValueError(
                    f"tool result name mismatch for {call_id}: "
                    f"expected {expected_name}, got {message.name}"
                )
            continue

        if message.role == "tool":
            raise ValueError(f"tool result has no pending call: {message.tool_call_id}")
        for call in message.tool_calls:
            if call.id in declared_ids:
                raise ValueError(f"duplicate tool_call id across conversation: {call.id}")
            declared_ids.add(call.id)
            pending[call.id] = call.name

    if pending and not allow_terminal_unresolved_tool_calls:
        raise ValueError("conversation has unresolved tool calls: " + ", ".join(sorted(pending)))
    if require_final_assistant:
        if messages[-1].role != "assistant" or not messages[-1].content.strip():
            raise ValueError("successful conversation must end with a textual assistant response")


__all__ = [
    "AgentMessage",
    "MessageRole",
    "ToolCall",
    "ToolCallStatus",
    "validate_message_sequence",
]
