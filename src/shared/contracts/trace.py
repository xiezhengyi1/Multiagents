from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .common import CONTRACT_VERSION, ContractModel, ContractVersion, ExecutionScope, NonEmptyStr
from .messages import AgentMessage, validate_message_sequence
from .tools import ToolCatalog


TraceSource = Literal["observed_llm", "deterministic_runtime"]
TraceStatus = Literal["success", "error"]
TargetKind = Literal[
    "observed_model_output",
    "validated_output",
    "compiled_output",
    "deterministic_output",
]
TargetProducer = Literal["model", "validator", "compiler", "runtime"]


class TraceTarget(ContractModel):
    kind: TargetKind
    producer: TargetProducer
    schema_name: NonEmptyStr
    output_schema: dict[str, JsonValue]
    value: JsonValue

    @property
    def sft_eligible(self) -> bool:
        return self.kind == "observed_model_output"

    @model_validator(mode="after")
    def validate_target_provenance(self) -> "TraceTarget":
        expected = {
            "observed_model_output": "model",
            "validated_output": "validator",
            "compiled_output": "compiler",
            "deterministic_output": "runtime",
        }[self.kind]
        if self.producer != expected:
            raise ValueError(f"target {self.kind} must be produced by {expected}")
        return self


class TraceMetadata(ContractModel):
    path_label: NonEmptyStr
    prompt_name: NonEmptyStr
    prompt_version: NonEmptyStr
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class AgentInvocation(ContractModel):
    messages: list[AgentMessage] = Field(min_length=1)
    metadata: TraceMetadata

    @model_validator(mode="after")
    def validate_invocation_messages(self) -> "AgentInvocation":
        if any(message.role == "system" for message in self.messages):
            raise ValueError("system prompt belongs to the agent contract, not invocation messages")
        validate_message_sequence(
            [AgentMessage(role="system", content="runtime-owned system prompt"), *self.messages],
            require_final_assistant=False,
        )
        if self.messages[-1].role != "user":
            raise ValueError("agent invocation must end with a user message")
        return self


class AgentRunResult(ContractModel):
    messages: list[AgentMessage] = Field(min_length=1)
    structured_response: JsonValue


class AgentTrace(ContractModel):
    schema_version: ContractVersion = CONTRACT_VERSION
    trace_id: NonEmptyStr
    agent_name: NonEmptyStr
    model_name: NonEmptyStr
    source: TraceSource
    scope: ExecutionScope
    started_at: datetime
    ended_at: datetime
    tool_catalog: ToolCatalog
    response_schema_name: NonEmptyStr
    response_schema: dict[str, JsonValue]
    messages: list[AgentMessage] = Field(min_length=2)
    targets: list[TraceTarget] = Field(default_factory=list)
    status: TraceStatus
    error: str | None = None
    metadata: TraceMetadata

    @model_validator(mode="after")
    def validate_trace(self) -> "AgentTrace":
        if self.ended_at < self.started_at:
            raise ValueError("trace ended_at cannot precede started_at")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful trace cannot contain error")
        if self.status == "error" and not str(self.error or "").strip():
            raise ValueError("error trace requires error text")
        validate_message_sequence(
            self.messages,
            require_final_assistant=self.status == "success",
            allow_terminal_unresolved_tool_calls=self.status == "error",
        )
        catalog_names = {tool.name for tool in self.tool_catalog.tools}
        called_names = {
            call.name
            for message in self.messages
            for call in message.tool_calls
        }
        undeclared = sorted(called_names - catalog_names)
        if undeclared:
            raise ValueError("trace calls undeclared tools: " + ", ".join(undeclared))
        observed = [target for target in self.targets if target.kind == "observed_model_output"]
        if self.source == "observed_llm" and self.status == "success" and len(observed) != 1:
            raise ValueError("successful observed_llm trace requires exactly one observed_model_output")
        if self.source == "deterministic_runtime" and observed:
            raise ValueError("deterministic_runtime trace cannot claim observed model output")
        return self


__all__ = [
    "AgentInvocation",
    "AgentRunResult",
    "AgentTrace",
    "TargetKind",
    "TargetProducer",
    "TraceMetadata",
    "TraceSource",
    "TraceStatus",
    "TraceTarget",
]
