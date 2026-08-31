from __future__ import annotations

import json
from datetime import datetime

from pydantic import Field, JsonValue, model_validator

from shared.contracts import (
    CONTRACT_VERSION,
    AgentMessage,
    ContractModel,
    ContractVersion,
    ExecutionScope,
    NonEmptyStr,
    ToolCatalog,
    TraceMetadata,
    TraceStatus,
    TraceTarget,
    validate_message_sequence,
)


class SftExampleV2(ContractModel):
    """Model-independent SFT example projected from one validated observed trace."""

    schema_version: ContractVersion = CONTRACT_VERSION
    sample_id: NonEmptyStr
    trace_id: NonEmptyStr
    agent_name: NonEmptyStr
    scope: ExecutionScope
    messages: list[AgentMessage] = Field(min_length=2)
    tool_catalog: ToolCatalog
    response_schema_name: NonEmptyStr
    response_schema: dict[str, JsonValue]
    observed_target: TraceTarget
    quality_targets: list[TraceTarget] = Field(min_length=1)
    metadata: TraceMetadata

    @model_validator(mode="after")
    def validate_example(self) -> "SftExampleV2":
        validate_message_sequence(self.messages, require_final_assistant=True)
        if self.observed_target.kind != "observed_model_output" or not self.observed_target.sft_eligible:
            raise ValueError("SFT example requires one eligible observed_model_output target")
        if self.observed_target.schema_name != self.response_schema_name:
            raise ValueError("observed target schema_name must match the response schema")
        if self.observed_target.output_schema != self.response_schema:
            raise ValueError("observed target output_schema must match the response schema")
        if any(target.kind not in {"validated_output", "compiled_output"} for target in self.quality_targets):
            raise ValueError("quality_targets must be validator/compiler outputs")
        catalog_names = {tool.name for tool in self.tool_catalog.tools}
        called_names = {
            call.name
            for message in self.messages
            for call in message.tool_calls
        }
        undeclared = sorted(called_names - catalog_names)
        if undeclared:
            raise ValueError("SFT example calls undeclared tools: " + ", ".join(undeclared))
        for message in self.messages:
            if "<think>" in message.content.lower() or "</think>" in message.content.lower():
                raise ValueError("SFT example contains private reasoning tags")
        final_value = json.loads(self.messages[-1].content)
        if final_value != self.observed_target.value:
            raise ValueError("final assistant JSON must equal observed target")
        return self


class Qwen3SftRecord(ContractModel):
    """Qwen3/Hermes-compatible messages and OpenAI-style function schemas."""

    schema_version: ContractVersion = CONTRACT_VERSION
    sample_id: NonEmptyStr
    messages: list[dict[str, JsonValue]] = Field(min_length=2)
    tools: list[dict[str, JsonValue]]
    metadata: dict[str, JsonValue]


class ProjectedTraceRecord(ContractModel):
    """Read-only evaluation view; never used as the SFT source contract."""

    trace_id: NonEmptyStr
    timestamp: datetime
    agent_name: NonEmptyStr
    scope: ExecutionScope
    model_name: NonEmptyStr
    messages: list[AgentMessage] = Field(min_length=2)
    tool_calls: list[dict[str, JsonValue]]
    tool_results: list[dict[str, JsonValue]]
    structured_response: JsonValue | None
    status: TraceStatus
    error: str | None = None
    path_label: NonEmptyStr
    quality_targets: list[TraceTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_error(self) -> "ProjectedTraceRecord":
        if self.status == "success" and self.error is not None:
            raise ValueError("successful projected trace cannot contain an error")
        if self.status == "error" and not str(self.error or "").strip():
            raise ValueError("failed projected trace requires error text")
        return self


class WorkflowTrajectoryRecord(ContractModel):
    scope: ExecutionScope
    record_index: int | None = Field(default=None, ge=1)
    status: TraceStatus
    completed: bool
    user_input: NonEmptyStr
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    context: str = ""
    global_intent: JsonValue | None = None
    unified_plan: JsonValue | None = None
    qos_feedback: JsonValue | None = None
    mobility_feedback: JsonValue | None = None
    diagnosis: JsonValue | None = None
    round_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    round_traces: list[dict[str, JsonValue]] = Field(default_factory=list)
    agent_trajectories: dict[NonEmptyStr, list[ProjectedTraceRecord]] = Field(default_factory=dict)
    tool_calls: list[dict[str, JsonValue]] = Field(default_factory=list)
    tool_results: list[dict[str, JsonValue]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_agent_trajectory_scopes(self) -> "WorkflowTrajectoryRecord":
        if self.scope.snapshot_id is None:
            raise ValueError("workflow trajectory requires scope.snapshot_id")
        for agent_name, traces in self.agent_trajectories.items():
            for trace in traces:
                if trace.scope.session_id != self.scope.session_id:
                    raise ValueError(
                        f"agent trajectory {agent_name}/{trace.trace_id} belongs to another session"
                    )
                if trace.scope.snapshot_id != self.scope.snapshot_id:
                    raise ValueError(
                        f"agent trajectory {agent_name}/{trace.trace_id} belongs to another snapshot"
                    )
        return self


__all__ = [
    "ProjectedTraceRecord",
    "Qwen3SftRecord",
    "SftExampleV2",
    "WorkflowTrajectoryRecord",
]
