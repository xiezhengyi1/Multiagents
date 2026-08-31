from __future__ import annotations

import json
from typing import Any

from pydantic import Field, JsonValue, model_validator

from shared.contracts import (
    CONTRACT_VERSION,
    AgentMessage,
    ContractModel,
    ContractVersion,
    ExecutionScope,
    ToolCatalog,
    TraceMetadata,
    TraceTarget,
    validate_message_sequence,
)


class SftExampleV2(ContractModel):
    """Model-independent SFT example projected from one validated observed trace."""

    schema_version: ContractVersion = CONTRACT_VERSION
    sample_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    scope: ExecutionScope
    messages: list[AgentMessage] = Field(min_length=2)
    tool_catalog: ToolCatalog
    response_schema_name: str = Field(min_length=1)
    response_schema: dict[str, JsonValue]
    observed_target: TraceTarget
    quality_targets: list[TraceTarget] = Field(min_length=1)
    metadata: TraceMetadata

    @model_validator(mode="after")
    def validate_example(self) -> "SftExampleV2":
        validate_message_sequence(self.messages, require_final_assistant=True)
        if self.observed_target.kind != "observed_model_output" or not self.observed_target.sft_eligible:
            raise ValueError("SFT example requires one eligible observed_model_output target")
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
    sample_id: str = Field(min_length=1)
    messages: list[dict[str, JsonValue]] = Field(min_length=2)
    tools: list[dict[str, JsonValue]]
    metadata: dict[str, JsonValue]


class ProjectedTraceRecord(ContractModel):
    """Read-only evaluation view; never used as the SFT source contract."""

    trace_id: str
    timestamp: str
    agent_name: str
    session_id: str
    snapshot_id: str
    thread_id: str
    model_name: str
    messages: list[AgentMessage]
    tool_calls: list[dict[str, JsonValue]]
    tool_results: list[dict[str, JsonValue]]
    structured_response: JsonValue | None
    status: str
    error: str | None = None
    scenario_id: str | None = None
    scenario_tags: list[str] = Field(default_factory=list)
    path_label: str
    quality_targets: list[TraceTarget] = Field(default_factory=list)


class WorkflowTrajectoryRecord(ContractModel):
    workflow_id: str
    record_index: int | None = None
    scenario_id: str | None = None
    scenario_tags: list[str] = Field(default_factory=list)
    session_id: str
    snapshot_id: str
    status: str
    completed: bool
    user_input: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: str = ""
    global_intent: Any = None
    unified_plan: Any = None
    qos_feedback: Any = None
    mobility_feedback: Any = None
    diagnosis: Any = None
    round_count: int = 0
    retry_count: int = 0
    round_traces: list[dict[str, Any]] = Field(default_factory=list)
    agent_trajectories: dict[str, list[ProjectedTraceRecord]] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "ProjectedTraceRecord",
    "Qwen3SftRecord",
    "SftExampleV2",
    "WorkflowTrajectoryRecord",
]
