from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import TerminalPayload
from shared.contracts.model_trace import AgentTrace, ModelMessage, ModelToolSpec


class TrainingInputRecord(ContractModel):
    record_id: NonEmptyStr
    episode_id: NonEmptyStr
    user_input: NonEmptyStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SftExample(ContractModel):
    example_id: NonEmptyStr
    episode_id: NonEmptyStr
    actor: NonEmptyStr
    input_messages: list[ModelMessage] = Field(min_length=1)
    output_message: ModelMessage
    tools: list[ModelToolSpec]
    output_event_id: NonEmptyStr
    validation_event_id: NonEmptyStr
    disposition: Literal["accepted", "accepted_upstream_repair"]


class EpisodeMetrics(ContractModel):
    round_count: int = Field(ge=1)
    repair_count: int = Field(ge=0)
    execution_attempt_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    model_error_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    objective_violation_count: int = Field(ge=0)
    execution_failure_count: int = Field(ge=0)
    first_attempt_success: bool
    recovered_after_repair: bool


class RlTrajectory(ContractModel):
    episode_id: NonEmptyStr
    events: list[dict[str, JsonValue]] = Field(min_length=1)
    traces: list[AgentTrace]
    terminal: TerminalPayload
    metrics: EpisodeMetrics


class CollectionResult(ContractModel):
    episode_directories: list[NonEmptyStr]


class EvaluationSummary(ContractModel):
    episode_count: int = Field(ge=1)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    mean_reward: float


__all__ = [
    "CollectionResult",
    "EpisodeMetrics",
    "EvaluationSummary",
    "RlTrajectory",
    "SftExample",
    "TrainingInputRecord",
]
