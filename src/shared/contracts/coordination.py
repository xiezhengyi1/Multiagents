from __future__ import annotations

import math
from typing import ClassVar, Generic, Literal, TypeVar

from pydantic import Field, model_validator

from .common import ContractModel, EventScope, NonEmptyStr

MessageType = Literal[
    "proposal",
    "handoff",
    "validation_result",
    "repair_request",
    "repair_response",
    "observation",
    "execution_feedback",
    "diagnosis",
    "terminal",
]

ActorId = NonEmptyStr


class CoordinationPayload(ContractModel):
    """Base class for payloads accepted by :class:`CoordinationEvent`."""

    allowed_message_types: ClassVar[frozenset[str]] = frozenset()


class ProposalPayload(CoordinationPayload):
    """Base for a model-authored semantic proposal or repaired proposal."""

    allowed_message_types = frozenset({"proposal", "repair_response"})


class HandoffPayload(CoordinationPayload):
    """Base for a validated or deterministically compiled downstream artifact."""

    allowed_message_types = frozenset({"handoff"})


class ObservationPayload(CoordinationPayload):
    """Base for an immutable read from a simulator or external control plane."""

    allowed_message_types = frozenset({"observation"})


class ExecutionFeedbackPayload(CoordinationPayload):
    """Base for post-action simulator or control-plane feedback."""

    allowed_message_types = frozenset({"execution_feedback"})


class DiagnosisPayload(CoordinationPayload):
    """Base for deterministic pre-execution and post-execution diagnoses."""

    allowed_message_types = frozenset({"diagnosis"})


class ValidationIssue(ContractModel):
    code: NonEmptyStr
    message: NonEmptyStr
    path: NonEmptyStr | None = None


class ValidationResultPayload(CoordinationPayload):
    allowed_message_types = frozenset({"validation_result"})

    notes: list[NonEmptyStr] = Field(default_factory=list)


class RepairRequestPayload(CoordinationPayload):
    allowed_message_types = frozenset({"repair_request"})

    issues: list[ValidationIssue] = Field(min_length=1)
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    failed_trace_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_attempt_budget(self) -> RepairRequestPayload:
        if self.attempt > self.max_attempts:
            raise ValueError("repair attempt cannot exceed max_attempts")
        return self


class TerminalPayload(CoordinationPayload):
    allowed_message_types = frozenset({"terminal"})

    outcome: Literal["success", "failure"]
    resolution: Literal["executed", "infeasible", "failed"]
    reason_code: NonEmptyStr
    reason: NonEmptyStr
    reward_total: float
    reward_components: dict[NonEmptyStr, float]

    @model_validator(mode="after")
    def validate_reward(self) -> TerminalPayload:
        if self.resolution in {"executed", "infeasible"} and self.outcome != "success":
            raise ValueError("executed and infeasible terminals must be successful")
        if self.resolution == "failed" and self.outcome != "failure":
            raise ValueError("failed terminals must be failures")
        values = [self.reward_total, *self.reward_components.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("terminal rewards must be finite")
        if not math.isclose(
            self.reward_total,
            sum(self.reward_components.values()),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("reward_total must equal the sum of reward_components")
        return self


PayloadT = TypeVar("PayloadT", bound=CoordinationPayload)


class CoordinationDecision(ContractModel, Generic[PayloadT]):
    """The target, message type, and payload selected by one actor."""

    target_agent: ActorId
    message_type: MessageType
    payload: PayloadT

    @model_validator(mode="after")
    def validate_payload_type(self) -> CoordinationDecision[PayloadT]:
        if self.message_type not in self.payload.allowed_message_types:
            allowed = ", ".join(sorted(self.payload.allowed_message_types)) or "none"
            raise ValueError(
                f"payload {type(self.payload).__name__} only supports message types: {allowed}"
            )
        return self


class CoordinationEvent(ContractModel, Generic[PayloadT]):
    """The sole communication record shared by model and deterministic actors."""

    event_id: NonEmptyStr
    event_index: int = Field(ge=0)
    scope: EventScope
    source_agent: ActorId
    target_agent: ActorId
    message_type: MessageType
    in_reply_to: NonEmptyStr | None = None
    input_artifact_ids: list[NonEmptyStr] = Field(default_factory=list)
    invalidated_artifact_ids: list[NonEmptyStr] = Field(default_factory=list)
    payload: PayloadT

    @model_validator(mode="after")
    def validate_event(self) -> CoordinationEvent[PayloadT]:
        if self.message_type not in self.payload.allowed_message_types:
            allowed = ", ".join(sorted(self.payload.allowed_message_types)) or "none"
            raise ValueError(
                f"payload {type(self.payload).__name__} only supports message types: {allowed}"
            )
        for name, values in (
            ("input_artifact_ids", self.input_artifact_ids),
            ("invalidated_artifact_ids", self.invalidated_artifact_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique ids")
        if set(self.input_artifact_ids) & set(self.invalidated_artifact_ids):
            raise ValueError(
                "an artifact cannot be both an input and invalidated by one event"
            )
        referenced = {
            *self.input_artifact_ids,
            *self.invalidated_artifact_ids,
            *([self.in_reply_to] if self.in_reply_to is not None else []),
        }
        if self.event_id in referenced:
            raise ValueError("event cannot reference itself")
        if self.message_type == "repair_response" and self.in_reply_to is None:
            raise ValueError("repair_response requires in_reply_to")
        if self.message_type == "repair_request" and self.in_reply_to is None:
            raise ValueError("repair_request requires in_reply_to")
        if self.message_type == "validation_result" and not self.input_artifact_ids:
            raise ValueError("validation_result requires an input artifact")
        return self


__all__ = [
    "ActorId",
    "CoordinationDecision",
    "CoordinationEvent",
    "CoordinationPayload",
    "DiagnosisPayload",
    "ExecutionFeedbackPayload",
    "HandoffPayload",
    "MessageType",
    "ObservationPayload",
    "PayloadT",
    "ProposalPayload",
    "RepairRequestPayload",
    "TerminalPayload",
    "ValidationIssue",
    "ValidationResultPayload",
]
