from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator

from shared.contracts import ContractModel, ExecutionScope, JsonValue, NonEmptyStr


class DiagnosisExecutionFeedback(ContractModel):
    pda: dict[str, JsonValue]
    qos: dict[str, JsonValue]
    mobility: dict[str, JsonValue]


class DiagnosisUpstreamContext(ContractModel):
    global_intent: dict[str, JsonValue]
    grounding_decision: dict[str, JsonValue]
    conflict_result: dict[str, JsonValue]


class AssuranceDiagnosisRequest(ContractModel):
    scope: ExecutionScope
    execution_feedback: DiagnosisExecutionFeedback
    dispatch_receipts: list[dict[str, JsonValue]]
    assurance_verdicts: list[dict[str, JsonValue]]
    telemetry_snapshot: dict[str, JsonValue]
    upstream_context: DiagnosisUpstreamContext

    @model_validator(mode="after")
    def validate_scope(self) -> "AssuranceDiagnosisRequest":
        if self.scope.snapshot_id is None:
            raise ValueError("assurance diagnosis requires scope.snapshot_id")
        return self


DiagnosisStatus = Literal["diagnosed", "insufficient_evidence"]


class AssuranceDiagnosisResult(ContractModel):
    status: DiagnosisStatus
    root_cause_category: NonEmptyStr
    root_cause: NonEmptyStr
    affected_policy_ids: list[NonEmptyStr]
    affected_flow_ids: list[NonEmptyStr]
    recommended_actions: list[NonEmptyStr]
    reason_summary: NonEmptyStr

    @field_validator("affected_policy_ids", "affected_flow_ids", "recommended_actions")
    @classmethod
    def validate_unique_lists(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("diagnosis list values must be unique")
        return values


__all__ = [
    "AssuranceDiagnosisRequest",
    "AssuranceDiagnosisResult",
    "DiagnosisExecutionFeedback",
    "DiagnosisStatus",
    "DiagnosisUpstreamContext",
]
