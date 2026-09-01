from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_serializer, model_validator

from shared.contracts import ContractModel, JsonValue

from ..grounding.contracts import GroundingDecision
from ..main.contracts import GlobalControlIntent, OpenQuestion


def json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SharedControlContext(ContractModel):
    main_intent: GlobalControlIntent = Field(description="Canonical Main decision")


class QosRetryConstraint(ContractModel):
    flow_id: str = Field(min_length=1)
    observed_latency_ms: float = Field(gt=0.0)
    allowed_latency_ms: float = Field(gt=0.0)
    minimum_target_latency_ms: int = Field(ge=1)
    tolerance_ratio: float = Field(ge=0.0)
    policy_id: str = Field(default="")


class PlanningContext(ContractModel):
    memory_context: str = Field(default="")
    shared_context: SharedControlContext
    feedback_context: str = Field(default="")
    handoff_history: list[dict[str, JsonValue]] = Field(default_factory=list)
    revision_requests: list[dict[str, JsonValue]] = Field(default_factory=list)
    unified_constraints: dict[str, JsonValue] = Field(default_factory=dict)
    qos_retry_constraints: list[QosRetryConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> "PlanningContext":
        if self.shared_context.main_intent.scope.snapshot_id is None:
            raise ValueError("planning context requires main_intent.scope.snapshot_id")
        return self


class PlanningRequest(ContractModel):
    grounding_decision: GroundingDecision
    context: PlanningContext


class PolicyDraft(BaseModel):
    recommended_actions: List[str] = Field(default_factory=list)
    supi: str = Field(default="")
    app_id: str = Field(default="")
    flow_id: str | None = Field(default=None)
    target_type: str = Field(default="flow")
    policy_id: str = Field(default="")
    policy_type: str = Field(...)
    resource_keys: List[str] = Field(default_factory=list)
    policy_details: Dict[str, Any] = Field(default_factory=dict)

    @field_serializer("policy_details", when_used="always")
    def _serialize_policy_details(self, value: Dict[str, Any]) -> Any:
        return json_friendly(value)


class PlanningRationale(BaseModel):
    selected_strategy_profile: str = Field(default="")
    decisive_evidence: List[str] = Field(default_factory=list)
    active_constraints: List[str] = Field(default_factory=list)
    explanation: str = Field(default="")
    rejected_alternatives: List[str] = Field(default_factory=list)
    main_constraints: List[str] = Field(default_factory=list)
    iea_grounding_basis: List[str] = Field(default_factory=list)
    osa_decision_basis: List[str] = Field(default_factory=list)
    unresolved_gaps: List[str] = Field(default_factory=list)


class PolicyPlanDraft(BaseModel):
    supi: str = Field(default="")
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    planning_status: str = Field(default="executable_plan")
    optimizer_result: Dict[str, Any] = Field(default_factory=dict)
    planning_rationale: PlanningRationale = Field(default_factory=PlanningRationale)
    all_policies: List[PolicyDraft] = Field(default_factory=list)
    partial_policies: List[PolicyDraft] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    blocked_targets: List[str] = Field(default_factory=list)
    upstream_requests: List[str] = Field(default_factory=list)
    planner_conflicts: List[str] = Field(default_factory=list)
    open_questions: List[OpenQuestion] = Field(default_factory=list)


__all__ = [
    "PlanningContext",
    "PlanningRationale",
    "PlanningRequest",
    "PolicyDraft",
    "PolicyPlanDraft",
    "QosRetryConstraint",
    "SharedControlContext",
]
