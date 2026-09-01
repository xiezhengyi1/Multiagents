from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from shared.contracts import ContractModel, ExecutionScope, JsonValue, MemoryContextView, NonEmptyStr

from ..grounding.contracts import GroundingDecision
from ..main.contracts import GlobalControlIntent, OpenQuestion
from ...diagnostics.mediation.contracts import DomainVerdict, RevisionRequest, UnifiedConstraintSet


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
    flow_id: NonEmptyStr
    observed_latency_ms: float = Field(gt=0.0)
    allowed_latency_ms: float = Field(gt=0.0)
    minimum_target_latency_ms: int = Field(ge=1)
    tolerance_ratio: float = Field(ge=0.0)
    policy_id: NonEmptyStr | None


class PlanningContext(ContractModel):
    memory_context: MemoryContextView
    shared_context: SharedControlContext
    feedback_context: str
    handoff_history: list[dict[str, JsonValue]]
    revision_requests: list[RevisionRequest]
    unified_constraints: UnifiedConstraintSet
    qos_retry_constraints: list[QosRetryConstraint]

    @model_validator(mode="after")
    def validate_scope(self) -> "PlanningContext":
        if self.shared_context.main_intent.scope.snapshot_id is None:
            raise ValueError("planning context requires main_intent.scope.snapshot_id")
        return self


class PlanningRequest(ContractModel):
    grounding_decision: GroundingDecision
    context: PlanningContext


PolicyType = Literal[
    "SmPolicyDecision",
    "UrspRuleRequest",
    "PcfAmPolicyControlPolicyAssociation",
]
PolicyTargetType = Literal["flow", "app", "ue"]
PlanningStatus = Literal["executable_plan", "partial_plan", "needs_upstream_reground"]


class PolicyDraft(ContractModel):
    supi: NonEmptyStr
    app_id: NonEmptyStr | None
    flow_id: NonEmptyStr | None
    target_type: PolicyTargetType
    policy_id: NonEmptyStr
    policy_type: PolicyType
    resource_keys: list[NonEmptyStr]
    policy_details: dict[str, JsonValue]

    @field_serializer("policy_details", when_used="always")
    def _serialize_policy_details(self, value: dict[str, JsonValue]) -> Any:
        return json_friendly(value)

    @model_validator(mode="after")
    def validate_binding(self) -> "PolicyDraft":
        if self.policy_type == "PcfAmPolicyControlPolicyAssociation":
            if self.target_type != "ue" or self.app_id is not None or self.flow_id is not None:
                raise ValueError("AM policy requires target_type=ue and no app_id or flow_id")
            return self
        if self.target_type == "ue":
            raise ValueError("SM and URSP policies cannot target a UE")
        if self.app_id is None:
            raise ValueError("SM and URSP policies require app_id")
        if self.target_type == "flow" and self.flow_id is None:
            raise ValueError("flow-targeted policy requires flow_id")
        if self.policy_type == "SmPolicyDecision" and self.target_type != "flow":
            raise ValueError("SmPolicyDecision requires target_type=flow")
        return self

    @field_validator("resource_keys")
    @classmethod
    def validate_resource_keys(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("policy resource_keys must be unique")
        return values


class PlanningRationale(ContractModel):
    selected_strategy_profile: NonEmptyStr
    decisive_evidence: list[NonEmptyStr]
    active_constraints: list[NonEmptyStr]
    explanation: NonEmptyStr
    rejected_alternatives: list[NonEmptyStr]
    main_constraints: list[NonEmptyStr]
    iea_grounding_basis: list[NonEmptyStr]
    osa_decision_basis: list[NonEmptyStr]
    unresolved_gaps: list[NonEmptyStr]


class OptimizerPlanEvidence(ContractModel):
    objective_breakdown: dict[str, JsonValue]
    cross_domain_verdicts: list[DomainVerdict]


class PolicyPlanDraft(ContractModel):
    scope: ExecutionScope
    planning_status: PlanningStatus
    optimizer_result: OptimizerPlanEvidence
    planning_rationale: PlanningRationale
    all_policies: list[PolicyDraft]
    partial_policies: list[PolicyDraft]
    missing_evidence: list[NonEmptyStr]
    blocked_targets: list[NonEmptyStr]
    upstream_requests: list[NonEmptyStr]
    planner_conflicts: list[NonEmptyStr]
    open_questions: list[OpenQuestion]

    @model_validator(mode="after")
    def validate_plan(self) -> "PolicyPlanDraft":
        if self.scope.snapshot_id is None or self.scope.supi is None:
            raise ValueError("policy plan requires scope.snapshot_id and scope.supi")
        policy_ids = [item.policy_id for item in [*self.all_policies, *self.partial_policies]]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy plan policy_id values must be unique")
        if self.planning_status == "executable_plan":
            if not self.all_policies or self.partial_policies:
                raise ValueError("executable_plan requires executable policies and no partial policies")
        elif self.planning_status == "partial_plan":
            if not self.all_policies and not self.partial_policies:
                raise ValueError("partial_plan requires executable or partial policies")
            if not self.missing_evidence and not self.blocked_targets and not self.planner_conflicts:
                raise ValueError("partial_plan requires explicit unresolved gaps")
        else:
            if self.all_policies or self.partial_policies:
                raise ValueError("needs_upstream_reground cannot contain policies")
            if not self.upstream_requests and not self.missing_evidence and not self.blocked_targets:
                raise ValueError("needs_upstream_reground requires explicit upstream work")
        return self


__all__ = [
    "PlanningContext",
    "OptimizerPlanEvidence",
    "PlanningRationale",
    "PlanningRequest",
    "PolicyDraft",
    "PolicyPlanDraft",
    "PolicyTargetType",
    "PolicyType",
    "PlanningStatus",
    "QosRetryConstraint",
    "SharedControlContext",
]
