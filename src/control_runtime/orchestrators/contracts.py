"""Artifacts exchanged between orchestration rounds and executors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, JsonValue, model_validator

from shared.contracts import ContractModel, ExecutionScope, NonEmptyStr

from ..agents.main.contracts import ControlDomain, GlobalControlIntent, OpenQuestion
from ..diagnostics.mediation.contracts import (
    DomainStatus,
    DomainVerdict,
    MediatorDecision,
    UnifiedConstraintSet,
)


class DomainProposal(ContractModel):
    domain: ControlDomain
    status: DomainStatus = DomainStatus.READY
    rationale: str = ""
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    policy_drafts: list[dict[str, JsonValue]] = Field(default_factory=list)


class AgentContribution(ContractModel):
    agent: NonEmptyStr
    summary: str = ""
    evidence: list[NonEmptyStr] = Field(default_factory=list)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class HandoffRecord(ContractModel):
    source_agent: NonEmptyStr
    target_agent: NonEmptyStr
    artifact_type: NonEmptyStr
    summary: str = ""
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class UnifiedControlPlan(ContractModel):
    scope: ExecutionScope
    global_intent: Optional[GlobalControlIntent]
    qos_proposal: Optional[DomainProposal]
    mobility_proposal: Optional[DomainProposal]
    domain_verdicts: List[DomainVerdict]
    mediator_decision: Optional[MediatorDecision]
    unified_constraints: UnifiedConstraintSet
    execution_order: List[ControlDomain]
    approved_policies: list[dict[str, JsonValue]]
    blocked_domains: List[ControlDomain]
    objective_breakdown: dict[str, JsonValue]
    control_churn_count: int = Field(ge=0)
    agent_contributions: List[AgentContribution]
    handoff_records: List[HandoffRecord]
    open_questions: List[OpenQuestion]

    @model_validator(mode="after")
    def validate_domain_sets(self) -> "UnifiedControlPlan":
        if len(self.execution_order) != len(set(self.execution_order)):
            raise ValueError("execution_order must not contain duplicate domains")
        if len(self.blocked_domains) != len(set(self.blocked_domains)):
            raise ValueError("blocked_domains must not contain duplicate domains")
        return self


class CoordinationIssue(ContractModel):
    source_agent: NonEmptyStr
    issue_type: NonEmptyStr
    domain: str = ""
    binding_keys: list[NonEmptyStr] = Field(default_factory=list)
    policy_objects: list[NonEmptyStr] = Field(default_factory=list)
    missing_evidence: list[NonEmptyStr] = Field(default_factory=list)
    rationale: str = ""


class DomainNegotiationRequest(ContractModel):
    round_index: int = Field(default=1, ge=1)
    source_agent: NonEmptyStr = "intent_encoding"
    requires_domain_review: bool = False
    issues: List[CoordinationIssue] = Field(default_factory=list)
    recommended_consumers: list[NonEmptyStr] = Field(default_factory=list)
    summary: str = ""


class PlanningBlockerReport(ContractModel):
    round_index: int = Field(default=1, ge=1)
    source_agent: NonEmptyStr = "optimization_strategy"
    planning_status: NonEmptyStr = "needs_upstream_reground"
    missing_evidence: list[NonEmptyStr] = Field(default_factory=list)
    blocked_targets: list[NonEmptyStr] = Field(default_factory=list)
    upstream_requests: list[NonEmptyStr] = Field(default_factory=list)
    planner_conflicts: list[NonEmptyStr] = Field(default_factory=list)
    recommended_consumers: list[NonEmptyStr] = Field(default_factory=list)
    summary: str = ""


class ExecutionReentryRequest(ContractModel):
    round_index: int = Field(default=1, ge=1)
    source_agent: NonEmptyStr = "policy_dispatch"
    recommended_consumers: list[NonEmptyStr] = Field(default_factory=list)
    target_bindings_at_risk: list[NonEmptyStr] = Field(default_factory=list)
    policy_objects_at_risk: list[NonEmptyStr] = Field(default_factory=list)
    reason_by_domain: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    failure_scope: NonEmptyStr = "none"
    failures: list[dict[str, JsonValue]] = Field(default_factory=list)
    summary: str = ""


class ControlRoundTrace(ContractModel):
    """JSON-safe trace for one control round."""

    round_index: int = Field(ge=1)
    global_intent: dict[str, JsonValue] = Field(default_factory=dict)
    grounding_decision: dict[str, JsonValue] = Field(default_factory=dict)
    policy_plan: dict[str, JsonValue] = Field(default_factory=dict)
    domain_verdicts: list[dict[str, JsonValue]] = Field(default_factory=list)
    pda_feedback: dict[str, JsonValue] = Field(default_factory=dict)
    qos_feedback: dict[str, JsonValue] = Field(default_factory=dict)
    mobility_feedback: dict[str, JsonValue] = Field(default_factory=dict)
    diagnosis: dict[str, JsonValue] = Field(default_factory=dict)
    mediator_decision: dict[str, JsonValue] = Field(default_factory=dict)
    negotiation_request: dict[str, JsonValue] = Field(default_factory=dict)
    planning_blocker: dict[str, JsonValue] = Field(default_factory=dict)
    execution_reentry: dict[str, JsonValue] = Field(default_factory=dict)
    single_agent_detail: dict[str, JsonValue] = Field(default_factory=dict)
    agent_elapsed_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ControlRoundTrace":
        return cls.model_validate(payload)

    def to_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def with_agent_elapsed_ms(self, agent_elapsed_ms: Dict[str, float]) -> "ControlRoundTrace":
        return self.__class__.model_validate(
            {**self.model_dump(mode="json"), "agent_elapsed_ms": agent_elapsed_ms}
        )

    def with_single_agent_detail(self, single_agent_detail: Dict[str, Any]) -> "ControlRoundTrace":
        return self.__class__.model_validate(
            {**self.model_dump(mode="json"), "single_agent_detail": single_agent_detail}
        )

    def with_pda_feedback(self, pda_feedback: Dict[str, Any]) -> "ControlRoundTrace":
        return self.__class__.model_validate(
            {**self.model_dump(mode="json"), "pda_feedback": pda_feedback}
        )


class ControlRoundResult(ContractModel):
    scope: ExecutionScope
    completed: bool
    global_intent: dict[str, JsonValue]
    unified_plan: dict[str, JsonValue]
    qos_feedback: dict[str, JsonValue]
    mobility_feedback: dict[str, JsonValue]
    diagnosis: dict[str, JsonValue]
    negotiation_request: dict[str, JsonValue]
    planning_blocker: dict[str, JsonValue]
    execution_reentry: dict[str, JsonValue]
    round_count: int = Field(ge=1)
    retry_count: int = Field(ge=0)
    round_traces: list[dict[str, JsonValue]]
    agent_elapsed_ms: dict[str, float]
    single_agent_details: list[dict[str, JsonValue]]

    @model_validator(mode="after")
    def validate_result_scope(self) -> "ControlRoundResult":
        if self.scope.snapshot_id is None:
            raise ValueError("control round result requires scope.snapshot_id")
        return self

__all__ = [
    "ControlRoundResult",
    "ControlRoundTrace",
    "CoordinationIssue",
    "DomainNegotiationRequest",
    "DomainProposal",
    "ExecutionReentryRequest",
    "PlanningBlockerReport",
    "UnifiedControlPlan",
]
