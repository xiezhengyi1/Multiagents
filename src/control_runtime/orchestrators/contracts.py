"""Artifacts exchanged between orchestration rounds and executors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..agents.main.contracts import ControlDomain, GlobalControlIntent, OpenQuestion
from ..diagnostics.mediation.contracts import (
    DomainStatus,
    DomainVerdict,
    MediatorDecision,
    UnifiedConstraintSet,
)


def _mapping(value: Any) -> Dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _mapping_list(value: Any) -> List[Dict[str, Any]]:
    return [deepcopy(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


class DomainProposal(BaseModel):
    domain: ControlDomain
    status: DomainStatus = DomainStatus.READY
    rationale: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    policy_drafts: List[Dict[str, Any]] = Field(default_factory=list)


class AgentContribution(BaseModel):
    agent: str = ""
    summary: str = ""
    evidence: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)


class HandoffRecord(BaseModel):
    source_agent: str = ""
    target_agent: str = ""
    artifact_type: str = ""
    summary: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


class UnifiedControlPlan(BaseModel):
    session_id: str
    snapshot_id: str
    supi: str
    global_intent: Optional[GlobalControlIntent] = None
    qos_proposal: Optional[DomainProposal] = None
    mobility_proposal: Optional[DomainProposal] = None
    domain_verdicts: List[DomainVerdict] = Field(default_factory=list)
    mediator_decision: Optional[MediatorDecision] = None
    unified_constraints: UnifiedConstraintSet = Field(default_factory=UnifiedConstraintSet)
    execution_order: List[ControlDomain] = Field(default_factory=list)
    approved_policies: List[Dict[str, Any]] = Field(default_factory=list)
    blocked_domains: List[ControlDomain] = Field(default_factory=list)
    objective_breakdown: Dict[str, Any] = Field(default_factory=dict)
    control_churn_count: int = 0
    agent_contributions: List[AgentContribution] = Field(default_factory=list)
    handoff_records: List[HandoffRecord] = Field(default_factory=list)
    open_questions: List[OpenQuestion] = Field(default_factory=list)


class CoordinationIssue(BaseModel):
    source_agent: str = ""
    issue_type: str = ""
    domain: str = ""
    binding_keys: List[str] = Field(default_factory=list)
    policy_objects: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    rationale: str = ""


class DomainNegotiationRequest(BaseModel):
    round_index: int = 1
    source_agent: str = "intent_encoding"
    requires_domain_review: bool = False
    issues: List[CoordinationIssue] = Field(default_factory=list)
    recommended_consumers: List[str] = Field(default_factory=list)
    summary: str = ""


class PlanningBlockerReport(BaseModel):
    round_index: int = 1
    source_agent: str = "optimization_strategy"
    planning_status: str = "needs_upstream_reground"
    missing_evidence: List[str] = Field(default_factory=list)
    blocked_targets: List[str] = Field(default_factory=list)
    upstream_requests: List[str] = Field(default_factory=list)
    planner_conflicts: List[str] = Field(default_factory=list)
    recommended_consumers: List[str] = Field(default_factory=list)
    summary: str = ""


class ExecutionReentryRequest(BaseModel):
    round_index: int = 1
    source_agent: str = "policy_dispatch"
    recommended_consumers: List[str] = Field(default_factory=list)
    target_bindings_at_risk: List[str] = Field(default_factory=list)
    policy_objects_at_risk: List[str] = Field(default_factory=list)
    reason_by_domain: Dict[str, str] = Field(default_factory=dict)
    failure_scope: str = "none"
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class ControlRoundTrace:
    """JSON-safe trace for one control round."""

    round_index: int
    global_intent: Dict[str, Any] = field(default_factory=dict)
    grounding_decision: Dict[str, Any] = field(default_factory=dict)
    policy_plan: Dict[str, Any] = field(default_factory=dict)
    domain_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    pda_feedback: Dict[str, Any] = field(default_factory=dict)
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    mediator_decision: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)
    single_agent_detail: Dict[str, Any] = field(default_factory=dict)
    agent_elapsed_ms: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_intent", _mapping(self.global_intent))
        object.__setattr__(self, "grounding_decision", _mapping(self.grounding_decision))
        object.__setattr__(self, "policy_plan", _mapping(self.policy_plan))
        object.__setattr__(self, "domain_verdicts", _mapping_list(self.domain_verdicts))
        object.__setattr__(self, "pda_feedback", _mapping(self.pda_feedback))
        object.__setattr__(self, "qos_feedback", _mapping(self.qos_feedback))
        object.__setattr__(self, "mobility_feedback", _mapping(self.mobility_feedback))
        object.__setattr__(self, "diagnosis", _mapping(self.diagnosis))
        object.__setattr__(self, "mediator_decision", _mapping(self.mediator_decision))
        object.__setattr__(self, "negotiation_request", _mapping(self.negotiation_request))
        object.__setattr__(self, "planning_blocker", _mapping(self.planning_blocker))
        object.__setattr__(self, "execution_reentry", _mapping(self.execution_reentry))
        object.__setattr__(self, "single_agent_detail", _mapping(self.single_agent_detail))
        object.__setattr__(self, "agent_elapsed_ms", _mapping(self.agent_elapsed_ms))

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ControlRoundTrace":
        return cls(
            round_index=int(payload.get("round_index") or 0),
            global_intent=_mapping(payload.get("global_intent")),
            grounding_decision=_mapping(payload.get("grounding_decision")),
            policy_plan=_mapping(payload.get("policy_plan")),
            domain_verdicts=_mapping_list(payload.get("domain_verdicts")),
            pda_feedback=_mapping(payload.get("pda_feedback")),
            qos_feedback=_mapping(payload.get("qos_feedback")),
            mobility_feedback=_mapping(payload.get("mobility_feedback")),
            diagnosis=_mapping(payload.get("diagnosis")),
            mediator_decision=_mapping(payload.get("mediator_decision")),
            negotiation_request=_mapping(payload.get("negotiation_request")),
            planning_blocker=_mapping(payload.get("planning_blocker")),
            execution_reentry=_mapping(payload.get("execution_reentry")),
            single_agent_detail=_mapping(payload.get("single_agent_detail")),
            agent_elapsed_ms=_mapping(payload.get("agent_elapsed_ms")),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "global_intent": _mapping(self.global_intent),
            "grounding_decision": _mapping(self.grounding_decision),
            "policy_plan": _mapping(self.policy_plan),
            "domain_verdicts": _mapping_list(self.domain_verdicts),
            "pda_feedback": _mapping(self.pda_feedback),
            "qos_feedback": _mapping(self.qos_feedback),
            "mobility_feedback": _mapping(self.mobility_feedback),
            "diagnosis": _mapping(self.diagnosis),
            "mediator_decision": _mapping(self.mediator_decision),
            "negotiation_request": _mapping(self.negotiation_request),
            "planning_blocker": _mapping(self.planning_blocker),
            "execution_reentry": _mapping(self.execution_reentry),
            "single_agent_detail": _mapping(self.single_agent_detail),
            "agent_elapsed_ms": _mapping(self.agent_elapsed_ms),
        }

    def with_agent_elapsed_ms(self, agent_elapsed_ms: Dict[str, float]) -> "ControlRoundTrace":
        return replace(self, agent_elapsed_ms=_mapping(agent_elapsed_ms))

    def with_single_agent_detail(self, single_agent_detail: Dict[str, Any]) -> "ControlRoundTrace":
        return replace(self, single_agent_detail=_mapping(single_agent_detail))


@dataclass
class ControlRoundResult:
    session_id: str
    snapshot_id: str
    completed: bool
    global_intent: Dict[str, Any]
    unified_plan: Dict[str, Any]
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)
    round_count: int = 1
    retry_count: int = 0
    round_traces: List[Dict[str, Any]] = field(default_factory=list)
    agent_elapsed_ms: Dict[str, float] = field(default_factory=dict)
    main_schema_projection_count: int = 0
    main_schema_projected_field_count: int = 0
    main_schema_projected_fields: List[str] = field(default_factory=list)
    single_agent_details: List[Dict[str, Any]] = field(default_factory=list)


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
