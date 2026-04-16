from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from model.PcfAmPolicyControl import (
    AccessType,
    Guami,
    MappingOfSnssai,
    PcfAmPolicyControlPolicyAssociation,
    PcfAmPolicyControlPolicyAssociationRequest,
    PcfAmPolicyControlRequestTrigger,
    PlmnIdNid,
    PresenceInfo,
    RatType,
    ServiceAreaRestriction,
    SmfSelectionData,
    Snssai,
    UserLocation,
    WirelineServiceAreaRestriction,
)


class ControlDomain(str, Enum):
    QOS = "qos"
    MOBILITY = "mobility"


class OptimizationTemplate(str, Enum):
    QOS_FIRST = "qos_first"
    MOBILITY_FIRST = "mobility_first"
    JOINT_BALANCED = "joint_balanced"
    STABILITY_FIRST = "stability_first"
    CONGESTION_RELIEF = "congestion_relief"


class DomainStatus(str, Enum):
    READY = "ready"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    INCOMPLETE_CONTEXT = "incomplete_context"
    SKIPPED = "skipped"
    FAILED = "failed"


class ObjectiveProfile(BaseModel):
    profile_name: str = Field(default="balanced")
    sla_violation_cost: float = Field(default=1.0, ge=0.0)
    mobility_risk_cost: float = Field(default=0.8, ge=0.0)
    control_churn_cost: float = Field(default=0.5, ge=0.0)
    resource_pressure_cost: float = Field(default=0.7, ge=0.0)
    fairness_cost: float = Field(default=0.3, ge=0.0)

    def to_legacy_qos_weights(self) -> Dict[str, float]:
        """
        Map semantic costs to the current QoS optimizer weights.

        Migration path:
        - resource_pressure_cost -> w1(load)
        - control_churn_cost -> w2(sig)
        - sla_violation_cost (+ fairness tail) -> w3(exp + qos_core)
        - fairness_cost (+ SLA auxiliary tail) -> w4(qos_aux)
        - mobility_risk_cost stays in the mobility planner / cross-domain checker
        """
        w1 = max(1.0, 100.0 * self.resource_pressure_cost)
        w2 = max(1.0, 50.0 * self.control_churn_cost)
        w3 = max(1.0, 800.0 * self.sla_violation_cost + 200.0 * self.fairness_cost)
        w4 = max(0.0, 250.0 * self.fairness_cost + 150.0 * self.sla_violation_cost)
        return {"w1": w1, "w2": w2, "w3": w3, "w4": w4}


class OptimizationProblemConfig(BaseModel):
    template: OptimizationTemplate = Field(default=OptimizationTemplate.JOINT_BALANCED)
    solver_mode: str = Field(default="incremental")
    active_objectives: List[str] = Field(
        default_factory=lambda: [
            "sla_violation_cost",
            "mobility_risk_cost",
            "control_churn_cost",
            "resource_pressure_cost",
            "fairness_cost",
        ]
    )
    active_constraints: List[str] = Field(
        default_factory=lambda: [
            "slice_capacity",
            "latency_bound",
            "jitter_bound",
            "gbr_floor",
            "snssai_alignment",
            "service_area_consistency",
            "ambr_consistency",
            "cross_domain_consistency",
        ]
    )
    decision_variables: List[str] = Field(
        default_factory=lambda: [
            "slice_assignment",
            "bandwidth_allocation",
            "sm_policy_update",
            "ursp_update",
            "rfsp",
            "allowed_snssais",
            "target_snssais",
            "mapping_snssais",
            "serv_area_res",
            "pras",
            "ue_ambr",
            "ue_slice_mbrs",
        ]
    )

    def normalized_for_domains(self, requested_domains: List["ControlDomain"]) -> "OptimizationProblemConfig":
        domains = set(requested_domains or [])
        active_objectives = list(dict.fromkeys(self.active_objectives))
        active_constraints = list(dict.fromkeys(self.active_constraints))
        decision_variables = list(dict.fromkeys(self.decision_variables))

        if ControlDomain.MOBILITY not in domains:
            active_objectives = [item for item in active_objectives if item != "mobility_risk_cost"]
            active_constraints = [
                item
                for item in active_constraints
                if item not in {"snssai_alignment", "service_area_consistency", "ambr_consistency", "cross_domain_consistency"}
            ]
            decision_variables = [
                item
                for item in decision_variables
                if item
                not in {
                    "rfsp",
                    "allowed_snssais",
                    "target_snssais",
                    "mapping_snssais",
                    "serv_area_res",
                    "pras",
                    "ue_ambr",
                    "ue_slice_mbrs",
                }
            ]
        if ControlDomain.QOS not in domains:
            active_constraints = [
                item for item in active_constraints if item not in {"slice_capacity", "latency_bound", "jitter_bound", "gbr_floor"}
            ]
            decision_variables = [
                item for item in decision_variables if item not in {"slice_assignment", "bandwidth_allocation", "sm_policy_update", "ursp_update"}
            ]
        return OptimizationProblemConfig(
            template=self.template,
            solver_mode=self.solver_mode,
            active_objectives=active_objectives,
            active_constraints=active_constraints,
            decision_variables=decision_variables,
        )


class GlobalControlIntent(BaseModel):
    session_id: str = ""
    snapshot_id: str = ""
    raw_input: str = ""
    user_goal: str = ""
    operation_type: str = "modify"
    urgency: str = "Normal"
    supi: str = ""
    app_id: str = ""
    app_name: Optional[str] = None
    target_flow_ids: List[str] = Field(default_factory=list)
    target_flow_names: List[str] = Field(default_factory=list)
    requested_domains: List[ControlDomain] = Field(default_factory=list)
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    mobility_triggers: List[PcfAmPolicyControlRequestTrigger] = Field(default_factory=list)
    active_constraints: List[str] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=list)
    forbidden_assumptions: List[str] = Field(default_factory=list)
    prompt_injections: Dict[str, str] = Field(default_factory=dict)

    @field_validator("objective_profile", mode="before")
    @classmethod
    def _normalize_objective_profile(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"profile_name": value}
        return value


class DomainTaskEnvelope(BaseModel):
    domain: ControlDomain
    session_id: str = ""
    snapshot_id: str = ""
    supi: str = ""
    goal: str = ""
    priority: str = "Normal"
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    required_evidence: List[str] = Field(default_factory=list)
    forbidden_assumptions: List[str] = Field(default_factory=list)
    prompt_injection: str = ""
    context_payload: Dict[str, Any] = Field(default_factory=dict)


class MobilityContextSnapshot(BaseModel):
    supi: str
    accessType: Optional[AccessType] = None
    accessTypes: Optional[List[AccessType]] = None
    ratType: Optional[RatType] = None
    ratTypes: Optional[List[RatType]] = None
    userLoc: Optional[UserLocation] = None
    guami: Optional[Guami] = None
    servingPlmn: Optional[PlmnIdNid] = None
    timeZone: Optional[str] = None
    presenceAreas: Dict[str, PresenceInfo] = Field(default_factory=dict)
    allowedSnssais: List[Snssai] = Field(default_factory=list)
    targetSnssais: List[Snssai] = Field(default_factory=list)
    mappingSnssais: List[MappingOfSnssai] = Field(default_factory=list)
    currentAssociationId: Optional[str] = None
    currentTriggers: List[PcfAmPolicyControlRequestTrigger] = Field(default_factory=list)
    currentServAreaRes: Optional[ServiceAreaRestriction] = None
    currentWlServAreaRes: Optional[WirelineServiceAreaRestriction] = None
    currentRfsp: Optional[int] = None
    currentSmfSelInfo: Optional[SmfSelectionData] = None
    missing_fields: List[str] = Field(default_factory=list)


class MobilityPolicyDraft(BaseModel):
    association_id: str
    request: PcfAmPolicyControlPolicyAssociationRequest
    policy: PcfAmPolicyControlPolicyAssociation
    rationale: str
    trigger_event: str
    expected_benefits: List[str] = Field(default_factory=list)


class DomainProposal(BaseModel):
    domain: ControlDomain
    status: DomainStatus = DomainStatus.READY
    rationale: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)
    prompt_injection: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    policy_drafts: List[Dict[str, Any]] = Field(default_factory=list)


class DomainVerdict(BaseModel):
    domain: ControlDomain
    status: DomainStatus
    summary: str = ""
    hard_conflicts: List[str] = Field(default_factory=list)
    soft_conflicts: List[str] = Field(default_factory=list)
    infeasible_reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class RevisionRequest(BaseModel):
    target_domain: ControlDomain
    conflict_type: str
    target_policy_ids: List[str] = Field(default_factory=list)
    target_objects: List[str] = Field(default_factory=list)
    reason: str = ""
    suggested_actions: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class UnifiedConstraintSet(BaseModel):
    hard_constraints: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class MediatorDecision(BaseModel):
    status: str = Field(default="approved")
    reason_summary: str = Field(default="")
    affected_domains: List[ControlDomain] = Field(default_factory=list)
    revision_requests: List[RevisionRequest] = Field(default_factory=list)
    unified_constraints: UnifiedConstraintSet = Field(default_factory=UnifiedConstraintSet)


class UnifiedControlPlan(BaseModel):
    session_id: str
    snapshot_id: str
    supi: str
    global_intent: GlobalControlIntent
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


class JointOptimizationRequest(BaseModel):
    session_id: str = ""
    snapshot_id: str = ""
    target_ues: List[str] = Field(default_factory=list)
    requested_domains: List[ControlDomain] = Field(default_factory=list)
    operation_intent: Dict[str, Any] = Field(default_factory=dict)
    traffic_state: Dict[str, Any] = Field(default_factory=dict)
    resource_state: Dict[str, Any] = Field(default_factory=dict)
    mobility_state: Dict[str, Any] = Field(default_factory=dict)
    policy_state: Dict[str, Any] = Field(default_factory=dict)
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    problem_config: OptimizationProblemConfig = Field(default_factory=OptimizationProblemConfig)
    prompt_injection: str = ""


class JointOptimizationResult(BaseModel):
    status: DomainStatus
    qos_plan: Dict[str, Any] = Field(default_factory=dict)
    mobility_plan: Dict[str, Any] = Field(default_factory=dict)
    am_plan: Dict[str, Any] = Field(default_factory=dict)
    cross_domain_verdicts: List[DomainVerdict] = Field(default_factory=list)
    objective_breakdown: Dict[str, Any] = Field(default_factory=dict)
    infeasible_reasons: List[str] = Field(default_factory=list)
