from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from ...contracts import ControlDomain, ObjectiveProfile
from ...diagnostics.mediation.contracts import DomainStatus, DomainVerdict


class OptimizationTemplate(str, Enum):
    QOS_FIRST = "qos_first"
    MOBILITY_FIRST = "mobility_first"
    JOINT_BALANCED = "joint_balanced"
    STABILITY_FIRST = "stability_first"
    CONGESTION_RELIEF = "congestion_relief"


SESSION_DECISION_VARIABLES = [
    "slice_assignment",
    "bandwidth_allocation",
    "sm_policy_update",
    "ursp_update",
]
MOBILITY_DECISION_VARIABLES = [
    "rfsp",
    "allowed_snssais",
    "target_snssais",
    "mapping_snssais",
    "serv_area_res",
    "pras",
    "ue_ambr",
    "ue_slice_mbrs",
    "triggers",
]
SESSION_CONSTRAINTS = [
    "slice_capacity",
    "latency_bound",
    "jitter_bound",
    "loss_bound",
    "gbr_floor",
    "service_type_sst_match",
]
MOBILITY_CONSTRAINTS = [
    "target_subset_allowed",
    "mandatory_triggers",
    "service_area_context",
]
COUPLING_CONSTRAINTS = [
    "snssai_alignment",
    "service_area_consistency",
    "ambr_consistency",
    "cross_domain_consistency",
]


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
            *SESSION_CONSTRAINTS,
            *MOBILITY_CONSTRAINTS,
            *COUPLING_CONSTRAINTS,
        ]
    )
    decision_variables: List[str] = Field(
        default_factory=lambda: [*SESSION_DECISION_VARIABLES, *MOBILITY_DECISION_VARIABLES]
    )
    qos_relaxation_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    slice_kpi_source: Literal["qos", "telemetry"] = Field(default="qos")
    qos_feasibility_mode: Literal["hard", "soft"] = Field(default="hard")
    enable_sla_constraints: bool = Field(default=True)

    def grouped_decision_variables(self) -> Dict[str, List[str]]:
        variables = set(self.decision_variables)
        return {
            "session_domain": [item for item in SESSION_DECISION_VARIABLES if item in variables],
            "mobility_domain": [item for item in MOBILITY_DECISION_VARIABLES if item in variables],
            "cross_domain": [],
        }

    def grouped_constraints(self) -> Dict[str, List[str]]:
        constraints = set(self.active_constraints)
        return {
            "session_feasibility": [item for item in SESSION_CONSTRAINTS if item in constraints],
            "mobility_feasibility": [item for item in MOBILITY_CONSTRAINTS if item in constraints],
            "coupling": [item for item in COUPLING_CONSTRAINTS if item in constraints],
        }

    def normalized_for_domains(self, requested_domains: List[ControlDomain]) -> "OptimizationProblemConfig":
        domains = set(requested_domains)
        active_objectives = list(dict.fromkeys(self.active_objectives))
        active_constraints = list(dict.fromkeys(self.active_constraints))
        decision_variables = list(dict.fromkeys(self.decision_variables))
        if ControlDomain.MOBILITY not in domains:
            active_objectives = [item for item in active_objectives if item != "mobility_risk_cost"]
            active_constraints = [
                item for item in active_constraints if item not in set(MOBILITY_CONSTRAINTS + COUPLING_CONSTRAINTS)
            ]
            decision_variables = [item for item in decision_variables if item not in set(MOBILITY_DECISION_VARIABLES)]
        if ControlDomain.QOS not in domains:
            active_constraints = [item for item in active_constraints if item not in set(SESSION_CONSTRAINTS)]
            decision_variables = [item for item in decision_variables if item not in set(SESSION_DECISION_VARIABLES)]
        return OptimizationProblemConfig(
            template=self.template,
            solver_mode=self.solver_mode,
            active_objectives=active_objectives,
            active_constraints=active_constraints,
            decision_variables=decision_variables,
            qos_relaxation_ratio=self.qos_relaxation_ratio,
            slice_kpi_source=self.slice_kpi_source,
            qos_feasibility_mode=self.qos_feasibility_mode,
            enable_sla_constraints=self.enable_sla_constraints,
        )


class OptimizerInput(BaseModel):
    """Agent-projected input accepted at the optimizer integration boundary."""

    session_id: str
    snapshot_id: str
    target_supis: List[str]
    requested_domains: List[ControlDomain]
    grounding_decision: Dict[str, Any]
    objective_profile: ObjectiveProfile
    raw_intent: str


class JointOptimizationRequest(BaseModel):
    session_id: str = ""
    snapshot_id: str = ""
    target_ues: List[str] = Field(default_factory=list)
    requested_domains: List[ControlDomain] = Field(default_factory=list)
    grounding_decision: Dict[str, Any] = Field(default_factory=dict)
    traffic_state: Dict[str, Any] = Field(default_factory=dict)
    resource_state: Dict[str, Any] = Field(default_factory=dict)
    mobility_state: Dict[str, Any] = Field(default_factory=dict)
    policy_state: Dict[str, Any] = Field(default_factory=dict)
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    problem_config: OptimizationProblemConfig = Field(default_factory=OptimizationProblemConfig)


class JointOptimizationResult(BaseModel):
    status: DomainStatus
    qos_plan: Dict[str, Any] = Field(default_factory=dict)
    mobility_plan: Dict[str, Any] = Field(default_factory=dict)
    am_plan: Dict[str, Any] = Field(default_factory=dict)
    cross_domain_verdicts: List[DomainVerdict] = Field(default_factory=list)
    objective_breakdown: Dict[str, Any] = Field(default_factory=dict)
    infeasible_reasons: List[str] = Field(default_factory=list)


__all__ = [
    "COUPLING_CONSTRAINTS",
    "JointOptimizationRequest",
    "JointOptimizationResult",
    "MOBILITY_CONSTRAINTS",
    "MOBILITY_DECISION_VARIABLES",
    "OptimizationProblemConfig",
    "OptimizationTemplate",
    "OptimizerInput",
    "SESSION_CONSTRAINTS",
    "SESSION_DECISION_VARIABLES",
]
