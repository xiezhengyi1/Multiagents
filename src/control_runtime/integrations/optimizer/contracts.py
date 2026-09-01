from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal

from pydantic import Field, model_validator

from shared.contracts import ContractModel, ExecutionScope, JsonValue, NonEmptyStr

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


class OptimizationProblemConfig(ContractModel):
    template: OptimizationTemplate
    solver_mode: NonEmptyStr
    active_objectives: list[NonEmptyStr]
    active_constraints: list[NonEmptyStr]
    decision_variables: list[NonEmptyStr]
    qos_relaxation_ratio: float = Field(ge=0.0, le=1.0)
    slice_kpi_source: Literal["qos", "telemetry"]
    qos_feasibility_mode: Literal["hard", "soft"]
    enable_sla_constraints: bool

    @model_validator(mode="after")
    def validate_unique_members(self) -> "OptimizationProblemConfig":
        for name, values in (
            ("active_objectives", self.active_objectives),
            ("active_constraints", self.active_constraints),
            ("decision_variables", self.decision_variables),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        return self

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
        active_objectives = list(self.active_objectives)
        active_constraints = list(self.active_constraints)
        decision_variables = list(self.decision_variables)
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


class OptimizerInput(ContractModel):
    """Agent-projected input accepted at the optimizer integration boundary."""

    scope: ExecutionScope
    target_supis: list[NonEmptyStr]
    requested_domains: list[ControlDomain]
    grounding_decision: dict[str, JsonValue]
    objective_profile: ObjectiveProfile
    raw_intent: str

    @model_validator(mode="after")
    def validate_optimizer_input(self) -> "OptimizerInput":
        if self.scope.snapshot_id is None:
            raise ValueError("optimizer input requires scope.snapshot_id")
        if not self.target_supis:
            raise ValueError("optimizer input requires target_supis")
        if len(self.target_supis) != len(set(self.target_supis)):
            raise ValueError("target_supis must not contain duplicates")
        if not self.requested_domains:
            raise ValueError("optimizer input requires requested_domains")
        if len(self.requested_domains) != len(set(self.requested_domains)):
            raise ValueError("requested_domains must not contain duplicates")
        return self


class JointOptimizationRequest(ContractModel):
    scope: ExecutionScope
    target_ues: list[NonEmptyStr]
    requested_domains: list[ControlDomain]
    grounding_decision: dict[str, JsonValue]
    traffic_state: dict[str, JsonValue]
    resource_state: dict[str, JsonValue]
    mobility_state: dict[str, JsonValue]
    policy_state: dict[str, JsonValue]
    objective_profile: ObjectiveProfile
    problem_config: OptimizationProblemConfig

    @model_validator(mode="after")
    def validate_request_scope(self) -> "JointOptimizationRequest":
        if self.scope.snapshot_id is None:
            raise ValueError("joint optimization requires scope.snapshot_id")
        if not self.target_ues:
            raise ValueError("joint optimization requires target_ues")
        if len(self.target_ues) != len(set(self.target_ues)):
            raise ValueError("target_ues must not contain duplicates")
        if not self.requested_domains:
            raise ValueError("joint optimization requires requested_domains")
        if len(self.requested_domains) != len(set(self.requested_domains)):
            raise ValueError("requested_domains must not contain duplicates")
        return self


class JointOptimizationResult(ContractModel):
    status: DomainStatus
    qos_plan: dict[str, JsonValue]
    mobility_plan: dict[str, JsonValue]
    am_plan: dict[str, JsonValue]
    cross_domain_verdicts: list[DomainVerdict]
    objective_breakdown: dict[str, JsonValue]
    infeasible_reasons: list[NonEmptyStr]


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
