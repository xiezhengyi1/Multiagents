from __future__ import annotations

from typing import Tuple

from ...domain.control_plane import JointOptimizationRequest, ObjectiveProfile, OptimizationProblemConfig, OptimizationTemplate


def resolve_problem_config(request: JointOptimizationRequest) -> Tuple[OptimizationProblemConfig, ObjectiveProfile]:
    config = request.problem_config.normalized_for_domains(request.requested_domains)
    profile = request.objective_profile.model_copy(deep=True)

    if config.template == OptimizationTemplate.QOS_FIRST:
        config.solver_mode = "incremental"
        profile.sla_violation_cost = max(profile.sla_violation_cost, 1.4)
        profile.resource_pressure_cost = max(profile.resource_pressure_cost, 0.8)
        profile.control_churn_cost = min(profile.control_churn_cost, 0.4)
        profile.mobility_risk_cost = min(profile.mobility_risk_cost, 0.5)
    elif config.template == OptimizationTemplate.MOBILITY_FIRST:
        config.solver_mode = "incremental"
        profile.mobility_risk_cost = max(profile.mobility_risk_cost, 1.4)
        profile.control_churn_cost = max(profile.control_churn_cost, 0.7)
        profile.sla_violation_cost = max(profile.sla_violation_cost, 0.9)
    elif config.template == OptimizationTemplate.STABILITY_FIRST:
        config.solver_mode = "incremental"
        profile.control_churn_cost = max(profile.control_churn_cost, 1.2)
        profile.fairness_cost = max(profile.fairness_cost, 0.6)
    elif config.template == OptimizationTemplate.CONGESTION_RELIEF:
        config.solver_mode = "hybrid"
        profile.resource_pressure_cost = max(profile.resource_pressure_cost, 1.3)
        profile.sla_violation_cost = max(profile.sla_violation_cost, 1.0)
        profile.control_churn_cost = min(profile.control_churn_cost, 0.35)
    else:
        config.solver_mode = config.solver_mode or "incremental"

    if bool((request.grounding_decision or {}).get("preserve_current_slice", False)):
        # IEA subscription evidence prohibits a migration. Keep the serving
        # slice fixed, but let the hybrid optimizer find the best QoS envelope
        # that can still be delivered on that entitlement.
        config.solver_mode = "hybrid"

    return config, profile


__all__ = ["resolve_problem_config"]
