from __future__ import annotations

from ...contracts import ObjectiveProfile
from .contracts import (
    JointOptimizationRequest,
    OptimizationProblemConfig,
    OptimizationTemplate,
)


def resolve_problem_config(request: JointOptimizationRequest) -> tuple[OptimizationProblemConfig, ObjectiveProfile]:
    config = request.problem_config.normalized_for_domains(request.requested_domains)
    profile = request.objective_profile

    if config.template == OptimizationTemplate.QOS_FIRST:
        config = config.model_copy(update={"solver_mode": "incremental"})
        profile = profile.model_copy(
            update={
                "sla_violation_cost": max(profile.sla_violation_cost, 1.4),
                "resource_pressure_cost": max(profile.resource_pressure_cost, 0.8),
                "control_churn_cost": min(profile.control_churn_cost, 0.4),
                "mobility_risk_cost": min(profile.mobility_risk_cost, 0.5),
            }
        )
    elif config.template == OptimizationTemplate.MOBILITY_FIRST:
        config = config.model_copy(update={"solver_mode": "incremental"})
        profile = profile.model_copy(
            update={
                "mobility_risk_cost": max(profile.mobility_risk_cost, 1.4),
                "control_churn_cost": max(profile.control_churn_cost, 0.7),
                "sla_violation_cost": max(profile.sla_violation_cost, 0.9),
            }
        )
    elif config.template == OptimizationTemplate.STABILITY_FIRST:
        config = config.model_copy(update={"solver_mode": "incremental"})
        profile = profile.model_copy(
            update={
                "control_churn_cost": max(profile.control_churn_cost, 1.2),
                "fairness_cost": max(profile.fairness_cost, 0.6),
            }
        )
    elif config.template == OptimizationTemplate.CONGESTION_RELIEF:
        config = config.model_copy(update={"solver_mode": "hybrid"})
        profile = profile.model_copy(
            update={
                "resource_pressure_cost": max(profile.resource_pressure_cost, 1.3),
                "sla_violation_cost": max(profile.sla_violation_cost, 1.0),
                "control_churn_cost": min(profile.control_churn_cost, 0.35),
            }
        )

    if bool((request.grounding_decision or {}).get("preserve_current_slice", False)):
        # IEA subscription evidence prohibits a migration. Keep the serving
        # slice fixed, but let the hybrid optimizer find the best QoS envelope
        # that can still be delivered on that entitlement.
        config = config.model_copy(update={"solver_mode": "hybrid"})

    return config, profile


__all__ = ["resolve_problem_config"]
