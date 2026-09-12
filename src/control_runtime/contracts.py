from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field

from shared.contracts import ContractModel, JsonValue, NonEmptyStr


class ControlDomain(str, Enum):
    QOS = "qos"
    MOBILITY = "mobility"


class ObjectiveProfile(ContractModel):
    profile_name: NonEmptyStr
    sla_violation_cost: float = Field(ge=0.0)
    mobility_risk_cost: float = Field(ge=0.0)
    control_churn_cost: float = Field(ge=0.0)
    resource_pressure_cost: float = Field(ge=0.0)
    fairness_cost: float = Field(ge=0.0)


class DomainStatus(str, Enum):
    READY = "ready"
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    INCOMPLETE_CONTEXT = "incomplete_context"
    SKIPPED = "skipped"
    FAILED = "failed"


class DomainVerdict(ContractModel):
    domain: ControlDomain
    status: DomainStatus
    summary: NonEmptyStr
    hard_conflicts: list[NonEmptyStr]
    soft_conflicts: list[NonEmptyStr]
    infeasible_reasons: list[NonEmptyStr]
    metrics: dict[str, JsonValue]


StandardObjectiveProfileName = Literal["balanced", "latency", "throughput", "stability"]


def standard_objective_profile(
    profile_name: StandardObjectiveProfileName,
) -> ObjectiveProfile:
    """Build the explicit baseline used by deterministic single-agent routing."""

    return ObjectiveProfile(
        profile_name=profile_name,
        sla_violation_cost=1.0,
        mobility_risk_cost=0.8,
        control_churn_cost=0.5,
        resource_pressure_cost=0.7,
        fairness_cost=0.3,
    )


__all__ = [
    "ControlDomain",
    "DomainStatus",
    "DomainVerdict",
    "ObjectiveProfile",
    "StandardObjectiveProfileName",
    "standard_objective_profile",
]
