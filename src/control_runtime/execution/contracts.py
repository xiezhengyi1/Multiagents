from __future__ import annotations

from typing import Literal

from pydantic import Field

from control_runtime.simulation import ExecutionAction, ExecutionFeedback
from shared.contracts import ContractModel, NonEmptyStr


class MetricRequirement(ContractModel):
    metric_name: NonEmptyStr
    relation: Literal["at_least", "at_most"]
    threshold: float


class MetricAssessment(ContractModel):
    metric_name: NonEmptyStr
    relation: Literal["at_least", "at_most"]
    threshold: float
    observed: float
    satisfied: bool


class AssuranceVerdict(ContractModel):
    status: Literal["satisfied", "violated"]
    assessments: list[MetricAssessment] = Field(min_length=1)


class ExecutionOutcome(ContractModel):
    """Deterministic result of one typed execution action."""

    action: ExecutionAction
    feedback: ExecutionFeedback
    assurance: AssuranceVerdict | None


__all__ = [
    "AssuranceVerdict",
    "ExecutionOutcome",
    "MetricAssessment",
    "MetricRequirement",
]
