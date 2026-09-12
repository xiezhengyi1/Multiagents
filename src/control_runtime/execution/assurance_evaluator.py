from __future__ import annotations

from collections.abc import Sequence

from control_runtime.simulation import ExecutionFeedback

from .contracts import AssuranceVerdict, MetricAssessment, MetricRequirement


class AssuranceEvaluator:
    """Evaluate explicit objective metrics returned by the selected backend."""

    def __init__(self, *, requirements: Sequence[MetricRequirement]) -> None:
        if not requirements:
            raise ValueError("assurance requirements cannot be empty")
        self.requirements = tuple(requirements)

    def evaluate(self, feedback: ExecutionFeedback) -> AssuranceVerdict:
        assessments: list[MetricAssessment] = []
        for requirement in self.requirements:
            observed = feedback.objective_metrics[requirement.metric_name]
            satisfied = (
                observed >= requirement.threshold
                if requirement.relation == "at_least"
                else observed <= requirement.threshold
            )
            assessments.append(
                MetricAssessment(
                    metric_name=requirement.metric_name,
                    relation=requirement.relation,
                    threshold=requirement.threshold,
                    observed=observed,
                    satisfied=satisfied,
                )
            )
        return AssuranceVerdict(
            status=(
                "satisfied"
                if all(assessment.satisfied for assessment in assessments)
                else "violated"
            ),
            assessments=assessments,
        )


__all__ = ["AssuranceEvaluator"]
