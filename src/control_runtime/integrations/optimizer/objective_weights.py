from __future__ import annotations

from dataclasses import dataclass

from ...contracts import ObjectiveProfile


@dataclass(frozen=True, slots=True)
class QosOptimizerWeights:
    """Semantic weights adapted to the QoS solver's positional inputs."""

    resource_pressure: float
    control_churn: float
    sla_and_fairness: float
    fairness_and_sla_tail: float

    def solver_arguments(self) -> tuple[float, float, float, float]:
        return (
            self.resource_pressure,
            self.control_churn,
            self.sla_and_fairness,
            self.fairness_and_sla_tail,
        )

    def to_payload(self) -> dict[str, float]:
        return {
            "resource_pressure": self.resource_pressure,
            "control_churn": self.control_churn,
            "sla_and_fairness": self.sla_and_fairness,
            "fairness_and_sla_tail": self.fairness_and_sla_tail,
        }


def build_qos_optimizer_weights(profile: ObjectiveProfile) -> QosOptimizerWeights:
    """Translate domain costs at the optimizer integration boundary."""

    return QosOptimizerWeights(
        resource_pressure=max(1.0, 100.0 * profile.resource_pressure_cost),
        control_churn=max(1.0, 50.0 * profile.control_churn_cost),
        sla_and_fairness=max(
            1.0,
            800.0 * profile.sla_violation_cost + 200.0 * profile.fairness_cost,
        ),
        fairness_and_sla_tail=max(
            0.0,
            250.0 * profile.fairness_cost + 150.0 * profile.sla_violation_cost,
        ),
    )


__all__ = ["QosOptimizerWeights", "build_qos_optimizer_weights"]
