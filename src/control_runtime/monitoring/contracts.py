from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import ObservationPayload

MetricRelation = Literal["above", "below"]
AlertSeverity = Literal["warning", "critical"]
IntentStatus = Literal[
    "active",
    "degraded",
    "replanning",
    "satisfied",
    "failed",
]


class MonitorRule(ContractModel):
    owner_entity_id: NonEmptyStr
    metric_name: NonEmptyStr
    relation: MetricRelation
    threshold: float
    severity: AlertSeverity
    suggested_domains: list[Literal["qos", "mobility"]] = Field(min_length=1)


class MonitorAlert(ContractModel):
    alert_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    owner_entity_id: NonEmptyStr
    metric_name: NonEmptyStr
    observed_value: float
    relation: MetricRelation
    threshold: float
    severity: AlertSeverity
    suggested_domains: list[Literal["qos", "mobility"]] = Field(min_length=1)
    summary: NonEmptyStr


class MonitorResult(ObservationPayload):
    snapshot_id: NonEmptyStr
    scanned_metric_count: int = Field(ge=0)
    alerts: list[MonitorAlert]


class IntentBinding(ContractModel):
    entity_id: NonEmptyStr
    relation: NonEmptyStr


class IntentLifecycle(ContractModel):
    intent_id: NonEmptyStr
    bindings: list[IntentBinding] = Field(min_length=1)
    status: IntentStatus
    origin: Literal["user", "monitor"]


class LifecycleTransition(ContractModel):
    intent_id: NonEmptyStr
    previous_status: IntentStatus
    next_status: IntentStatus
    reason: NonEmptyStr

    @model_validator(mode="after")
    def reject_noop(self) -> LifecycleTransition:
        if self.previous_status == self.next_status:
            raise ValueError("lifecycle transition must change status")
        return self


__all__ = [
    "AlertSeverity",
    "IntentBinding",
    "IntentLifecycle",
    "IntentStatus",
    "LifecycleTransition",
    "MetricRelation",
    "MonitorAlert",
    "MonitorResult",
    "MonitorRule",
]
