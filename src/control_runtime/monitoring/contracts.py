from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MonitorAlert:
    alert_id: str
    snapshot_id: str
    supi: str
    app_id: str
    app_name: str
    flow_id: str
    flow_name: str
    severity: str
    status: str
    violated_metrics: List[str]
    metric_deltas: Dict[str, Dict[str, float]]
    recommended_consumer: str
    summary: str
    suggested_domains: List[str] = field(default_factory=list)
    reuse_binding: bool = True
    observed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "snapshot_id": self.snapshot_id,
            "supi": self.supi,
            "app_id": self.app_id,
            "app_name": self.app_name,
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "severity": self.severity,
            "status": self.status,
            "violated_metrics": list(self.violated_metrics),
            "metric_deltas": dict(self.metric_deltas),
            "recommended_consumer": self.recommended_consumer,
            "summary": self.summary,
            "suggested_domains": list(self.suggested_domains),
            "reuse_binding": self.reuse_binding,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class MonitorResult:
    snapshot_id: str
    scanned_flow_count: int
    alerts: List[MonitorAlert] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "scanned_flow_count": self.scanned_flow_count,
            "alerts": [alert.to_dict() for alert in self.alerts],
            "summary": self.summary,
        }


@dataclass(frozen=True)
class RequirementDraft:
    source_alert_id: str
    snapshot_id: str
    user_input: str
    routing_hint: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_alert_id": self.source_alert_id,
            "snapshot_id": self.snapshot_id,
            "user_input": self.user_input,
            "routing_hint": dict(self.routing_hint),
        }


@dataclass(frozen=True)
class ReentryLoopResult:
    snapshot_id: str
    monitor_result: MonitorResult
    reentry_triggered: bool
    requirement: Optional[RequirementDraft] = None
    control_result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "monitor_result": self.monitor_result.to_dict(),
            "reentry_triggered": self.reentry_triggered,
            "requirement": self.requirement.to_dict() if self.requirement is not None else None,
            "control_result": self.control_result,
        }


@dataclass(frozen=True)
class WatchLoopTriggerRecord:
    trigger_type: str
    snapshot_id: str
    user_input: str
    source_alert_id: str = ""
    previous_context: str = ""
    context_truncated: bool = False
    routing_hint: Dict[str, Any] = field(default_factory=dict)
    control_result: Any = None
    data_flywheel: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "snapshot_id": self.snapshot_id,
            "user_input": self.user_input,
            "source_alert_id": self.source_alert_id,
            "previous_context": self.previous_context,
            "context_truncated": self.context_truncated,
            "routing_hint": dict(self.routing_hint),
            "control_result": self.control_result,
            "data_flywheel": dict(self.data_flywheel),
        }


@dataclass(frozen=True)
class WatchLoopIterationResult:
    snapshot_id: str
    monitor_result: Optional[MonitorResult] = None
    records: List[WatchLoopTriggerRecord] = field(default_factory=list)
    user_input_seen: bool = False
    pending_monitor_alerts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pending_monitor_alert_count(self) -> int:
        return len(self.pending_monitor_alerts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "monitor_result": self.monitor_result.to_dict() if self.monitor_result is not None else None,
            "records": [record.to_dict() for record in self.records],
            "user_input_seen": self.user_input_seen,
            "pending_monitor_alert_count": self.pending_monitor_alert_count,
            "pending_monitor_alerts": [dict(alert) for alert in self.pending_monitor_alerts],
        }
