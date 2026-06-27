from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from shared.tools.wrapper_think import tool_with_reason

from .contracts import MonitorAlert, MonitorResult
from .flow_monitor import FlowTelemetryMonitor
from .requirement_agent import AutonomousRequirementAgent


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _compact_alert(alert: MonitorAlert) -> Dict[str, Any]:
    return {
        "alert_id": alert.alert_id,
        "snapshot_id": alert.snapshot_id,
        "supi": alert.supi,
        "app_id": alert.app_id,
        "app_name": alert.app_name,
        "flow_id": alert.flow_id,
        "flow_name": alert.flow_name,
        "severity": alert.severity,
        "status": alert.status,
        "violated_metrics": list(alert.violated_metrics),
        "recommended_consumer": alert.recommended_consumer,
        "reuse_binding": alert.reuse_binding,
        "summary": alert.summary,
    }


def _latest_alerts(cache: Dict[str, Any]) -> List[MonitorAlert]:
    result = cache.get("latest_monitor_result")
    if isinstance(result, MonitorResult):
        return list(result.alerts)
    return []


def _find_alert(cache: Dict[str, Any], alert_id: str) -> MonitorAlert:
    normalized = str(alert_id or "").strip()
    if not normalized:
        raise ValueError("alert_id is required")
    for alert in _latest_alerts(cache):
        if alert.alert_id == normalized:
            return alert
    raise LookupError(f"monitor alert not found in latest result: alert_id={normalized}")


class _TemplateFallbackLlm:
    def invoke(self, _: str) -> str:
        return ""


def build_monitoring_tools(
    *,
    load_snapshot_by_id: Callable[[str], Optional[Dict[str, Any]]],
    requirement_agent: Optional[AutonomousRequirementAgent] = None,
    tolerance_ratio: float = 0.10,
    max_alerts: int = 20,
) -> tuple[List[Any], Dict[str, Any]]:
    monitor = FlowTelemetryMonitor(
        load_snapshot_by_id=load_snapshot_by_id,
        tolerance_ratio=tolerance_ratio,
        max_alerts=max_alerts,
    )
    requirement_rewriter = requirement_agent or AutonomousRequirementAgent(llm=_TemplateFallbackLlm())
    cache: Dict[str, Any] = {}

    @tool_with_reason
    def evaluate_flow_telemetry(snapshot_id: str) -> str:
        """Evaluate current flow telemetry against SLA targets and cache compact monitor alerts."""
        result = monitor.evaluate_snapshot(snapshot_id)
        cache["latest_monitor_result"] = result
        return _json(
            {
                "status": "ok",
                "snapshot_id": result.snapshot_id,
                "scanned_flow_count": result.scanned_flow_count,
                "summary": result.summary,
                "alerts": [_compact_alert(alert) for alert in result.alerts],
            }
        )

    @tool_with_reason
    def inspect_monitor_alert(alert_id: str) -> str:
        """Return full evidence for one cached monitor alert by alert_id."""
        alert = _find_alert(cache, alert_id)
        return _json({"status": "ok", "alert": alert.to_dict()})

    @tool_with_reason
    def draft_monitor_reentry_requirement(
        alert_id: str,
        previous_user_intent: str = "",
        previous_control_context: str = "",
    ) -> str:
        """Rewrite one cached monitor alert into a natural-language autonomous reentry requirement."""
        alert = _find_alert(cache, alert_id)
        draft = requirement_rewriter.generate_requirement(
            alert,
            previous_user_intent=previous_user_intent,
            extra_context={
                "previous_control_context": previous_control_context,
                "context_strategy": "bounded_recent_context_plus_retrievable_memory",
            },
        )
        payload = draft.to_dict()
        cache["latest_requirement_draft"] = payload
        return _json({"status": "ok", "requirement": payload})

    return [evaluate_flow_telemetry, inspect_monitor_alert, draft_monitor_reentry_requirement], cache


__all__ = ["build_monitoring_tools"]
