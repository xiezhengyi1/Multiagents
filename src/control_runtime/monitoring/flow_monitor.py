from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Iterable, Optional

from .contracts import MonitorAlert, MonitorResult


MetricSpec = tuple[str, str, str]

UPPER_BOUND_METRICS: tuple[MetricSpec, ...] = (
    ("telemetry.latency", "sla.latency", "latency"),
    ("telemetry.jitter", "sla.jitter", "jitter"),
    ("telemetry.loss_rate", "sla.loss_rate", "loss_rate"),
)
LOWER_BOUND_METRICS: tuple[MetricSpec, ...] = (
    ("telemetry.throughput_ul", "sla.guaranteed_bandwidth_ul", "throughput_ul"),
    ("telemetry.throughput_dl", "sla.guaranteed_bandwidth_dl", "throughput_dl"),
)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_get(payload: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _metric_snapshot(metrics: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        owner_key = str(metric.get("owner_key") or "").strip()
        metric_name = str(metric.get("metric_name") or "").strip()
        if not owner_key or not metric_name:
            continue
        latest.setdefault(owner_key, {})[metric_name] = {
            "value": metric.get("metric_value"),
            "observed_at": str(metric.get("observed_at") or "").strip(),
        }
    return latest


def _flow_property_id(properties: Dict[str, Any]) -> str:
    return str(properties.get("flow_id") or properties.get("id") or "").strip()


def _binding_from_node_key(node_key: str) -> Dict[str, str]:
    parts = str(node_key or "").strip().split(":")
    if len(parts) < 4 or parts[0] != "flow":
        return {}
    return {
        "supi": parts[1].strip(),
        "app_id": parts[2].strip(),
        "flow_id": parts[3].strip(),
    }


def _severity(max_ratio: float, violation_count: int) -> str:
    if max_ratio >= 0.50 or violation_count >= 2:
        return "critical"
    if max_ratio >= 0.20:
        return "major"
    return "minor"


class FlowTelemetryMonitor:
    def __init__(
        self,
        *,
        load_snapshot_by_id: Callable[[str], Optional[Dict[str, Any]]],
        tolerance_ratio: float = 0.10,
        max_alerts: int = 20,
    ) -> None:
        self.load_snapshot_by_id = load_snapshot_by_id
        self.tolerance_ratio = max(0.0, float(tolerance_ratio))
        self.max_alerts = max(1, int(max_alerts))

    def evaluate_snapshot(self, snapshot_id: str) -> MonitorResult:
        normalized_snapshot_id = str(snapshot_id or "").strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id is required")
        snapshot = self.load_snapshot_by_id(normalized_snapshot_id)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"network graph snapshot not found: snapshot_id={normalized_snapshot_id}")

        metrics_by_owner = _metric_snapshot(snapshot.get("metrics") or [])
        alerts: list[MonitorAlert] = []
        scanned = 0
        for node in snapshot.get("nodes") or []:
            if not isinstance(node, dict) or str(node.get("node_type") or "") != "flow":
                continue
            scanned += 1
            alert = self._evaluate_flow_node(
                snapshot_id=normalized_snapshot_id,
                node_key=str(node.get("node_key") or "").strip(),
                label=str(node.get("label") or "").strip(),
                properties=dict(node.get("properties") or {}),
                metric_values=metrics_by_owner.get(str(node.get("node_key") or "").strip(), {}),
            )
            if alert is not None:
                alerts.append(alert)

        if scanned == 0:
            for app in snapshot.get("apps") or []:
                if not isinstance(app, dict):
                    continue
                app_id = str(app.get("app_id") or app.get("id") or "").strip()
                app_name = str(app.get("app_name") or app.get("name") or "").strip()
                supi = str(app.get("supi") or "").strip()
                for flow in app.get("flows") or []:
                    if not isinstance(flow, dict):
                        continue
                    scanned += 1
                    flow_id = str(flow.get("flow_id") or flow.get("id") or "").strip()
                    node_key = f"flow:{supi}:{app_id}:{flow_id}"
                    properties = dict(flow)
                    properties.setdefault("id", flow_id)
                    properties.setdefault("flow_id", flow_id)
                    properties.setdefault("supi", supi)
                    properties.setdefault("app_id", app_id)
                    properties.setdefault("app_name", app_name)
                    alert = self._evaluate_flow_node(
                        snapshot_id=normalized_snapshot_id,
                        node_key=node_key,
                        label=str(flow.get("flow_name") or flow.get("name") or flow_id).strip(),
                        properties=properties,
                        metric_values={},
                    )
                    if alert is not None:
                        alerts.append(alert)

        alerts.sort(key=lambda alert: (-len(alert.violated_metrics), alert.severity, alert.flow_id))
        alerts = alerts[: self.max_alerts]
        return MonitorResult(
            snapshot_id=normalized_snapshot_id,
            scanned_flow_count=scanned,
            alerts=alerts,
            summary=f"scanned {scanned} flow nodes; found {len(alerts)} SLA alerts",
        )

    def _observed_value(self, properties: Dict[str, Any], metric_values: Dict[str, Dict[str, Any]], metric_name: str) -> tuple[Optional[float], str]:
        metric_payload = metric_values.get(metric_name)
        if isinstance(metric_payload, dict):
            value = _as_float(metric_payload.get("value"))
            if value is not None:
                return value, str(metric_payload.get("observed_at") or "")
        value = _as_float(_nested_get(properties, metric_name))
        return value, ""

    def _evaluate_flow_node(
        self,
        *,
        snapshot_id: str,
        node_key: str,
        label: str,
        properties: Dict[str, Any],
        metric_values: Dict[str, Dict[str, Any]],
    ) -> Optional[MonitorAlert]:
        violated: list[str] = []
        deltas: Dict[str, Dict[str, float]] = {}
        observed_at = ""
        max_ratio = 0.0

        for observed_key, target_key, _short_name in UPPER_BOUND_METRICS:
            observed, timestamp = self._observed_value(properties, metric_values, observed_key)
            target = _as_float(_nested_get(properties, target_key))
            if observed is None or target is None or target <= 0.0:
                continue
            limit = target * (1.0 + self.tolerance_ratio)
            if observed > limit:
                ratio = (observed - target) / target
                max_ratio = max(max_ratio, ratio)
                observed_at = timestamp or observed_at
                violated.append(observed_key)
                deltas[observed_key] = {"observed": observed, "target": target, "deviation_ratio": ratio}

        for observed_key, target_key, _short_name in LOWER_BOUND_METRICS:
            observed, timestamp = self._observed_value(properties, metric_values, observed_key)
            target = _as_float(_nested_get(properties, target_key))
            if observed is None or target is None or target <= 0.0:
                continue
            limit = target * (1.0 - self.tolerance_ratio)
            if observed < limit:
                ratio = (target - observed) / target
                max_ratio = max(max_ratio, ratio)
                observed_at = timestamp or observed_at
                violated.append(observed_key)
                deltas[observed_key] = {"observed": observed, "target": target, "deviation_ratio": ratio}

        if not violated:
            return None

        binding_from_key = _binding_from_node_key(node_key)
        flow_id = _flow_property_id(properties) or binding_from_key.get("flow_id", "")
        flow_name = str(properties.get("flow_name") or properties.get("name") or label or flow_id).strip()
        supi = str(properties.get("supi") or binding_from_key.get("supi") or "").strip()
        app_id = str(properties.get("app_id") or binding_from_key.get("app_id") or "").strip()
        app_name = str(properties.get("app_name") or "").strip()
        alert_id = hashlib.sha1(f"{snapshot_id}|{node_key}|{','.join(sorted(violated))}".encode("utf-8")).hexdigest()[:16]
        severity = _severity(max_ratio, len(violated))
        return MonitorAlert(
            alert_id=alert_id,
            snapshot_id=snapshot_id,
            supi=supi,
            app_id=app_id,
            app_name=app_name,
            flow_id=flow_id,
            flow_name=flow_name,
            severity=severity,
            status="violated",
            violated_metrics=violated,
            metric_deltas=deltas,
            recommended_consumer="autonomous_requirement_agent",
            suggested_domains=["qos"],
            reuse_binding=True,
            observed_at=observed_at,
            summary=f"Flow {flow_name or flow_id} violates SLA metrics: {', '.join(violated)}",
        )
