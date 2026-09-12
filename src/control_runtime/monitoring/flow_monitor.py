from __future__ import annotations

from collections.abc import Sequence

from control_runtime.simulation import EnvironmentSnapshot

from .contracts import MonitorAlert, MonitorResult, MonitorRule


class FlowTelemetryMonitor:
    """Apply explicit experiment rules to one normalized environment snapshot."""

    def __init__(self, *, rules: Sequence[MonitorRule]) -> None:
        if not rules:
            raise ValueError("monitor rules cannot be empty")
        keys = [(rule.owner_entity_id, rule.metric_name) for rule in rules]
        if len(keys) != len(set(keys)):
            raise ValueError("monitor rule owner/metric keys must be unique")
        self.rules = tuple(rules)

    def evaluate(self, snapshot: EnvironmentSnapshot) -> MonitorResult:
        metrics = {
            (metric.owner_entity_id, metric.metric_name): metric
            for metric in snapshot.metrics
        }
        alerts: list[MonitorAlert] = []
        for rule in self.rules:
            key = (rule.owner_entity_id, rule.metric_name)
            metric = metrics[key]
            if isinstance(metric.value, bool) or not isinstance(
                metric.value, (int, float)
            ):
                raise TypeError(
                    f"monitor metric {rule.owner_entity_id}/{rule.metric_name} must be numeric"
                )
            observed = float(metric.value)
            violated = (
                observed > rule.threshold
                if rule.relation == "above"
                else observed < rule.threshold
            )
            if not violated:
                continue
            comparison = ">" if rule.relation == "above" else "<"
            alerts.append(
                MonitorAlert(
                    alert_id=(
                        f"{snapshot.snapshot_id}:{rule.owner_entity_id}:"
                        f"{rule.metric_name}"
                    ),
                    snapshot_id=snapshot.snapshot_id,
                    owner_entity_id=rule.owner_entity_id,
                    metric_name=rule.metric_name,
                    observed_value=observed,
                    relation=rule.relation,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    suggested_domains=rule.suggested_domains,
                    summary=(
                        f"{rule.owner_entity_id} {rule.metric_name}={observed:g} "
                        f"violates {comparison} {rule.threshold:g}"
                    ),
                )
            )
        return MonitorResult(
            snapshot_id=snapshot.snapshot_id,
            scanned_metric_count=len(snapshot.metrics),
            alerts=alerts,
        )


__all__ = ["FlowTelemetryMonitor"]
