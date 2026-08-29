from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from control_runtime.monitoring.contracts import IntentStatus, MonitorAlert, MonitorResult
from control_runtime.monitoring.flow_monitor import FlowTelemetryMonitor
from control_runtime.monitoring.watch_loop import (
    AutonomousWatchLoop,
    build_monitor_reentry_requirement,
)
import control_runtime.monitoring.watch_loop as watch_loop_module


def _alert(
    flow_id: str = "flow-1",
    *,
    alert_id: str = "alert-1",
    reuse_binding: bool = True,
) -> MonitorAlert:
    return MonitorAlert(
        alert_id=alert_id,
        snapshot_id="snapshot-1",
        supi="imsi-001",
        app_id="app-1",
        app_name="video",
        flow_id=flow_id,
        flow_name=flow_id,
        severity="major",
        status="violated",
        violated_metrics=["telemetry.latency"],
        metric_deltas={"telemetry.latency": {"observed": 30.0, "target": 10.0}},
        recommended_consumer="intent_lifecycle",
        summary="latency violated",
        suggested_domains=["qos"],
        reuse_binding=reuse_binding,
    )


def _monitor_result(*alerts: MonitorAlert) -> MonitorResult:
    return MonitorResult(
        snapshot_id="snapshot-1",
        scanned_flow_count=2,
        alerts=list(alerts),
    )


class SequenceMonitor:
    def __init__(self, *results: MonitorResult) -> None:
        self.results = list(results)

    def evaluate_snapshot(self, _snapshot_id: str) -> MonitorResult:
        return self.results.pop(0)


class RecordingOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, user_input: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"user_input": user_input, **kwargs})
        binding = kwargs.get("routing_hint", {}).get("monitor_binding", {}) if kwargs.get("routing_hint") else {}
        return SimpleNamespace(
            session_id=f"session-{len(self.calls)}",
            snapshot_id=kwargs["snapshot_id"],
            completed=True,
            round_traces=[
                {
                    "grounding_decision": {
                        "flows": [
                            {
                                "supi": binding.get("supi", "imsi-001"),
                                "app_id": binding.get("app_id", "app-1"),
                                "flow_id": binding.get("flow_id", "flow-1"),
                            }
                        ]
                    }
                }
            ],
        )


def test_monitor_requirement_is_deterministic_and_has_no_llm_contract() -> None:
    draft = build_monitor_reentry_requirement(_alert())

    assert "沿用当前业务绑定" in draft.user_input
    assert set(draft.routing_hint) == {
        "source",
        "source_alert_id",
        "requested_domains",
        "reuse_binding",
        "severity",
        "monitor_binding",
        "violated_metrics",
        "metric_deltas",
    }
    source = inspect.getsource(watch_loop_module).lower()
    assert "requirement_agent" not in source
    assert "requirement_llm" not in source
    assert "重新确认业务绑定" in build_monitor_reentry_requirement(
        _alert(reuse_binding=False)
    ).user_input


def test_flow_monitor_raises_alert_from_sla_threshold_rules() -> None:
    snapshot = {
        "nodes": [
            {
                "node_type": "flow",
                "node_key": "flow:imsi-001:app-1:flow-1",
                "label": "video",
                "properties": {
                    "supi": "imsi-001",
                    "app_id": "app-1",
                    "flow_id": "flow-1",
                    "telemetry": {"latency": 16.0},
                    "sla": {"latency": 10.0},
                },
            }
        ]
    }
    monitor = FlowTelemetryMonitor(load_snapshot_by_id=lambda _snapshot_id: snapshot)

    result = monitor.evaluate_snapshot("snapshot-1")

    assert result.scanned_flow_count == 1
    assert result.alerts[0].severity == "critical"
    assert result.alerts[0].recommended_consumer == "intent_lifecycle"


def test_incident_is_suppressed_until_recovery_then_can_recur() -> None:
    alert = _alert()
    monitor = SequenceMonitor(
        _monitor_result(alert),
        _monitor_result(alert),
        _monitor_result(),
        _monitor_result(_alert(alert_id="alert-2")),
    )
    orchestrator = RecordingOrchestrator()
    loop = AutonomousWatchLoop(monitor=monitor, orchestrator=orchestrator)

    first = loop.run_once(snapshot_id="snapshot-1")
    second = loop.run_once(snapshot_id="snapshot-1")
    recovered = loop.run_once(snapshot_id="snapshot-1")
    recurred = loop.run_once(snapshot_id="snapshot-1")

    assert [record.trigger_type for record in first.records] == ["monitor"]
    assert second.records == []
    assert second.lifecycles[0]["status"] == IntentStatus.DEGRADED.value
    assert recovered.lifecycles[0]["status"] == IntentStatus.SATISFIED.value
    assert [record.trigger_type for record in recurred.records] == ["monitor"]
    assert len(orchestrator.calls) == 2


def test_deferred_alert_runs_on_next_tick_and_does_not_persist() -> None:
    first_alert = _alert("flow-1", alert_id="alert-1")
    second_alert = _alert("flow-2", alert_id="alert-2")
    monitor = SequenceMonitor(
        _monitor_result(first_alert, second_alert),
        _monitor_result(first_alert, second_alert),
    )
    orchestrator = RecordingOrchestrator()
    loop = AutonomousWatchLoop(
        monitor=monitor,
        orchestrator=orchestrator,
        max_monitor_reentries_per_iteration=1,
    )

    first = loop.run_once(snapshot_id="snapshot-1")
    second = loop.run_once(snapshot_id="snapshot-1")

    assert [record.trigger_type for record in first.records] == ["monitor", "monitor_deferred"]
    assert first.pending_monitor_alert_count == 1
    assert [record.trigger_type for record in second.records] == ["monitor"]
    assert second.pending_monitor_alert_count == 0
    assert len(orchestrator.calls) == 2


def test_new_user_intent_skips_concurrent_pre_action_telemetry() -> None:
    inputs = iter(["protect flow-1", ""])
    monitor = SequenceMonitor(_monitor_result(_alert()), _monitor_result(_alert()))
    orchestrator = RecordingOrchestrator()
    loop = AutonomousWatchLoop(
        monitor=monitor,
        orchestrator=orchestrator,
        user_input_source=lambda: next(inputs),
    )

    first = loop.run_once(snapshot_id="snapshot-1")
    second = loop.run_once(snapshot_id="snapshot-1")

    assert [record.trigger_type for record in first.records] == ["user"]
    assert first.lifecycles[0]["status"] == IntentStatus.ACTIVE.value
    assert [record.trigger_type for record in second.records] == ["monitor"]
    assert len(orchestrator.calls) == 2
