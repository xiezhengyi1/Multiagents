from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from shared.contracts import ExecutionScope
from control_runtime.monitoring.contracts import IntentStatus, MonitorAlert
from control_runtime.monitoring.lifecycle import IntentLifecycleManager


def _alert(*, app_id: str = "app-1") -> MonitorAlert:
    return MonitorAlert(
        alert_id="alert-1",
        snapshot_id="snapshot-2",
        supi="imsi-001",
        app_id=app_id,
        app_name="video",
        flow_id="flow-1",
        flow_name="stream",
        severity="major",
        status="violated",
        violated_metrics=["telemetry.latency"],
        metric_deltas={},
        recommended_consumer="monitor",
        summary="latency violated",
    )


def _result(*, completed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
            round_index=1,
        ),
        completed=completed,
        global_intent={},
        unified_plan={},
        round_traces=(
            [
                {
                    "grounding_decision": {
                        "flows": [
                            {
                                "supi": "imsi-001",
                                "app_id": "app-1",
                                "flow_id": "flow-1",
                            }
                        ]
                    }
                }
            ]
            if completed
            else []
        ),
    )


def test_lifecycle_moves_from_active_to_satisfied_after_successful_replan() -> None:
    manager = IntentLifecycleManager()
    lifecycle = manager.record_initial(_result(completed=True))

    assert lifecycle.status is IntentStatus.ACTIVE

    actions = manager.observe([_alert()])

    assert actions == [(lifecycle, _alert())]
    assert lifecycle.status is IntentStatus.DEGRADED

    manager.begin_replan(lifecycle.intent_id, _alert())
    assert lifecycle.status is IntentStatus.REPLANNING

    manager.finish_replan(lifecycle.intent_id, _result(completed=True))
    assert lifecycle.status is IntentStatus.ACTIVE

    assert manager.observe([]) == []
    assert lifecycle.status is IntentStatus.SATISFIED


def test_nonmatching_alert_creates_monitor_originated_lifecycle() -> None:
    manager = IntentLifecycleManager()
    lifecycle = manager.record_initial(_result(completed=True))

    actions = manager.observe([_alert(app_id="app-2")])

    assert lifecycle.status is IntentStatus.SATISFIED
    monitor_lifecycle, alert = actions[0]
    assert monitor_lifecycle.origin == "monitor"
    assert monitor_lifecycle.bindings[0].app_id == "app-2"
    assert alert.app_id == "app-2"

    manager.begin_replan(monitor_lifecycle.intent_id, alert)
    manager.finish_replan(monitor_lifecycle.intent_id, _result(completed=True))
    manager.observe([])

    repeated_lifecycle, _ = manager.observe([_alert(app_id="app-2")])[0]
    assert repeated_lifecycle is monitor_lifecycle
    assert repeated_lifecycle.status is IntentStatus.DEGRADED


def test_failed_replan_marks_lifecycle_failed() -> None:
    manager = IntentLifecycleManager()
    lifecycle = manager.record_initial(_result(completed=True))
    manager.begin_replan(lifecycle.intent_id, _alert())

    manager.finish_replan(lifecycle.intent_id, _result(completed=False))

    assert lifecycle.status is IntentStatus.FAILED
    assert manager.observe([_alert()]) == []
    assert lifecycle.status is IntentStatus.FAILED
