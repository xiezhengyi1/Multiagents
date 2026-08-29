from __future__ import annotations

import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from .contracts import (
    IntentStatus,
    MonitorAlert,
    MonitorResult,
    RequirementDraft,
    WatchLoopIterationResult,
    WatchLoopTriggerRecord,
)
from .flow_monitor import FlowTelemetryMonitor
from .lifecycle import IntentLifecycleManager


def build_monitor_reentry_requirement(alert: MonitorAlert) -> RequirementDraft:
    flow_label = alert.flow_name or alert.flow_id
    metrics = "、".join(alert.violated_metrics)
    binding_action = "沿用当前业务绑定" if alert.reuse_binding else "重新确认业务绑定"
    user_input = (
        f"监控发现 {flow_label}（SUPI {alert.supi}，app {alert.app_id}，flow {alert.flow_id}）"
        f"违反 SLA：{metrics}。请{binding_action}并重新规划 QoS，使该业务流恢复目标。"
    )
    return RequirementDraft(
        source_alert_id=alert.alert_id,
        snapshot_id=alert.snapshot_id,
        user_input=user_input,
        routing_hint={
            "source": "deterministic_monitor",
            "source_alert_id": alert.alert_id,
            "requested_domains": list(alert.suggested_domains),
            "reuse_binding": alert.reuse_binding,
            "severity": alert.severity,
            "monitor_binding": {
                "supi": alert.supi,
                "app_id": alert.app_id,
                "flow_id": alert.flow_id,
            },
            "violated_metrics": list(alert.violated_metrics),
            "metric_deltas": dict(alert.metric_deltas),
        },
    )


class AutonomousWatchLoop:
    def __init__(
        self,
        *,
        monitor: FlowTelemetryMonitor,
        orchestrator: Any,
        lifecycle_manager: IntentLifecycleManager | None = None,
        user_input_source: Optional[Callable[[], str]] = None,
        snapshot_id_source: Optional[Callable[[], str]] = None,
        poll_interval_seconds: float = 1.0,
        max_monitor_reentries_per_iteration: int = 1,
    ) -> None:
        self.monitor = monitor
        self.orchestrator = orchestrator
        self.lifecycle_manager = lifecycle_manager or IntentLifecycleManager()
        self.user_input_source = user_input_source or (lambda: "")
        self.snapshot_id_source = snapshot_id_source or self._default_snapshot_id_source
        self.poll_interval_seconds = poll_interval_seconds
        self.max_monitor_reentries_per_iteration = max_monitor_reentries_per_iteration

    def run_once(
        self,
        *,
        snapshot_id: str = "",
        scenario_id: str = "",
        scenario_tags: Optional[list[str]] = None,
    ) -> WatchLoopIterationResult:
        resolved_snapshot_id = str(snapshot_id or self.snapshot_id_source() or "").strip()
        if not resolved_snapshot_id:
            raise RuntimeError("failed to resolve snapshot_id for autonomous watch loop")

        with ThreadPoolExecutor(max_workers=2) as executor:
            user_future = executor.submit(self.user_input_source)
            monitor_future = executor.submit(self.monitor.evaluate_snapshot, resolved_snapshot_id)
            raw_user_input = str(user_future.result() or "").strip()
            monitor_result = monitor_future.result()

        records: list[WatchLoopTriggerRecord] = []
        tags = list(scenario_tags or [])
        new_intent_ids: set[str] = set()
        if raw_user_input:
            control_result = self.orchestrator.run(
                raw_user_input,
                scenario_id=scenario_id,
                scenario_tags=tags,
                snapshot_id=resolved_snapshot_id,
            )
            lifecycle = self.lifecycle_manager.record_initial(control_result)
            new_intent_ids.add(lifecycle.intent_id)
            records.append(
                WatchLoopTriggerRecord(
                    trigger_type="user",
                    snapshot_id=resolved_snapshot_id,
                    user_input=raw_user_input,
                    intent_id=lifecycle.intent_id,
                    lifecycle_status=lifecycle.status.value,
                    control_result=control_result,
                    data_flywheel={"stage": "human_requirement_to_orchestrator"},
                )
            )

        monitor_records, pending = self._handle_monitor_alerts(
            monitor_result,
            snapshot_id=resolved_snapshot_id,
            skip_intent_ids=new_intent_ids,
            scenario_id=scenario_id,
            scenario_tags=tags,
        )
        records.extend(monitor_records)
        return WatchLoopIterationResult(
            snapshot_id=resolved_snapshot_id,
            monitor_result=monitor_result,
            records=records,
            user_input_seen=bool(raw_user_input),
            pending_monitor_alerts=pending,
            lifecycles=[lifecycle.to_dict() for lifecycle in self.lifecycle_manager.lifecycles],
        )

    def run_forever(
        self,
        *,
        max_iterations: Optional[int] = None,
        max_retained_results: int = 100,
        **run_once_kwargs: Any,
    ) -> list[WatchLoopIterationResult]:
        results: list[WatchLoopIterationResult] = []
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            results.append(self.run_once(**run_once_kwargs))
            if len(results) > max_retained_results:
                del results[: len(results) - max_retained_results]
            iteration += 1
            if self.poll_interval_seconds:
                time.sleep(self.poll_interval_seconds)
        return results

    def _handle_monitor_alerts(
        self,
        monitor_result: MonitorResult,
        *,
        snapshot_id: str,
        skip_intent_ids: set[str],
        scenario_id: str,
        scenario_tags: list[str],
    ) -> tuple[list[WatchLoopTriggerRecord], list[dict[str, Any]]]:
        actions = self.lifecycle_manager.observe(
            monitor_result.alerts,
            skip_intent_ids=skip_intent_ids,
        )
        records: list[WatchLoopTriggerRecord] = []
        pending: list[dict[str, Any]] = []
        reentry_count = 0
        for lifecycle, alert in actions:
            if lifecycle.status is IntentStatus.FAILED:
                continue
            if reentry_count >= self.max_monitor_reentries_per_iteration:
                pending.append(self._pending_alert_payload(alert, lifecycle.intent_id))
                records.append(
                    WatchLoopTriggerRecord(
                        trigger_type="monitor_deferred",
                        snapshot_id=snapshot_id,
                        user_input="",
                        source_alert_id=alert.alert_id,
                        intent_id=lifecycle.intent_id,
                        lifecycle_status=lifecycle.status.value,
                        data_flywheel={
                            "stage": "monitor_alert_deferred",
                            "reason": "per_iteration_reentry_limit",
                        },
                    )
                )
                continue

            requirement = build_monitor_reentry_requirement(alert)
            self.lifecycle_manager.begin_replan(lifecycle.intent_id, alert)
            control_result = self.orchestrator.run(
                requirement.user_input,
                scenario_id=scenario_id,
                scenario_tags=list(dict.fromkeys([*scenario_tags, "monitor_reentry"])),
                snapshot_id=snapshot_id,
                routing_hint=requirement.routing_hint,
            )
            lifecycle = self.lifecycle_manager.finish_replan(lifecycle.intent_id, control_result)
            reentry_count += 1
            records.append(
                WatchLoopTriggerRecord(
                    trigger_type="monitor",
                    snapshot_id=snapshot_id,
                    user_input=requirement.user_input,
                    source_alert_id=alert.alert_id,
                    intent_id=lifecycle.intent_id,
                    lifecycle_status=lifecycle.status.value,
                    routing_hint=requirement.routing_hint,
                    control_result=control_result,
                    data_flywheel={"stage": "monitor_alert_to_orchestrator_reentry"},
                )
            )
        return records, pending

    @staticmethod
    def _pending_alert_payload(alert: MonitorAlert, intent_id: str) -> dict[str, Any]:
        return {
            "alert_id": alert.alert_id,
            "snapshot_id": alert.snapshot_id,
            "supi": alert.supi,
            "app_id": alert.app_id,
            "flow_id": alert.flow_id,
            "severity": alert.severity,
            "violated_metrics": list(alert.violated_metrics),
            "intent_id": intent_id,
            "defer_reason": "per_iteration_reentry_limit",
        }

    @staticmethod
    def _default_snapshot_id_source() -> str:
        from ..integrations.storage import get_latest_snapshot_metadata

        metadata = get_latest_snapshot_metadata() or {}
        return str(metadata.get("snapshot_id") or "").strip()


def build_default_autonomous_watch_loop(
    *,
    orchestrator: Any,
    snapshot_loader: Any = None,
    user_input_source: Optional[Callable[[], str]] = None,
    snapshot_id_source: Optional[Callable[[], str]] = None,
    tolerance_ratio: float = 0.10,
    max_alerts: int = 20,
    poll_interval_seconds: float = 1.0,
    max_monitor_reentries_per_iteration: int = 1,
) -> AutonomousWatchLoop:
    if snapshot_loader is None:
        from ..integrations.storage import get_snapshot_data_by_id

        snapshot_loader = get_snapshot_data_by_id
    return AutonomousWatchLoop(
        monitor=FlowTelemetryMonitor(
            load_snapshot_by_id=snapshot_loader,
            tolerance_ratio=tolerance_ratio,
            max_alerts=max_alerts,
        ),
        orchestrator=orchestrator,
        user_input_source=user_input_source,
        snapshot_id_source=snapshot_id_source,
        poll_interval_seconds=poll_interval_seconds,
        max_monitor_reentries_per_iteration=max_monitor_reentries_per_iteration,
    )


class ConsoleUserInputSource:
    def __init__(self, *, prompt: str = "") -> None:
        self.prompt = prompt
        self._queue: queue.Queue[str] = queue.Queue()
        self._started = False

    def __call__(self) -> str:
        self._ensure_started()
        try:
            return self._queue.get_nowait().strip()
        except queue.Empty:
            return ""

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._read_forever, daemon=True).start()

    def _read_forever(self) -> None:
        while True:
            if self.prompt and sys.stdin.isatty():
                print(self.prompt, end="", flush=True)
            line = sys.stdin.readline()
            if line == "":
                time.sleep(0.2)
                continue
            text = line.strip()
            if text:
                self._queue.put(text)
