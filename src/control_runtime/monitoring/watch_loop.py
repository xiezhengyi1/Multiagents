from __future__ import annotations

import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from .contracts import MonitorResult, WatchLoopIterationResult, WatchLoopTriggerRecord
from .flow_monitor import FlowTelemetryMonitor
from .requirement_agent import AutonomousRequirementAgent


class PreviousContextBuilder:
    def __init__(self, *, max_chars: int = 4000) -> None:
        self.max_chars = max(200, int(max_chars))

    def build(
        self,
        *,
        previous_result: Any = None,
        previous_user_intent: str = "",
        prior_context: str = "",
    ) -> tuple[str, bool]:
        payload = {
            "previous_user_intent": str(previous_user_intent or "").strip(),
            "prior_context": str(prior_context or "").strip(),
            "previous_result": self._summarize_result(previous_result),
        }
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= self.max_chars:
            return text, False

        result_summary = payload["previous_result"] if isinstance(payload["previous_result"], dict) else {}
        head = json.dumps(
            {
                "previous_user_intent": payload["previous_user_intent"],
                "context_policy": "truncated_to_recent_and_operationally_relevant_fields",
                "session_id": result_summary.get("session_id", ""),
                "snapshot_id": result_summary.get("snapshot_id", ""),
                "completed": result_summary.get("completed", ""),
                "diagnosis": result_summary.get("diagnosis", {}),
                "execution_reentry": result_summary.get("execution_reentry", {}),
            },
            ensure_ascii=False,
            default=str,
        )
        remaining = max(0, self.max_chars - len(head) - 44)
        tail = text[-remaining:] if remaining else ""
        compacted = f"{head}\n[truncated_previous_context_tail]\n{tail}"
        return compacted[: self.max_chars + 40], True

    @staticmethod
    def _summarize_result(previous_result: Any) -> dict[str, Any]:
        if previous_result is None:
            return {}
        if isinstance(previous_result, dict):
            raw = previous_result
        else:
            raw = {
                "session_id": getattr(previous_result, "session_id", ""),
                "snapshot_id": getattr(previous_result, "snapshot_id", ""),
                "completed": getattr(previous_result, "completed", None),
                "global_intent": getattr(previous_result, "global_intent", {}),
                "diagnosis": getattr(previous_result, "diagnosis", {}),
                "execution_reentry": getattr(previous_result, "execution_reentry", {}),
                "planning_blocker": getattr(previous_result, "planning_blocker", {}),
                "qos_feedback": getattr(previous_result, "qos_feedback", {}),
                "mobility_feedback": getattr(previous_result, "mobility_feedback", {}),
                "round_count": getattr(previous_result, "round_count", 0),
                "retry_count": getattr(previous_result, "retry_count", 0),
                "round_traces": getattr(previous_result, "round_traces", []),
            }
        round_traces = raw.get("round_traces") if isinstance(raw.get("round_traces"), list) else []
        return {
            "session_id": raw.get("session_id", ""),
            "snapshot_id": raw.get("snapshot_id", ""),
            "completed": raw.get("completed", None),
            "global_intent": raw.get("global_intent", {}) or {},
            "diagnosis": raw.get("diagnosis", {}) or {},
            "execution_reentry": raw.get("execution_reentry", {}) or {},
            "planning_blocker": raw.get("planning_blocker", {}) or {},
            "qos_feedback": raw.get("qos_feedback", {}) or {},
            "mobility_feedback": raw.get("mobility_feedback", {}) or {},
            "round_count": raw.get("round_count", 0),
            "retry_count": raw.get("retry_count", 0),
            "recent_round_traces": round_traces[-2:],
        }


class AutonomousWatchLoop:
    def __init__(
        self,
        *,
        monitor: FlowTelemetryMonitor,
        orchestrator: Any,
        requirement_agent: Optional[AutonomousRequirementAgent] = None,
        user_input_source: Optional[Callable[[], str]] = None,
        snapshot_id_source: Optional[Callable[[], str]] = None,
        previous_context_builder: Optional[PreviousContextBuilder] = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.monitor = monitor
        self.orchestrator = orchestrator
        self.requirement_agent = requirement_agent or AutonomousRequirementAgent()
        self.user_input_source = user_input_source or (lambda: "")
        self.snapshot_id_source = snapshot_id_source or self._default_snapshot_id_source
        self.previous_context_builder = previous_context_builder or PreviousContextBuilder()
        self.poll_interval_seconds = max(0.0, float(poll_interval_seconds))

    def run_once(
        self,
        *,
        snapshot_id: str = "",
        previous_result: Any = None,
        previous_user_intent: str = "",
        prior_context: str = "",
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
        if raw_user_input:
            control_result = self.orchestrator.run(
                raw_user_input,
                scenario_id=scenario_id,
                scenario_tags=tags,
                snapshot_id=resolved_snapshot_id,
            )
            records.append(
                WatchLoopTriggerRecord(
                    trigger_type="user",
                    snapshot_id=resolved_snapshot_id,
                    user_input=raw_user_input,
                    control_result=control_result,
                    data_flywheel={
                        "stage": "human_requirement_to_orchestrator",
                        "source": "user_input_source",
                    },
                )
            )

        records.extend(
            self._handle_monitor_alerts(
                monitor_result,
                snapshot_id=resolved_snapshot_id,
                previous_result=previous_result,
                previous_user_intent=previous_user_intent or raw_user_input,
                prior_context=prior_context,
                scenario_id=scenario_id,
                scenario_tags=tags,
            )
        )
        return WatchLoopIterationResult(
            snapshot_id=resolved_snapshot_id,
            monitor_result=monitor_result,
            records=records,
            user_input_seen=bool(raw_user_input),
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
            result = self.run_once(**run_once_kwargs)
            results.append(result)
            if len(results) > max(1, int(max_retained_results)):
                del results[: len(results) - max(1, int(max_retained_results))]
            iteration += 1
            if self.poll_interval_seconds:
                time.sleep(self.poll_interval_seconds)
        return results

    def _handle_monitor_alerts(
        self,
        monitor_result: MonitorResult,
        *,
        snapshot_id: str,
        previous_result: Any,
        previous_user_intent: str,
        prior_context: str,
        scenario_id: str,
        scenario_tags: list[str],
    ) -> list[WatchLoopTriggerRecord]:
        if not monitor_result.alerts:
            return []

        previous_context, context_truncated = self.previous_context_builder.build(
            previous_result=previous_result,
            previous_user_intent=previous_user_intent,
            prior_context=prior_context,
        )
        records: list[WatchLoopTriggerRecord] = []
        for alert in monitor_result.alerts:
            requirement = self.requirement_agent.generate_requirement(
                alert,
                previous_user_intent=previous_user_intent,
                extra_context={
                    "monitor_summary": monitor_result.summary,
                    "previous_control_context": previous_context,
                    "previous_context_truncated": context_truncated,
                    "context_strategy": "bounded_recent_context_plus_retrievable_memory",
                },
            )
            control_result = self.orchestrator.run(
                requirement.user_input,
                scenario_id=scenario_id,
                scenario_tags=self._merge_tags(scenario_tags, ["monitor_reentry", "data_flywheel"]),
                snapshot_id=snapshot_id,
            )
            records.append(
                WatchLoopTriggerRecord(
                    trigger_type="monitor",
                    snapshot_id=snapshot_id,
                    user_input=requirement.user_input,
                    source_alert_id=alert.alert_id,
                    previous_context=previous_context,
                    context_truncated=context_truncated,
                    control_result=control_result,
                    data_flywheel={
                        "stage": "monitor_alert_to_orchestrator_reentry",
                        "alert_id": alert.alert_id,
                        "flow_id": alert.flow_id,
                        "synthetic_requirement": True,
                        "llm_output_contract": "natural_language",
                    },
                )
            )
        return records

    @staticmethod
    def _merge_tags(base: list[str], additions: list[str]) -> list[str]:
        merged: list[str] = []
        for tag in [*base, *additions]:
            normalized = str(tag or "").strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    @staticmethod
    def _default_snapshot_id_source() -> str:
        from ..integrations.storage import get_latest_snapshot_metadata

        metadata = get_latest_snapshot_metadata() or {}
        return str(metadata.get("snapshot_id") or "").strip()


def build_default_autonomous_watch_loop(
    *,
    orchestrator: Any = None,
    snapshot_loader: Any = None,
    requirement_llm: Any = None,
    user_input_source: Optional[Callable[[], str]] = None,
    snapshot_id_source: Optional[Callable[[], str]] = None,
    tolerance_ratio: float = 0.10,
    max_alerts: int = 20,
    previous_context_max_chars: int = 4000,
    poll_interval_seconds: float = 1.0,
) -> AutonomousWatchLoop:
    if snapshot_loader is None:
        from ..integrations.storage import get_snapshot_data_by_id

        snapshot_loader = get_snapshot_data_by_id
    if orchestrator is None:
        from ..orchestrators.main_control_orchestrator import MainControlOrchestrator

        orchestrator = MainControlOrchestrator()
    return AutonomousWatchLoop(
        monitor=FlowTelemetryMonitor(
            load_snapshot_by_id=snapshot_loader,
            tolerance_ratio=tolerance_ratio,
            max_alerts=max_alerts,
        ),
        orchestrator=orchestrator,
        requirement_agent=AutonomousRequirementAgent(llm=requirement_llm),
        user_input_source=user_input_source,
        snapshot_id_source=snapshot_id_source,
        previous_context_builder=PreviousContextBuilder(max_chars=previous_context_max_chars),
        poll_interval_seconds=poll_interval_seconds,
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
        thread = threading.Thread(target=self._read_forever, daemon=True)
        thread.start()

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
