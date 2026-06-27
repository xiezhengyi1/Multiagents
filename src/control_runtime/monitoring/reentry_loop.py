from __future__ import annotations

from typing import Any, Optional

from .contracts import ReentryLoopResult
from .flow_monitor import FlowTelemetryMonitor
from .requirement_agent import AutonomousRequirementAgent


class AutonomousMonitorReentryLoop:
    def __init__(
        self,
        *,
        monitor: FlowTelemetryMonitor,
        orchestrator: Any,
        requirement_agent: Optional[AutonomousRequirementAgent] = None,
    ) -> None:
        self.monitor = monitor
        self.orchestrator = orchestrator
        self.requirement_agent = requirement_agent or AutonomousRequirementAgent()

    def run_once(
        self,
        *,
        snapshot_id: str,
        previous_user_intent: str = "",
        scenario_id: str = "",
        scenario_tags: Optional[list[str]] = None,
    ) -> ReentryLoopResult:
        monitor_result = self.monitor.evaluate_snapshot(snapshot_id)
        if not monitor_result.alerts:
            return ReentryLoopResult(
                snapshot_id=snapshot_id,
                monitor_result=monitor_result,
                reentry_triggered=False,
            )

        alert = monitor_result.alerts[0]
        requirement = self.requirement_agent.generate_requirement(
            alert,
            previous_user_intent=previous_user_intent,
            extra_context={"monitor_summary": monitor_result.summary},
        )
        control_result = self.orchestrator.run(
            requirement.user_input,
            scenario_id=scenario_id,
            scenario_tags=list(scenario_tags or []),
            snapshot_id=snapshot_id,
            routing_hint=requirement.routing_hint,
        )
        return ReentryLoopResult(
            snapshot_id=snapshot_id,
            monitor_result=monitor_result,
            reentry_triggered=True,
            requirement=requirement,
            control_result=control_result,
        )


def build_default_monitor_reentry_loop(
    *,
    orchestrator: Any = None,
    snapshot_loader: Any = None,
    requirement_llm: Any = None,
    tolerance_ratio: float = 0.10,
    max_alerts: int = 20,
) -> AutonomousMonitorReentryLoop:
    if snapshot_loader is None:
        from ..integrations.storage import get_snapshot_data_by_id

        snapshot_loader = get_snapshot_data_by_id
    if orchestrator is None:
        from ..orchestrators.main_control_orchestrator import MainControlOrchestrator

        orchestrator = MainControlOrchestrator()
    return AutonomousMonitorReentryLoop(
        monitor=FlowTelemetryMonitor(
            load_snapshot_by_id=snapshot_loader,
            tolerance_ratio=tolerance_ratio,
            max_alerts=max_alerts,
        ),
        requirement_agent=AutonomousRequirementAgent(llm=requirement_llm),
        orchestrator=orchestrator,
    )
