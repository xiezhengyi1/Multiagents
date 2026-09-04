from __future__ import annotations

from control_runtime.simulation import EnvironmentSnapshot
from shared.contracts.coordination import CoordinationEvent
from shared.runtime.journal import EpisodeJournal

from .contracts import MonitorResult
from .flow_monitor import FlowTelemetryMonitor


class MonitoringCoordinator:
    """Record one deterministic monitoring pass in the episode event graph."""

    actor_id = "monitoring"

    def __init__(
        self,
        *,
        journal: EpisodeJournal,
        monitor: FlowTelemetryMonitor,
    ) -> None:
        self.journal = journal
        self.monitor = monitor

    def evaluate_snapshot_event(
        self, *, snapshot_event_id: str
    ) -> CoordinationEvent[MonitorResult]:
        snapshot_event = self.journal.get_event(snapshot_event_id)
        if not isinstance(snapshot_event.payload, EnvironmentSnapshot):
            raise TypeError("monitoring requires an EnvironmentSnapshot event")
        result = self.monitor.evaluate(snapshot_event.payload)
        return self.journal.append_event(
            round_index=snapshot_event.scope.round_index,
            source_agent=self.actor_id,
            target_agent="episode_controller",
            message_type="observation",
            in_reply_to=snapshot_event.event_id,
            input_artifact_ids=(snapshot_event.event_id,),
            payload=result,
        )


__all__ = ["MonitoringCoordinator"]
