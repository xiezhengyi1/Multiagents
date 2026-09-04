from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from control_runtime.monitoring import (
    FlowTelemetryMonitor,
    MonitoringCoordinator,
    MonitorResult,
    MonitorRule,
)
from control_runtime.simulation import (
    EnvironmentSnapshot,
    SnapshotEntity,
    SnapshotMetric,
)
from shared.runtime.journal import EpisodeJournal


def test_monitoring_emits_a_coordination_observation(tmp_path: Path) -> None:
    observed_at = datetime.now(UTC)
    snapshot = EnvironmentSnapshot(
        snapshot_id="snapshot-1",
        observed_at=observed_at,
        entities=[
            SnapshotEntity(
                entity_id="flow-1",
                entity_type="flow",
                attributes={"app_id": "app-video"},
            )
        ],
        relations=[],
        metrics=[
            SnapshotMetric(
                owner_entity_id="flow-1",
                metric_name="latency_ms",
                value=35.0,
                observed_at=observed_at,
            )
        ],
    )
    journal = EpisodeJournal(root=tmp_path, episode_id="monitoring-1")
    snapshot_event = journal.append_event(
        round_index=0,
        source_agent="simulator",
        target_agent="monitoring",
        message_type="observation",
        payload=snapshot,
    )
    coordinator = MonitoringCoordinator(
        journal=journal,
        monitor=FlowTelemetryMonitor(
            rules=[
                MonitorRule(
                    owner_entity_id="flow-1",
                    metric_name="latency_ms",
                    relation="above",
                    threshold=20.0,
                    severity="critical",
                    suggested_domains=["qos"],
                )
            ]
        ),
    )

    event = coordinator.evaluate_snapshot_event(
        snapshot_event_id=snapshot_event.event_id
    )

    assert event.source_agent == "monitoring"
    assert event.target_agent == "episode_controller"
    assert event.message_type == "observation"
    assert event.in_reply_to == snapshot_event.event_id
    assert isinstance(event.payload, MonitorResult)
    assert event.payload.alerts[0].owner_entity_id == "flow-1"
    assert event.payload.alerts[0].observed_value == 35.0
