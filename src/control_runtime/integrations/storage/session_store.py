from __future__ import annotations

from typing import Any

from ..testbed import TestbedClient, TestbedConfig


def _client() -> TestbedClient:
    return TestbedClient.create(TestbedConfig.from_env())


def get_latest_snapshot_data() -> dict[str, Any]:
    return _client().latest_snapshot()


def get_snapshot_data_by_id(snapshot_id: str | int) -> dict[str, Any]:
    normalized = str(snapshot_id).strip()
    if not normalized:
        raise ValueError("snapshot_id is required")
    return _client().snapshot(normalized)


def get_latest_snapshot_metadata() -> dict[str, Any]:
    snapshot = get_latest_snapshot_data()
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "run_id": snapshot["run_id"],
        "observed_from": snapshot["observed_from"],
        "observed_to": snapshot["observed_to"],
        "trigger_event": snapshot["trigger_event"],
    }


__all__ = [
    "get_latest_snapshot_data",
    "get_latest_snapshot_metadata",
    "get_snapshot_data_by_id",
]
