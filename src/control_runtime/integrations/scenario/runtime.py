from __future__ import annotations

import argparse
from typing import Any

from ..storage import get_latest_snapshot_data


def get_initial_scenario() -> dict[str, Any]:
    return get_latest_snapshot_data()


def get_current_scenario() -> dict[str, Any]:
    return get_latest_snapshot_data()


def initialize_scenario() -> dict[str, Any]:
    snapshot = get_latest_snapshot_data()
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "run_id": snapshot["run_id"],
        "flows": len(snapshot["flows"]),
        "slices": len(snapshot["slice_states"]),
        "sessions": len(snapshot["sessions"]),
        "ran_nodes": len(snapshot["ran_nodes"]),
        "core_nodes": len(snapshot["core_nodes"]),
        "mode": "live-testbed",
    }


def init_main() -> None:
    parser = argparse.ArgumentParser(description="Validate the live testbed scenario")
    parser.parse_args()
    print(initialize_scenario())


__all__ = [
    "get_current_scenario",
    "get_initial_scenario",
    "init_main",
    "initialize_scenario",
]
