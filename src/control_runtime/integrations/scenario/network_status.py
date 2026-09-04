from __future__ import annotations

from typing import Any

from ..storage import get_latest_snapshot_data, get_snapshot_data_by_id


def get_network_status(snapshot_id: str = "") -> dict[str, Any]:
    snapshot = (
        get_snapshot_data_by_id(snapshot_id)
        if snapshot_id.strip()
        else get_latest_snapshot_data()
    )
    flows = snapshot["flows"]
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "run_id": snapshot["run_id"],
        "flow_count": len(flows),
        "allocated_bandwidth_ul_mbps": sum(
            flow["allocation"]["allocated_bandwidth_ul"] for flow in flows
        ),
        "allocated_bandwidth_dl_mbps": sum(
            flow["allocation"]["allocated_bandwidth_dl"] for flow in flows
        ),
        "throughput_ul_mbps": sum(flow["telemetry"]["throughput_ul"] for flow in flows),
        "throughput_dl_mbps": sum(flow["telemetry"]["throughput_dl"] for flow in flows),
        "containers": snapshot["containers"],
        "slice_states": snapshot["slice_states"],
    }


__all__ = ["get_network_status"]
