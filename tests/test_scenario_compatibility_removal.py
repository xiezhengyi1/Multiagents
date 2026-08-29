from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://offline:offline@127.0.0.1:5432/offline_test",
)

from control_runtime.integrations.scenario.common import _build_slice_from_dict, _resolve_slice_current_bandwidth
from control_runtime.integrations.scenario.yaml_loader import _build_apps_from_yaml, _create_scenario_from_yaml


def test_slice_deserialization_uses_canonical_nested_load_and_telemetry() -> None:
    slice_obj = _build_slice_from_dict(
        {
            "name": "slice-a",
            "sst": 1,
            "sd": "000001",
            "capacity": {"total_bandwidth_ul": 100.0, "total_bandwidth_dl": 200.0},
            "load": {"current_bandwidth_ul": 12.0},
            "telemetry": {"utilization_dl": 0.25},
        }
    )

    assert slice_obj.load.current_bandwidth_ul == 12.0
    assert slice_obj.load.current_bandwidth_dl == 50.0
    assert slice_obj.telemetry.utilization_ul is None
    assert slice_obj.telemetry.utilization_dl == 0.25


def test_slice_deserialization_rejects_removed_legacy_load_and_utilization_fields() -> None:
    slice_obj = _build_slice_from_dict(
        {
            "name": "slice-a",
            "sst": 1,
            "sd": "000001",
            "capacity": {"total_bandwidth_ul": 100.0, "total_bandwidth_dl": 200.0},
            "current_load_bw_ul": 12.0,
            "sim_utilization_dl": 0.25,
        }
    )

    assert "legacy_load_value" not in inspect.signature(_resolve_slice_current_bandwidth).parameters
    assert "legacy_utilization_value" not in inspect.signature(_resolve_slice_current_bandwidth).parameters
    assert slice_obj.load.current_bandwidth_ul == 0.0
    assert slice_obj.load.current_bandwidth_dl == 0.0
    assert slice_obj.telemetry.utilization_dl is None


@pytest.mark.parametrize(
    "scenario_name",
    (
        "s1_basic_single_slice.yaml",
        "s2_medium_complexity.yaml",
        "s3_high_complexity.yaml",
    ),
)
def test_all_builtin_scenarios_load_through_the_canonical_loader(scenario_name: str) -> None:
    apps, slices, nodes = _create_scenario_from_yaml(ROOT / "experiments" / "scenarios" / scenario_name)

    assert apps
    assert slices
    assert nodes
    assert all(app.supi for app in apps)
    assert all(flow.id for app in apps for flow in app.flows)


def test_loader_requires_every_flow_to_reference_a_declared_app() -> None:
    with pytest.raises(ValueError, match="flows reference app_id values that are not declared in apps: missing-app"):
        _build_apps_from_yaml(
            apps_payload=[],
            flows_payload=[{"flow_id": "flow-1", "app_id": "missing-app", "supi": "imsi-001"}],
            ues_payload=[],
            label_to_snssai={},
        )


def test_loader_uses_declared_app_identity_without_flow_supi() -> None:
    apps = _build_apps_from_yaml(
        apps_payload=[
            {
                "app_id": "app-0001",
                "name": "video",
                "supi": "imsi-001",
                "flow_ids": ["flow-0001"],
            }
        ],
        flows_payload=[
            {
                "flow_id": "flow-0001",
                "app_id": "app-0001",
                "name": "video-stream",
                "service_type": "eMBB",
                "service_type_id": 1,
                "slice_ref": "slice-a",
                "allocated_bandwidth_ul_mbps": 3.0,
                "allocated_bandwidth_dl_mbps": 4.0,
                "sla_target": {
                    "bandwidth_ul_mbps": 3.0,
                    "bandwidth_dl_mbps": 4.0,
                    "guaranteed_bandwidth_ul_mbps": 2.0,
                    "guaranteed_bandwidth_dl_mbps": 3.0,
                    "latency_ms": 10.0,
                    "jitter_ms": 1.0,
                    "loss_rate": 0.001,
                    "priority": 1,
                },
            }
        ],
        ues_payload=[],
        label_to_snssai={"slice-a": "01000001"},
    )

    assert [(app.id, app.supi, [flow.id for flow in app.flows]) for app in apps] == [
        ("app-0001", "imsi-001", ["flow-0001"])
    ]
