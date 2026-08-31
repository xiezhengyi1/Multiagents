from control_runtime.agents.single.tools import compact_flow_catalog_for_single_agent
from control_runtime.integrations.pcf.read_tools import _project_sm_flow_catalog_for_agent
from control_runtime.integrations.scenario.network_graph import NetworkGraph


def _catalog() -> dict:
    return {
        "supi": "imsi-208930000000001",
        "app_catalog": [],
        "flow_catalog": [
            {
                "supi": "imsi-208930000000001",
                "app_id": "mec-inference",
                "flow_id": "flow-1",
                "flow_name": "inference",
                "service": {"service_type": "urllc", "service_type_id": 1},
                "sla": {"latency": 20.0, "loss_rate": 0.001},
                "allocation": {"current_slice_snssai": "01000001"},
                "telemetry": {
                    "throughput_ul": 18.5,
                    "latency": 12.0,
                    "jitter": 1.2,
                    "loss_rate": 0.0005,
                    "tcp_retransmits_per_second": 0.2,
                    "tcp_rtt_ms_average": 11.0,
                    "observed_at": "2026-08-30T10:00:00Z",
                },
                "ran": {
                    "serving_cell_id": "cell-1",
                    "qfi": 7,
                    "mcs_ul": 18,
                    "bler_ul": 0.03,
                    "prb_ul": 12,
                    "sinr_db": 17.5,
                },
                "session": {
                    "pdu_session_id": 1,
                    "dnn": "mec",
                    "snssai": "01000001",
                    "qfi": 7,
                    "five_qi": 2,
                    "upf_id": "upf-edge",
                },
                "path": {
                    "site_id": "edge-a",
                    "service_instance_id": "mec-app-1",
                    "upf_path": ["upf-edge"],
                },
                "data_quality": {
                    "window_start": "2026-08-30T09:59:50Z",
                    "window_end": "2026-08-30T10:00:00Z",
                    "sample_count": 10,
                    "stale": False,
                    "correlation_status": "complete",
                },
            }
        ],
    }


def test_multi_agent_flow_catalog_keeps_runtime_metric_evidence() -> None:
    projected = _project_sm_flow_catalog_for_agent(_catalog())
    flow = projected["flow_catalog"][0]

    assert flow["telemetry"]["latency"] == 12.0
    assert flow["telemetry"]["tcp_rtt_ms_average"] == 11.0
    assert flow["ran"]["mcs_ul"] == 18
    assert flow["session"]["qfi"] == 7
    assert flow["path"]["site_id"] == "edge-a"
    assert flow["data_quality"]["correlation_status"] == "complete"


def test_single_agent_flow_catalog_uses_the_same_runtime_metric_contract() -> None:
    projected = compact_flow_catalog_for_single_agent(_catalog())
    flow = projected["flow_catalog"][0]

    assert flow["telemetry"]["tcp_retransmits_per_second"] == 0.2
    assert flow["ran"]["sinr_db"] == 17.5
    assert flow["session"]["upf_id"] == "upf-edge"
    assert flow["data_quality"]["sample_count"] == 10


def test_scenario_snapshot_preserves_graph_metric_series() -> None:
    graph = NetworkGraph(snapshot_id="snapshot-1", trigger_event="periodic")
    graph.add_node(
        "flow:imsi-208930000000001:mec-inference:flow-1",
        "flow",
        properties=_catalog()["flow_catalog"][0],
    )
    graph.upsert_metric(
        "node",
        "flow:imsi-208930000000001:mec-inference:flow-1",
        "telemetry.latency",
        12.0,
        observed_at="2026-08-30T10:00:00Z",
    )

    snapshot = graph.to_scenario_snapshot()

    assert snapshot["metrics"][0]["metric_name"] == "telemetry.latency"
    assert snapshot["metrics"][0]["metric_value"] == 12.0
