from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from control_runtime.integrations.optimizer.interface import (
    _load_optimizer_scenario,
    optimize_network_slices,
)
from control_runtime.integrations.pcf import dispatch_policy_to_pcf_request
from control_runtime.integrations.scenario import get_network_status
from control_runtime.integrations.storage import (
    get_latest_snapshot_data,
    get_ue_context_by_supi,
    search_flow_targets_by_semantic,
)
from control_runtime.integrations.testbed import (
    NativeSnapshot,
)
from control_runtime.integrations.testbed import (
    TestbedClient as LiveTestbedClient,
)
from control_runtime.integrations.testbed import (
    TestbedConfig as LiveTestbedConfig,
)
from control_runtime.integrations.testbed import (
    TestbedSimulatorAdapter as LiveTestbedAdapter,
)
from control_runtime.integrations.testbed.projection import project_environment_snapshot
from control_runtime.simulation import (
    ExecutionAction,
    ExecutionOperation,
    ObservationRequest,
)
from model.SmPolicyDecision import PCCRule, QosData, SmPolicyDecision


def _snapshot(snapshot_id: str = "snapshot-1") -> dict[str, Any]:
    flow = {
        "id": "ue-1-flow",
        "name": "ue-1-flow",
        "flow_id": "ue-1-flow",
        "supi": "imsi-208930000000001",
        "app_id": "video-app",
        "app_name": "video-app",
        "sla": {
            "latency": 20.0,
            "jitter": 3.0,
            "loss_rate": 0.001,
            "bandwidth_ul": 8.0,
            "bandwidth_dl": 10.0,
            "guaranteed_bandwidth_ul": 5.0,
            "guaranteed_bandwidth_dl": 6.0,
            "priority": 2,
        },
        "allocation": {
            "current_slice_snssai": "01010203",
            "allocated_bandwidth_ul": 8.0,
            "allocated_bandwidth_dl": 10.0,
            "source_action_id": None,
        },
        "traffic": {
            "five_tuple": ["10.60.0.1", "172.34.0.100", 41001, 5202, "udp"],
            "packet_size": 9600.0,
            "packet_size_unit": "bits",
            "packet_size_bytes": 1200,
            "arrival_rate": 2083.3333333333335,
            "arrival_rate_unit": "packets_per_second",
            "offered_rate_mbps": 20.0,
        },
        "telemetry": {
            "throughput_ul": 4.5,
            "throughput_dl": 6.5,
            "latency": 18.0,
            "jitter": 1.0,
            "loss_rate": 0.0,
            "packet_sent": 100.0,
            "packet_received": 100.0,
            "tcp_retransmits": 0.0,
            "observed_at": "2026-09-01T00:00:10Z",
        },
        "ran": {
            "serving_cell_id": "cell-1",
            "gnb_id": "gnb-1",
            "mcs_ul": 28.0,
            "mcs_dl": 27.0,
            "bler_ul": 0.0,
            "bler_dl": 0.01,
            "prb_ul": 20.0,
            "rnti": "0x1001",
            "drb_id": 1,
        },
        "session": {
            "pdu_session_id": 1,
            "dnn": "embb",
            "snssai": "01010203",
            "qfi": 1,
            "five_qi": 9,
            "upf_id": "upf-1",
            "ue_ip": "10.60.0.1",
            "ul_teid": "0x1",
            "dl_teid": "0x2",
            "sm_policy_uri": None,
        },
        "path": {
            "site_id": "edge-a",
            "service_instance_id": "service-1",
            "upf_path": ["upf-1"],
            "app_endpoint": "172.34.0.100:8080",
        },
        "data_quality": {
            "window_start": "2026-09-01T00:00:00Z",
            "window_end": "2026-09-01T00:00:10Z",
            "sample_count": 5,
            "stale": False,
            "correlation_status": "complete",
        },
    }
    return {
        "snapshot_id": snapshot_id,
        "run_id": "run-1",
        "observed_from": "2026-09-01T00:00:00Z",
        "observed_to": "2026-09-01T00:00:10Z",
        "topology": {
            "nodes": [
                {
                    "id": "slice-1",
                    "type": "slice",
                    "attributes": {"sst": 1, "sd": "010203"},
                },
                {"id": "edge-a", "type": "site", "attributes": {}},
                {"id": "gnb-1", "type": "gnb", "attributes": {}},
                {"id": "cell-1", "type": "cell", "attributes": {}},
                {"id": "upf-1", "type": "upf", "attributes": {}},
                {"id": "service-1", "type": "service_instance", "attributes": {}},
                {
                    "id": "ue-1",
                    "type": "ue",
                    "attributes": {"supi": "imsi-208930000000001"},
                },
                {
                    "id": "ue-1-flow",
                    "type": "flow",
                    "attributes": {"protocol": "udp", "session_id": "session-1"},
                },
            ],
            "edges": [
                {"source": "ue-1", "target": "gnb-1", "type": "attached_to"},
                {"source": "ue-1", "target": "ue-1-flow", "type": "originates"},
                {"source": "slice-1", "target": "ue-1-flow", "type": "carries"},
            ],
        },
        "flows": [flow],
        "slice_states": [
            {
                "slice_id": "slice-1",
                "snssai": "01010203",
                "phase": "running",
                "sequence": 1,
                "observed_at": "2026-09-01T00:00:10Z",
                "upf_containers": ["upf-1"],
                "ue_ids": ["ue-1"],
                "flow_ids": ["ue-1-flow"],
                "capacity": {
                    "total_bandwidth_ul": 20.0,
                    "total_bandwidth_dl": 80.0,
                    "guaranteed_bandwidth_ul": 10.0,
                    "guaranteed_bandwidth_dl": 40.0,
                },
                "capacity_unit": "Mbps",
                "load": {
                    "current_bandwidth_ul": 8.0,
                    "current_bandwidth_dl": 10.0,
                },
                "load_unit": "Mbps",
                "qos": {
                    "latency": 20.0,
                    "processing_delay": 0.0,
                    "jitter": 3.0,
                    "loss_rate": 0.001,
                },
                "qos_source": "declared_flow_sla",
                "telemetry": {
                    "utilization_ul": 0.4,
                    "utilization_dl": 0.125,
                    "latency": 18.0,
                    "jitter": 1.0,
                    "loss_rate": 0.0,
                },
                "telemetry_source": "flow_aggregate_worst_case",
            }
        ],
        "sessions": [
            {
                "supi": "imsi-208930000000001",
                "dnn": "embb",
                "pdu_session_id": 1,
                "drb_id": 1,
                "qfi": 1,
                "five_qi": 9,
                "ul_teid": "0x1",
                "dl_teid": "0x2",
                "ue_id": "ue-1",
                "session_id": "session-1",
                "slice_id": "slice-1",
                "ue_ip": "10.60.0.1",
                "rnti": "0x1001",
                "cu_ue_id": 1,
                "correlation_status": "complete",
            }
        ],
        "ran_nodes": [
            {
                "id": "gnb-1",
                "name": "gnb-1",
                "node_type": "AN",
                "capacity": {
                    "cpu": 3.0,
                    "memory": 8589934592.0,
                    "mec": 0.0,
                    "prb": 106.0,
                },
                "capacity_units": {
                    "cpu": "cores",
                    "memory": "bytes",
                    "mec": "not_applicable",
                    "prb": "physical_resource_blocks",
                },
                "hosted_slice_snssais": ["01010203"],
                "telemetry": {
                    "cpu_utilization": 0.5,
                    "memory_utilization": 0.1,
                    "mec_utilization": 0.0,
                    "prb_utilization": 0.2,
                    "ues": [
                        {
                            "ue_id": "ue-1",
                            "supi": "imsi-208930000000001",
                            "rnti": "0x1001",
                            "mcs_ul": 28.0,
                        }
                    ],
                },
            }
        ],
        "core_nodes": [
            {
                "id": "upf-1",
                "name": "upf-1",
                "node_type": "CN",
                "capacity": {
                    "cpu": 2.0,
                    "memory": 1073741824.0,
                    "mec": 0.0,
                    "prb": 0.0,
                },
                "capacity_units": {
                    "cpu": "cores",
                    "memory": "bytes",
                    "mec": "not_applicable",
                    "prb": "physical_resource_blocks",
                },
                "hosted_slice_snssais": ["01010203"],
                "telemetry": {
                    "cpu_utilization": 0.1,
                    "memory_utilization": 0.1,
                    "mec_utilization": 0.0,
                    "prb_utilization": 0.0,
                },
                "dnn": "embb",
            }
        ],
        "containers": [
            {
                "id": "upf-1",
                "cpu_cores": 0.1,
                "cpu_limit_cores": 2.0,
                "memory_working_set_bytes": 1000.0,
                "memory_limit_bytes": 2000.0,
            }
        ],
        "mec_sites": [{"id": "edge-a", "type": "edge"}],
        "mobility": [],
        "active_actions": [],
        "trigger_event": "periodic",
    }


def _subscriber() -> dict[str, Any]:
    return {
        "AuthenticationSubscription": {
            "permanentKey": {"permanentKeyValue": "must-not-leak"}
        },
        "AccessAndMobilitySubscriptionData": {
            "subscribedUeAmbr": {"uplink": "1 Gbps", "downlink": "1 Gbps"},
            "nssai": {
                "defaultSingleNssais": [{"sst": 1, "sd": "010203"}],
                "singleNssais": [{"sst": 1, "sd": "010203"}],
            },
        },
        "SessionManagementSubscriptionData": [
            {
                "singleNssai": {"sst": 1, "sd": "010203"},
                "dnnConfigurations": {"embb": {"sessionAmbr": {}}},
            }
        ],
    }


class _Client:
    def __init__(self) -> None:
        self.config = SimpleNamespace(run_id="run-1", plmn_id="20893")
        self.native = _snapshot()
        self.dispatched: list[dict[str, Any]] = []

    def latest_snapshot(self) -> dict[str, Any]:
        return self.native

    def snapshot(self, snapshot_id: str) -> dict[str, Any]:
        self.native = _snapshot(snapshot_id)
        return self.native

    def subscriber(self, supi: str) -> dict[str, Any]:
        assert supi == "imsi-208930000000001"
        return _subscriber()

    def dispatch_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.dispatched.append(payload)
        return {
            "action_id": "action-1",
            "run_id": "run-1",
            "snapshot_before": payload["snapshot_id"],
            "status": "verified",
            "dispatch": {
                "accepted_at": "2026-09-01T00:00:10Z",
                "completed_at": "2026-09-01T00:00:11Z",
            },
            "observed_effect": {"updated": True},
            "snapshot_after": "snapshot-2",
        }


def test_testbed_adapter_projects_live_graph_and_all_observation_contracts() -> None:
    adapter = LiveTestbedAdapter(client=_Client())
    snapshot = adapter.capture_snapshot()

    assert snapshot.snapshot_id == "snapshot-1"
    assert {entity.entity_type for entity in snapshot.entities} == {
        "app",
        "core_node",
        "flow",
        "ran_node",
        "session",
        "slice",
        "ue",
    }
    assert any(
        metric.owner_entity_id == "ue-1-flow"
        and metric.metric_name == "telemetry.throughput_dl"
        and metric.value == 6.5
        for metric in snapshot.metrics
    )

    supi = "imsi-208930000000001"
    requests = (
        ObservationRequest(
            query_name="get_sm_ue_flow_catalog", arguments={"supi": supi}
        ),
        ObservationRequest(
            query_name="search_sm_flow_targets",
            arguments={"supi": supi, "query": "video"},
        ),
        ObservationRequest(
            query_name="get_am_policy_context", arguments={"supi": supi}
        ),
        ObservationRequest(
            query_name="search_am_policy_targets",
            arguments={"supi": supi, "query": "01010203"},
        ),
        ObservationRequest(
            query_name="get_ue_slice_subscription", arguments={"supi": supi}
        ),
        ObservationRequest(
            query_name="preview_qos_optimizer",
            arguments={"supi": supi, "flow_ids": ["ue-1-flow"]},
        ),
        ObservationRequest(
            query_name="inspect_mobility_ue_policies", arguments={"supi": supi}
        ),
    )
    results = {
        request.query_name: adapter.observe(request).result for request in requests
    }

    assert len(results["get_sm_ue_flow_catalog"]["flow_catalog"]) == 1
    assert len(results["search_sm_flow_targets"]["matches"]) == 1
    assert results["get_ue_slice_subscription"]["allowed_snssais"] == ["01010203"]
    assert "must-not-leak" not in repr(results)
    assert results["get_am_policy_context"]["association_id"] is None
    assert results["inspect_mobility_ue_policies"]["policies"] == []
    assert results["preview_qos_optimizer"]["ready"] is True
    assert results["preview_qos_optimizer"]["blocked_flows"] == []
    assert results["preview_qos_optimizer"]["qos_flow_assignments"] == [
        {
            "flow_id": "ue-1-flow",
            "app_id": "video-app",
            "bw_ul": 8.0,
            "bw_dl": 10.0,
            "source": "snapshot.flow.sla",
        }
    ]
    traffic = results["get_sm_ue_flow_catalog"]["flow_catalog"][0]["traffic"]
    assert traffic["five_tuple"] == [
        "10.60.0.1",
        "172.34.0.100",
        41001,
        5202,
        "udp",
    ]
    assert traffic["packet_size"] == 9600.0
    assert traffic["arrival_rate"] == 2083.3333333333335


def test_testbed_flow_search_normalizes_legacy_name_separators() -> None:
    native = _snapshot()
    native["flows"][0]["id"] = "remote-drive-video-1"
    native["flows"][0]["name"] = "remote-drive-video-1"
    native["flows"][0]["flow_id"] = "remote-drive-video-1"
    flow_node = next(
        node for node in native["topology"]["nodes"] if node["type"] == "flow"
    )
    flow_node["id"] = "remote-drive-video-1"
    for edge in native["topology"]["edges"]:
        if edge["source"] == "ue-1-flow":
            edge["source"] = "remote-drive-video-1"
        if edge["target"] == "ue-1-flow":
            edge["target"] = "remote-drive-video-1"
    client = _Client()
    client.native = native
    adapter = LiveTestbedAdapter(client=client)
    adapter.capture_snapshot()

    result = adapter.observe(
        ObservationRequest(
            query_name="search_sm_flow_targets",
            arguments={
                "supi": "imsi-208930000000001",
                "query": "Remote_Drive_video_1",
            },
        )
    ).result

    assert [flow["id"] for flow in result["matches"]] == ["remote-drive-video-1"]


def test_testbed_flow_search_accepts_legacy_token_order() -> None:
    native = _snapshot()
    native["flows"][0]["id"] = "video-4k-stream-1"
    native["flows"][0]["name"] = "video-4k-stream-1"
    native["flows"][0]["flow_id"] = "video-4k-stream-1"
    flow_node = next(
        node for node in native["topology"]["nodes"] if node["type"] == "flow"
    )
    flow_node["id"] = "video-4k-stream-1"
    for edge in native["topology"]["edges"]:
        if edge["source"] == "ue-1-flow":
            edge["source"] = "video-4k-stream-1"
        if edge["target"] == "ue-1-flow":
            edge["target"] = "video-4k-stream-1"
    client = _Client()
    client.native = native
    adapter = LiveTestbedAdapter(client=client)
    adapter.capture_snapshot()

    result = adapter.observe(
        ObservationRequest(
            query_name="search_sm_flow_targets",
            arguments={
                "supi": "imsi-208930000000001",
                "query": "4K_Video_stream_1",
            },
        )
    ).result

    assert [flow["id"] for flow in result["matches"]] == ["video-4k-stream-1"]


def test_testbed_adapter_dispatches_supported_sm_projection() -> None:
    client = _Client()
    adapter = LiveTestbedAdapter(client=client)
    adapter.capture_snapshot()
    action = ExecutionAction(
        operations=[
            ExecutionOperation(
                operation_id="op-001",
                domain="qos",
                operation="update",
                resource_type="SmPolicyDecision",
                resource_id="smp-video-app-ue-1-flow",
                payload=SmPolicyDecision(
                    pccRules={
                        "pcc-ue-1-flow": PCCRule(
                            pccRuleId="pcc-ue-1-flow",
                            appId="video-app",
                            refQosData=["qos-ue-1-flow"],
                        )
                    },
                    qosDecs={
                        "qos-ue-1-flow": QosData(
                            qosId="qos-ue-1-flow",
                            gbrUl="5 Mbps",
                            maxbrUl="8 Mbps",
                            gbrDl="6 Mbps",
                            maxbrDl="10 Mbps",
                        )
                    },
                ),
            )
        ]
    )

    feedback = adapter.execute(action)

    assert feedback.status == "success"
    assert feedback.snapshot.snapshot_id == "snapshot-2"
    assert feedback.receipts[0].status == "success"
    assert client.dispatched == [
        {
            "run_id": "run-1",
            "snapshot_id": "snapshot-1",
            "action_type": "sm_policy_update",
            "target": {
                "supi": "imsi-208930000000001",
                "flow_id": "ue-1-flow",
            },
            "parameters": {
                "five_qi": 9,
                "gbr_ul_mbps": 5.0,
                "mbr_ul_mbps": 8.0,
                "gbr_dl_mbps": 6.0,
                "mbr_dl_mbps": 10.0,
            },
        }
    ]


def _bind_live_client(monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
    monkeypatch.setattr(
        LiveTestbedConfig,
        "from_env",
        classmethod(lambda _cls: client.config),
    )
    monkeypatch.setattr(
        LiveTestbedClient,
        "create",
        classmethod(lambda _cls, _config: client),
    )


def test_native_snapshot_contract_requires_complete_bidirectional_flow_data() -> None:
    validated = NativeSnapshot.model_validate(_snapshot())
    assert validated.flows[0].traffic.packet_size == 9600

    invalid = _snapshot()
    del invalid["flows"][0]["sla"]["bandwidth_dl"]
    with pytest.raises(ValidationError):
        NativeSnapshot.model_validate(invalid)


def test_native_snapshot_preserves_prb_capacity_when_utilization_is_unobserved() -> None:
    native = _snapshot()
    native["ran_nodes"][0]["telemetry"]["prb_utilization"] = None

    validated = NativeSnapshot.model_validate(native)
    projected = project_environment_snapshot(validated.model_dump(mode="json"))
    ran_node = next(entity for entity in projected.entities if entity.entity_id == "gnb-1")

    assert validated.ran_nodes[0].capacity.prb == 106.0
    assert validated.ran_nodes[0].telemetry.prb_utilization is None
    assert ran_node.attributes["capacity"]["prb"] == 106.0
    assert ran_node.attributes["telemetry"]["prb_utilization"] is None


def test_testbed_client_validates_snapshot_and_action_without_retyping_subscription() -> (
    None
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/snapshots/latest":
            return httpx.Response(200, json=_snapshot())
        if request.url.path.startswith("/api/subscriber/"):
            return httpx.Response(200, json=_subscriber())
        if request.url.path == "/v1/actions/policy":
            return httpx.Response(
                200,
                json={
                    "action_id": "action-1",
                    "run_id": "run-1",
                    "snapshot_before": "snapshot-1",
                    "status": "verified",
                    "dispatch": {},
                    "observed_effect": {},
                    "snapshot_after": "snapshot-2",
                },
            )
        raise AssertionError(request.url)

    config = LiveTestbedConfig(
        snapshot_base_url="http://snapshot",
        action_base_url="http://action",
        subscriber_base_url="http://subscriber",
        run_id="run-1",
        plmn_id="20893",
        action_api_key="secret",
        timeout_seconds=5.0,
    )
    client = LiveTestbedClient(
        config=config,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.latest_snapshot()["snapshot_id"] == "snapshot-1"
    assert client.subscriber("imsi-208930000000001")[
        "AccessAndMobilitySubscriptionData"
    ]["nssai"]["singleNssais"] == [{"sst": 1, "sd": "010203"}]
    assert client.dispatch_policy({})["status"] == "verified"


def test_storage_scenario_and_semantic_modules_read_the_live_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    _bind_live_client(monkeypatch, client)

    native = get_latest_snapshot_data()
    context = get_ue_context_by_supi("imsi-208930000000001")
    search = search_flow_targets_by_semantic(
        supi="imsi-208930000000001",
        app_name="video",
    )
    status = get_network_status()

    assert native["snapshot_id"] == "snapshot-1"
    assert context["flow_catalog"][0]["traffic"]["five_tuple"][0] == "10.60.0.1"
    assert search["candidates"][0]["flow_id"] == "ue-1-flow"
    assert status["allocated_bandwidth_ul_mbps"] == 8.0
    assert status["throughput_dl_mbps"] == 6.5


def test_optimizer_scenario_uses_live_slice_and_node_capacities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "control_runtime.integrations.optimizer.interface.get_snapshot_data_by_id",
        lambda _snapshot_id: _snapshot(),
    )

    apps, slices, nodes = _load_optimizer_scenario("snapshot-1")

    assert apps[0].flows[0].ran_node_id == "gnb-1"
    assert slices[0].capacity.total_bandwidth_dl == 80.0
    assert slices[0].load.current_bandwidth_ul == 8.0
    assert {node.id: node.capacity.prb for node in nodes} == {
        "gnb-1": 106.0,
        "upf-1": 0.0,
    }


def test_full_topology_optimizer_returns_json_safe_live_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "control_runtime.integrations.optimizer.interface.get_snapshot_data_by_id",
        lambda _snapshot_id: _snapshot(),
    )
    native_flow = _snapshot()["flows"][0]

    result = optimize_network_slices(
        {
            "app_id": native_flow["app_id"],
            "name": native_flow["app_name"],
            "supi": native_flow["supi"],
            "flows": [
                {
                    "flow_id": native_flow["id"],
                    "app_id": native_flow["app_id"],
                    "app_name": native_flow["app_name"],
                    "supi": native_flow["supi"],
                }
            ],
        },
        1.0,
        1.0,
        1.0,
        0.2,
        mode="incremental",
        return_json=True,
        debug_context={
            "episode_id": "capacity-test",
            "snapshot_id": "snapshot-1",
            "requested_domains": ["qos"],
            "target_ues": [native_flow["supi"]],
            "policy_state": {},
            "slice_kpi_source": "qos",
            "qos_relaxation_ratio": 0.2,
            "qos_feasibility_mode": "hard",
            "enable_sla_constraints": True,
        },
    )

    assert result["meta"]["status"] == "Optimal"
    assert isinstance(result["target_app"]["flows"][0]["traffic"]["five_tuple"], list)


def test_pcf_dispatch_migrates_to_the_bidirectional_action_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    _bind_live_client(monkeypatch, client)

    result = dispatch_policy_to_pcf_request(
        "SmPolicyDecision",
        {
            "supi": "imsi-208930000000001",
            "flow_id": "ue-1-flow",
            "qosDecs": {
                "qos-ue-1-flow": {
                    "gbrUl": "5 Mbps",
                    "maxbrUl": "8 Mbps",
                    "gbrDl": "6 Mbps",
                    "maxbrDl": "10 Mbps",
                }
            },
        },
        snapshot_id="snapshot-1",
    )

    assert result["status"] == "verified"
    assert client.dispatched[0]["parameters"] == {
        "five_qi": 9,
        "gbr_ul_mbps": 5.0,
        "mbr_ul_mbps": 8.0,
        "gbr_dl_mbps": 6.0,
        "mbr_dl_mbps": 10.0,
    }
