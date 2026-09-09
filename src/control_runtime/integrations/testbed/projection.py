from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from control_runtime.simulation import (
    EnvironmentSnapshot,
    SnapshotEntity,
    SnapshotMetric,
    SnapshotRelation,
)

NATIVE_ENTITY_TYPES = {
    "ue": "ue",
    "flow": "flow",
    "slice": "slice",
    "gnb": "ran_node",
    "cell": "ran_node",
    "upf": "core_node",
    "site": "core_node",
    "service_instance": "app",
}


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def snssai_code(value: dict[str, Any]) -> str:
    return f"{int(value['sst']):02X}{value['sd']}"


def project_subscription(
    *, supi: str, plmn_id: str, subscriber: dict[str, Any]
) -> dict[str, Any]:
    am_data = subscriber["AccessAndMobilitySubscriptionData"]
    nssai = am_data["nssai"]
    sm_data = subscriber["SessionManagementSubscriptionData"]
    default_snssais = [snssai_code(item) for item in nssai["defaultSingleNssais"]]
    subscribed_snssais = [snssai_code(item) for item in nssai["singleNssais"]]
    sessions = [
        {
            "snssai": snssai_code(item["singleNssai"]),
            "dnn_configurations": item["dnnConfigurations"],
        }
        for item in sm_data
    ]
    return {
        "supi": supi,
        "plmn_id": plmn_id,
        "default_snssais": default_snssais,
        "subscribed_snssais": subscribed_snssais,
        "allowed_snssais": subscribed_snssais,
        "subscribed_ue_ambr": am_data["subscribedUeAmbr"],
        "session_management": sessions,
        "dnns": sorted(
            dnn for session in sessions for dnn in session["dnn_configurations"]
        ),
        "source": "free5gc-subscriber-api",
    }


def _numeric_metrics(
    *,
    owner: str,
    prefix: str,
    payload: dict[str, Any],
    observed_at: datetime,
) -> list[SnapshotMetric]:
    metrics: list[SnapshotMetric] = []
    for name, value in payload.items():
        metric_name = f"{prefix}.{name}"
        if isinstance(value, dict):
            metrics.extend(
                _numeric_metrics(
                    owner=owner,
                    prefix=metric_name,
                    payload=value,
                    observed_at=observed_at,
                )
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics.append(
                SnapshotMetric(
                    owner_entity_id=owner,
                    metric_name=metric_name,
                    value=value,
                    observed_at=observed_at,
                )
            )
    return metrics


def project_environment_snapshot(native: dict[str, Any]) -> EnvironmentSnapshot:
    observed_at = parse_timestamp(native["observed_to"])
    entities: dict[str, SnapshotEntity] = {}

    for node in native["topology"]["nodes"]:
        native_type = node["type"]
        entities[node["id"]] = SnapshotEntity(
            entity_id=node["id"],
            entity_type=NATIVE_ENTITY_TYPES[native_type],
            attributes={"native_type": native_type, **node["attributes"]},
        )

    enrichment_groups = (
        ("flows", "flow", "id"),
        ("slice_states", "slice", "slice_id"),
        ("ran_nodes", "ran_node", "id"),
        ("core_nodes", "core_node", "id"),
        ("mec_sites", "core_node", "id"),
    )
    for group_name, entity_type, id_field in enrichment_groups:
        for item in native[group_name]:
            entity_id = item[id_field]
            current = entities.get(entity_id)
            attributes = dict(current.attributes) if current is not None else {}
            if group_name == "slice_states":
                attributes["data_availability"] = {
                    "missing_fields": [
                        "capacity.total_bandwidth_ul",
                        "capacity.total_bandwidth_dl",
                        "capacity.guaranteed_bandwidth_ul",
                        "capacity.guaranteed_bandwidth_dl",
                        "load.current_bandwidth_ul",
                        "load.current_bandwidth_dl",
                        "qos",
                    ]
                }
            entities[entity_id] = SnapshotEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                attributes={**attributes, **item},
            )

    for container in native["containers"]:
        entity_id = f"container:{container['id']}"
        entities[entity_id] = SnapshotEntity(
            entity_id=entity_id,
            entity_type="core_node",
            attributes={"native_type": "container", **container},
        )

    for session in native["sessions"]:
        entity_id = f"session:{session['supi']}:{session['pdu_session_id']}"
        entities[entity_id] = SnapshotEntity(
            entity_id=entity_id,
            entity_type="session",
            attributes=session,
        )

    for action in native["active_actions"]:
        action_id = action["source_action_id"]
        entities[action_id] = SnapshotEntity(
            entity_id=action_id,
            entity_type="policy_binding",
            attributes=action,
        )

    relations = [
        SnapshotRelation(
            relation_id=f"topology:{index}",
            relation_type=edge["type"],
            source_entity_id=edge["source"],
            target_entity_id=edge["target"],
        )
        for index, edge in enumerate(native["topology"]["edges"], start=1)
    ]
    flow_by_id = {flow["id"]: flow for flow in native["flows"]}
    for session in native["sessions"]:
        session_id = f"session:{session['supi']}:{session['pdu_session_id']}"
        ue_entity_id = next(
            node["id"]
            for node in native["topology"]["nodes"]
            if node["type"] == "ue" and node["attributes"]["supi"] == session["supi"]
        )
        relations.append(
            SnapshotRelation(
                relation_id=f"{ue_entity_id}:uses:{session_id}",
                relation_type="uses_session",
                source_entity_id=ue_entity_id,
                target_entity_id=session_id,
            )
        )
        for flow_id, flow in flow_by_id.items():
            if (
                flow["supi"] == session["supi"]
                and flow["session"]["pdu_session_id"] == session["pdu_session_id"]
            ):
                relations.append(
                    SnapshotRelation(
                        relation_id=f"{flow_id}:uses:{session_id}",
                        relation_type="uses_session",
                        source_entity_id=flow_id,
                        target_entity_id=session_id,
                    )
                )

    metrics: dict[tuple[str, str], SnapshotMetric] = {}
    for flow in native["flows"]:
        flow_observed_at = parse_timestamp(flow["telemetry"]["observed_at"])
        for prefix in ("sla", "allocation", "telemetry", "ran", "data_quality"):
            for metric in _numeric_metrics(
                owner=flow["id"],
                prefix=prefix,
                payload=flow[prefix],
                observed_at=flow_observed_at,
            ):
                metrics[(metric.owner_entity_id, metric.metric_name)] = metric
    for container in native["containers"]:
        owner = f"container:{container['id']}"
        for metric in _numeric_metrics(
            owner=owner,
            prefix="telemetry",
            payload=container,
            observed_at=observed_at,
        ):
            metrics[(metric.owner_entity_id, metric.metric_name)] = metric

    return EnvironmentSnapshot(
        snapshot_id=native["snapshot_id"],
        observed_at=observed_at,
        entities=list(entities.values()),
        relations=relations,
        metrics=list(metrics.values()),
    )


def build_flow_catalog(native: dict[str, Any], supi: str) -> dict[str, Any]:
    topology_flows = {
        node["id"]: node["attributes"]
        for node in native["topology"]["nodes"]
        if node["type"] == "flow"
    }
    flows = []
    for flow in native["flows"]:
        if flow["supi"] != supi:
            continue
        traffic = topology_flows[flow["id"]]
        missing = [
            field
            for field in (
                "sla.bandwidth_ul",
                "sla.bandwidth_dl",
                "sla.guaranteed_bandwidth_dl",
                "sla.priority",
                "allocation.allocated_bandwidth_ul",
                "allocation.allocated_bandwidth_dl",
                "traffic.five_tuple",
                "traffic.packet_size",
                "traffic.arrival_rate",
            )
            if flow[field.split(".")[0]].get(field.split(".")[1]) is None
        ]
        flows.append(
            {
                **flow,
                "service": {
                    "service_type": flow["app_name"],
                    "service_type_id": None,
                    "dnn": flow["session"]["dnn"],
                },
                "traffic": {**flow["traffic"], "protocol": traffic["protocol"]},
                "data_availability": {
                    "ready_for_grounding": True,
                    "ready_for_bidirectional_qos_optimization": not missing,
                    "missing_fields": missing,
                    "ran_measurement_scope": "ue_scheduler_associated_with_drb",
                },
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for flow in flows:
        grouped[flow["app_id"]].append(flow)
    return {
        "supi": supi,
        "app_catalog": [
            {
                "supi": supi,
                "app_id": app_id,
                "app_name": app_flows[0]["app_name"],
                "flow_count": len(app_flows),
            }
            for app_id, app_flows in grouped.items()
        ],
        "flow_catalog": flows,
        "source_snapshot_id": native["snapshot_id"],
    }


def build_am_context(
    native: dict[str, Any], subscription: dict[str, Any], supi: str
) -> dict[str, Any]:
    ue_node = next(
        node
        for node in native["topology"]["nodes"]
        if node["type"] == "ue" and node["attributes"]["supi"] == supi
    )
    serving_gnb = next(
        edge["target"]
        for edge in native["topology"]["edges"]
        if edge["type"] == "attached_to" and edge["source"] == ue_node["id"]
    )
    sessions = [item for item in native["sessions"] if item["supi"] == supi]
    slice_snssais = {
        node["id"]: snssai_code(node["attributes"])
        for node in native["topology"]["nodes"]
        if node["type"] == "slice"
    }
    ran_ues = [
        ue
        for ran in native["ran_nodes"]
        for ue in ran["telemetry"]["ues"]
        if ue["supi"] == supi
    ]
    policy_rows = [row for row in native.get("ue_policies", []) if row["supi"] == supi]
    am = policy_rows[0]["am_policy_associations"] if policy_rows else []
    ue_policy = policy_rows[0]["ue_policy_associations"] if policy_rows else []
    return {
        "supi": supi,
        "association_id": am[0]["id"] if len(am) == 1 else None,
        "rfsp": am[0]["rfsp"] if len(am) == 1 else None,
        "allowed_snssais": subscription["allowed_snssais"],
        "target_snssais": sorted(
            {slice_snssais[item["slice_id"]] for item in sessions}
        ),
        "access_type": "3GPP_ACCESS",
        "serving_gnb": serving_gnb,
        "rnti": ran_ues[0]["rnti"],
        "service_area_restriction": None,
        "associations": am,
        "data_availability": {
            "ready_for_mobility_grounding": len(am) == 1,
            "missing_fields": []
            if len(am) == 1
            else [
                "association_id",
                "rfsp",
                "service_area_restriction",
            ],
        },
        "execution_capability": {
            "am_policy": len(am) == 1,
            "ursp": len(ue_policy) == 1,
            "am_fields": ["rfsp"] if am else [],
            "ursp_profile": "ipv4_remote_tuple_existing_pdu_sessions"
            if ue_policy
            else None,
            "radio_frequency_reselection": False,
        },
        "source_snapshot_id": native["snapshot_id"],
    }


def build_optimizer_preview(
    native: dict[str, Any], supi: str, flow_ids: list[str]
) -> dict[str, Any]:
    catalog = build_flow_catalog(native, supi)["flow_catalog"]
    selected = [flow for flow in catalog if flow["id"] in flow_ids]
    assignments = [
        {
            "flow_id": flow["id"],
            "app_id": flow["app_id"],
            "bw_ul": flow["sla"]["bandwidth_ul"],
            "bw_dl": flow["sla"]["bandwidth_dl"],
            "source": "snapshot.flow.sla",
        }
        for flow in selected
        if flow["data_availability"]["ready_for_bidirectional_qos_optimization"]
    ]
    assigned_ids = {item["flow_id"] for item in assignments}
    selected_ids = {flow["id"] for flow in selected}
    blocked = [
        {
            "flow_id": flow["id"],
            "missing_fields": flow["data_availability"]["missing_fields"],
        }
        for flow in selected
        if flow["id"] not in assigned_ids
    ]
    blocked.extend(
        {"flow_id": flow_id, "missing_fields": ["flow"]}
        for flow_id in flow_ids
        if flow_id not in selected_ids
    )
    return {
        "supi": supi,
        "qos_flow_assignments": assignments,
        "blocked_flows": blocked,
        "ready": not blocked and len(selected) == len(flow_ids),
        "source_snapshot_id": native["snapshot_id"],
    }


__all__ = [
    "build_am_context",
    "build_flow_catalog",
    "build_optimizer_preview",
    "parse_timestamp",
    "project_environment_snapshot",
    "project_subscription",
]
