from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from shared.logging import setup_logger
from database.models import (
    UeAmPolicyAssociationRecord,
    UeContextRecord,
    UeMobilityEventRecord,
    UeServingNfBindingRecord,
)

from ..storage import (
    build_flow_info_from_five_tuple,
    get_latest_snapshot_data,
    record_mobility_event,
    session_scope,
    sync_latest_snapshot_flow_catalog_to_ue_context,
    upsert_am_policy_association,
    upsert_serving_nf_binding,
    upsert_ue_context,
)
from ..optimizer.models import App, Node, Slice
from .common import (
    _asdict_list,
    _filter_dataclass_kwargs,
    _map_5qi_by_service_type,
    _normalize_or_generate_id,
    _normalize_supi,
    _service_type_id_to_name,
    _service_type_name_to_id,
    _build_flow_from_dict,
    _build_node_from_dict,
    _build_slice_from_dict,
    cache_scenario,
    snapshot_uses_new_schema,
)
from .yaml_loader import _deserialize_scenario

logger = setup_logger(__name__)

def _build_seed_mobility_context(supi: str, index: int, app: App) -> Dict[str, Any]:
    allowed_snssai = []
    target_snssai = []
    for flow in app.flows:
        try:
            sst = int(flow.service.service_type_id)
        except (TypeError, ValueError):
            sst = 1
        sd = "000001" if sst in (1, 2) else "000002"
        snssai = {"sst": sst, "sd": sd}
        if snssai not in allowed_snssai:
            allowed_snssai.append(snssai)
        if flow.sla.priority <= 3 and snssai not in target_snssai:
            target_snssai.append(snssai)
    if not target_snssai:
        target_snssai = list(allowed_snssai)

    return {
        "accessType": "3GPP_ACCESS",
        "accessTypes": ["3GPP_ACCESS"],
        "ratType": "NR" if index % 2 == 0 else "EUTRA",
        "ratTypes": ["NR", "EUTRA"],
        "userLoc": {
            "nrLocation": {
                "tai": {"plmnId": {"mcc": "208", "mnc": "93"}, "tac": f"{1000 + index:06d}"},
                "ncgi": {"plmnId": {"mcc": "208", "mnc": "93"}, "nrCellId": f"{(index + 1) * 1000000:09d}"},
            }
        },
        "guami": {"plmnId": {"mcc": "208", "mnc": "93"}, "amfId": f"{index + 1:06x}"},
        "servingPlmn": {"mcc": "208", "mnc": "93"},
        "timeZone": "+08:00",
        "presenceAreas": {
            f"pra-{index + 1:03d}": {
                "praId": f"{index + 1}",
                "presenceState": "IN_AREA",
                "trackingAreaList": [{"plmnId": {"mcc": "208", "mnc": "93"}, "tac": f"{1000 + index:06d}"}],
            }
        },
        "allowedSnssais": allowed_snssai,
        "targetSnssais": target_snssai,
        "mappingSnssais": [{"servingSnssai": item, "homeSnssai": item} for item in allowed_snssai],
        "mobilityEventType": "INITIAL_ATTACH",
    }


def _build_seed_am_policy_context(supi: str, index: int, mobility_context: Dict[str, Any]) -> Dict[str, Any]:
    pol_asso_id = f"{supi}-am-{index + 1}"
    return {
        "polAssociationIDGenerator": index + 1,
        "associations": {
            pol_asso_id: {
                "request": {
                    "notificationUri": f"http://localhost:8000/notify/{supi}",
                    "supi": supi,
                    "accessType": mobility_context["accessType"],
                    "accessTypes": mobility_context["accessTypes"],
                    "ratType": mobility_context["ratType"],
                    "ratTypes": mobility_context["ratTypes"],
                    "userLoc": mobility_context["userLoc"],
                    "guami": mobility_context["guami"],
                    "servingPlmn": mobility_context["servingPlmn"],
                    "timeZone": mobility_context["timeZone"],
                    "allowedSnssais": mobility_context["allowedSnssais"],
                    "targetSnssais": mobility_context["targetSnssais"],
                    "mappingSnssais": mobility_context["mappingSnssais"],
                    "rfsp": 1 + (index % 4),
                    "suppFeat": "1",
                },
                "triggers": ["LOC_CH", "PRA_CH", "ALLOWED_NSSAI_CH"],
                "rfsp": 1 + (index % 4),
                "pras": mobility_context["presenceAreas"],
                "suppFeat": "1",
            }
        },
        "allowedSnssais": mobility_context["allowedSnssais"],
        "targetSnssais": mobility_context["targetSnssais"],
        "mappingSnssais": mobility_context["mappingSnssais"],
        "rfsp": 1 + (index % 4),
        "pras": mobility_context["presenceAreas"],
    }


def _seed_ue_contexts_from_apps(apps: List[App]) -> int:
    """
    根据场景中的 flow 为每个 UE 生成并写入 UeContext 关键字段。
    返回成功 upsert 的 UE 数量。
    """
    ue_payload: Dict[str, Dict[str, Any]] = {}

    for app_index, app in enumerate(apps):
        supi = _normalize_supi(getattr(app, "supi", None))
        mobility_context = _build_seed_mobility_context(supi, app_index, app) if supi else {}
        am_policy_context = _build_seed_am_policy_context(supi, app_index, mobility_context) if supi else {}
        for flow in app.flows:
            flow_id = flow.id
            if not supi or not flow_id:
                continue

            try:
                service_type_id = int(flow.service.service_type_id)
            except (TypeError, ValueError):
                service_type_id = 1

            sm_policy_id = f"{supi}-1"
            pcc_rule_id = f"pcc-{flow_id}"
            qos_id = f"qos-{flow_id}"
            sess_rule_id = f"sess-{flow_id}"

            if supi not in ue_payload:
                ue_payload[supi] = {
                    "sm_policy_data": {},
                    "pcc_rules": {},
                    "qos_decs": {},
                    "sess_rules": {},
                    "traff_cont_decs": {},
                    "chg_decs": {},
                    "app_catalog": [],
                    "flow_catalog": [],
                    "access_mobility_context": mobility_context,
                    "am_policy_context": am_policy_context,
                    "serving_nf_context": {
                        "pcf_id": "pcf-sim-1",
                        "pcf_uri": "http://localhost:8000/pcf",
                        "amf_id": f"amf-sim-{app_index + 1}",
                        "amf_uri": f"http://localhost:8000/amf/{supi}",
                    },
                    "mobility_summary": {
                        "currentAssociationId": next(iter(am_policy_context.get("associations", {}).keys()), None),
                        "currentTriggers": ["LOC_CH", "PRA_CH", "ALLOWED_NSSAI_CH"],
                        "lastMobilityEventType": "INITIAL_ATTACH",
                        "currentRfsp": am_policy_context.get("rfsp"),
                        "lastUpdatedReason": "scenario_seed",
                    },
                }

            if not any(item.get("app_id") == app.id for item in ue_payload[supi]["app_catalog"]):
                ue_payload[supi]["app_catalog"].append(
                    {
                        "supi": supi,
                        "app_name": app.name,
                        "app_id": app.id,
                        "flow_count": len(app.flows or []),
                    }
                )

            # 关键步骤：按 smPolicyId 分组，和 UeContext.smPolicyData 语义一致
            ue_payload[supi]["sm_policy_data"].setdefault(sm_policy_id, {})
            ue_payload[supi]["pcc_rules"].setdefault(sm_policy_id, {})
            ue_payload[supi]["qos_decs"].setdefault(sm_policy_id, {})
            ue_payload[supi]["sess_rules"].setdefault(sm_policy_id, {})
            ue_payload[supi]["flow_catalog"].append(
                {
                    "supi": supi,
                    "app_name": app.name,
                    "app_id": app.id,
                    "flow_name": flow.name,
                    "flow_id": flow_id,
                    "service": {
                        "service_type": flow.service.service_type,
                        "service_type_id": service_type_id,
                    },
                    "sla": {
                        "bandwidth_ul": flow.sla.bandwidth_ul,
                        "bandwidth_dl": flow.sla.bandwidth_dl,
                        "guaranteed_bandwidth_ul": flow.sla.guaranteed_bandwidth_ul,
                        "guaranteed_bandwidth_dl": flow.sla.guaranteed_bandwidth_dl,
                        "latency": flow.sla.latency,
                        "jitter": flow.sla.jitter,
                        "loss_rate": flow.sla.loss_rate,
                        "priority": flow.sla.priority,
                    },
                    "allocation": {
                        "current_slice_snssai": flow.allocation.current_slice_snssai,
                        "allocated_bandwidth_ul": (
                            flow.allocation.allocated_bandwidth_ul
                            if flow.allocation.allocated_bandwidth_ul is not None
                            else flow.sla.bandwidth_ul
                        ),
                        "allocated_bandwidth_dl": (
                            flow.allocation.allocated_bandwidth_dl
                            if flow.allocation.allocated_bandwidth_dl is not None
                            else flow.sla.bandwidth_dl
                        ),
                    },
                    "traffic": {
                        "packet_size": flow.traffic.packet_size,
                        "arrival_rate": flow.traffic.arrival_rate,
                        "five_tuple": list(flow.traffic.five_tuple)
                        if isinstance(flow.traffic.five_tuple, (list, tuple))
                        else None,
                    },
                }
            )

            flow_info = build_flow_info_from_five_tuple(flow.traffic.five_tuple)
            if not flow_info:
                flow_info = {
                    "flowDescription": f"permit out ip from {supi} to any",
                    "flowDirection": "BIDIRECTIONAL",
                }

            ue_payload[supi]["pcc_rules"][sm_policy_id][pcc_rule_id] = {
                "pccRuleId": pcc_rule_id,
                "precedence": int(flow.sla.priority),
                "flowInfos": [flow_info],
                "refQosData": [qos_id],
                # 补充 TSCAI 信息 (如果 flow 有相关字段)
                "tscaiInputDl": [{
                    "periodicity": int(1000.0 / flow.traffic.arrival_rate * 1e6) if flow.traffic.arrival_rate > 0 else None,
                    "surTimeInNumMsg": int(flow.traffic.packet_size)
                }],
                "tscaiInputUl": [{
                    "periodicity": int(1000.0 / flow.traffic.arrival_rate * 1e6) if flow.traffic.arrival_rate > 0 else None,
                    "surTimeInNumMsg": int(flow.traffic.packet_size)
                }]
            }

            ue_payload[supi]["qos_decs"][sm_policy_id][qos_id] = {
                "qosId": qos_id,
                "5qi": _map_5qi_by_service_type(service_type_id),
                "gbrUl": str(flow.sla.guaranteed_bandwidth_ul),
                "gbrDl": str(flow.sla.guaranteed_bandwidth_dl),
                "maxbrUl": str(flow.sla.bandwidth_ul),
                "maxbrDl": str(flow.sla.bandwidth_dl),
                "packetDelayBudget": int(flow.sla.latency),
                "packetErrorRate": str(flow.sla.loss_rate),
                "priorityLevel": int(flow.sla.priority),
            }

            ue_payload[supi]["sess_rules"][sm_policy_id][sess_rule_id] = {
                "sessRuleId": sess_rule_id,
                "authDefQos": {
                    "5qi": _map_5qi_by_service_type(service_type_id),
                    "priorityLevel": int(flow.sla.priority),
                    "gbrUl": str(flow.sla.guaranteed_bandwidth_ul),
                    "gbrDl": str(flow.sla.guaranteed_bandwidth_dl),
                },
            }

    success = 0
    for supi, payload in ue_payload.items():
        ok = upsert_ue_context(
            supi=supi,
            sm_policy_data=payload["sm_policy_data"],
            pcc_rules=payload["pcc_rules"],
            qos_decs=payload["qos_decs"],
            sess_rules=payload["sess_rules"],
            traff_cont_decs=payload["traff_cont_decs"],
            chg_decs=payload["chg_decs"],
            app_catalog=payload["app_catalog"],
            flow_catalog=payload["flow_catalog"],
            access_mobility_context=payload["access_mobility_context"],
            am_policy_context=payload["am_policy_context"],
            serving_nf_context=payload["serving_nf_context"],
            mobility_summary=payload["mobility_summary"],
        )
        if ok:
            success += 1

    return success


def sync_latest_flow_five_tuples_to_ue_context(snapshot_id: str = "") -> Dict[str, int]:
    """
    Rebuild UE flow catalogs from the latest snapshot and refresh PCC flowInfos
    so flowDescription matches the authoritative five_tuple when available.
    """
    return sync_latest_snapshot_flow_catalog_to_ue_context(snapshot_id=snapshot_id)


def _build_mobility_snapshot_payload(apps: List[App]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for app_index, app in enumerate(apps):
        supi = _normalize_supi(getattr(app, "supi", None))
        if not supi:
            continue
        mobility_context = _build_seed_mobility_context(supi, app_index, app)
        payload.append({"supi": supi, **mobility_context})
    return payload


def _build_policy_state_payload(apps: List[App]) -> Dict[str, Any]:
    policy_state: Dict[str, Any] = {}
    for app_index, app in enumerate(apps):
        supi = _normalize_supi(getattr(app, "supi", None))
        if not supi:
            continue
        mobility_context = _build_seed_mobility_context(supi, app_index, app)
        policy_state[supi] = {
            "amPolicy": _build_seed_am_policy_context(supi, app_index, mobility_context),
            "appId": app.id,
            "appName": app.name,
        }
    return policy_state


def _load_graph_scenario_strict(snapshot_id: str) -> Tuple[List[App], List[Slice], List[Node], str]:
    from .network_graph import NetworkGraph, get_graph_snapshot_payload

    normalized_snapshot_id = str(snapshot_id or "").strip()
    if not normalized_snapshot_id:
        raise RuntimeError("graph snapshot_id is required")

    snapshot_data = get_graph_snapshot_payload(normalized_snapshot_id)
    if not isinstance(snapshot_data, dict):
        raise RuntimeError(f"graph snapshot not found: snapshot_id={normalized_snapshot_id}")
    snapshot_data = NetworkGraph.from_payload(snapshot_data).to_compatibility_snapshot()
    if not snapshot_uses_new_schema(snapshot_data):
        raise RuntimeError(f"graph snapshot does not contain a valid scenario payload: snapshot_id={normalized_snapshot_id}")

    apps, slices, nodes = _deserialize_scenario(snapshot_data)
    snapshot_id = str(snapshot_data.get("snapshot_id") or "").strip()
    return apps, slices, nodes, snapshot_id


def _load_latest_graph_scenario_strict() -> Tuple[List[App], List[Slice], List[Node], str]:
    from .network_graph import get_latest_graph_snapshot_metadata

    metadata = get_latest_graph_snapshot_metadata() or {}
    return _load_graph_scenario_strict(str(metadata.get("snapshot_id") or "").strip())


def rebuild_ue_related_tables_from_graph_snapshot(snapshot_id: str) -> Dict[str, Any]:
    apps, slices, nodes, snapshot_id = _load_graph_scenario_strict(snapshot_id)

    with session_scope() as session:
        session.query(UeMobilityEventRecord).delete()
        session.query(UeServingNfBindingRecord).delete()
        session.query(UeAmPolicyAssociationRecord).delete()
        session.query(UeContextRecord).delete()

    seeded = _seed_ue_contexts_from_apps(apps)
    sync_summary = sync_latest_flow_five_tuples_to_ue_context(snapshot_id=snapshot_id)

    association_count = 0
    binding_count = 0
    mobility_event_count = 0
    for app_index, app in enumerate(apps):
        supi = _normalize_supi(getattr(app, "supi", None))
        if not supi:
            continue

        mobility_context = _build_seed_mobility_context(supi, app_index, app)
        am_policy_context = _build_seed_am_policy_context(supi, app_index, mobility_context)
        associations = am_policy_context.get("associations") if isinstance(am_policy_context, dict) else {}
        if not isinstance(associations, dict):
            associations = {}

        for pol_asso_id, association in associations.items():
            association = association if isinstance(association, dict) else {}
            association_policy = {
                "triggers": list(association.get("triggers") or []),
                "rfsp": association.get("rfsp"),
                "pras": association.get("pras") or {},
                "suppFeat": association.get("suppFeat"),
                "allowedSnssais": am_policy_context.get("allowedSnssais") or [],
                "targetSnssais": am_policy_context.get("targetSnssais") or [],
                "mappingSnssais": am_policy_context.get("mappingSnssais") or [],
            }
            if upsert_am_policy_association(
                supi=supi,
                pol_asso_id=str(pol_asso_id),
                association_request=association.get("request") or {},
                association_policy=association_policy,
                status="active",
                trigger_event="GRAPH_REBUILD",
                snapshot_id=snapshot_id,
            ):
                association_count += 1

        serving_nf_context = {
            "pcf_id": "pcf-sim-1",
            "pcf_uri": "http://localhost:8000/pcf",
            "amf_id": f"amf-sim-{app_index + 1}",
            "amf_uri": f"http://localhost:8000/amf/{supi}",
        }
        if upsert_serving_nf_binding(
            supi=supi,
            nf_type="PCF",
            nf_instance_id=serving_nf_context["pcf_id"],
            nf_uri=serving_nf_context["pcf_uri"],
            binding_info={"supi": supi, "snapshot_id": snapshot_id},
            status="active",
        ):
            binding_count += 1
        if upsert_serving_nf_binding(
            supi=supi,
            nf_type="AMF",
            nf_instance_id=serving_nf_context["amf_id"],
            nf_uri=serving_nf_context["amf_uri"],
            binding_info={"supi": supi, "snapshot_id": snapshot_id},
            status="active",
        ):
            binding_count += 1

        if record_mobility_event(
            supi=supi,
            event_type=str(mobility_context.get("mobilityEventType") or "INITIAL_ATTACH"),
            event_payload=mobility_context,
            event_summary="rebuild_from_latest_graph",
            snapshot_id=snapshot_id,
        ):
            mobility_event_count += 1

    cache_scenario(
        apps,
        slices,
        nodes,
        _build_mobility_snapshot_payload(apps),
        _build_policy_state_payload(apps),
        snapshot_id=snapshot_id,
    )

    return {
        "snapshot_id": snapshot_id,
        "ues": seeded,
        "flow_catalog_ues": sync_summary.get("ues", 0),
        "flow_catalog_flows": sync_summary.get("flows", 0),
        "am_policy_associations": association_count,
        "serving_nf_bindings": binding_count,
        "mobility_events": mobility_event_count,
    }


def rebuild_ue_related_tables_from_latest_graph() -> Dict[str, Any]:
    apps, slices, nodes, snapshot_id = _load_latest_graph_scenario_strict()
    return rebuild_ue_related_tables_from_graph_snapshot(snapshot_id)

