from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from ..optimizer.models import (
    App,
    Flow,
    FlowAllocation,
    FlowService,
    FlowSLA,
    FlowTraffic,
    Node,
    NodeCapacity,
    Slice,
    SliceCapacity,
    SliceLoad,
    SliceQos,
)
from .common import (
    _extract_app_supi,
    _filter_dataclass_kwargs,
    _normalize_or_generate_id,
    _normalize_supi,
    _service_type_id_to_name,
    _service_type_name_to_id,
    _build_flow_from_dict,
    _build_node_from_dict,
    _build_slice_from_dict,
)

def _deserialize_scenario(data: Dict[str, Any]) -> Tuple[List[App], List[Slice], List[Node]]:
    used_suffixes: set = set()
    apps: List[App] = []
    for app_dict in data.get("apps", []):
        flows = []
        for f_dict in app_dict.get("flows", []):
            flows.append(_build_flow_from_dict(f_dict, used_suffixes=used_suffixes))

        app_kwargs = {
            "name": app_dict.get("name"),
            "id": app_dict.get("id", app_dict.get("app_id")),
        }
        app_kwargs["id"] = _normalize_or_generate_id(
            app_kwargs.get("id"),
            "app",
            used_suffixes,
        )
        app_kwargs["supi"] = _extract_app_supi(app_dict)
        apps.append(App(flows=flows, **app_kwargs))

    slices: List[Slice] = []
    for s_dict in data.get("slices", []):
        slices.append(_build_slice_from_dict(s_dict))

    nodes: List[Node] = []
    for n_dict in data.get("nodes", []):
        nodes.append(_build_node_from_dict(n_dict))

    return apps, slices, nodes


def _load_yaml_payload(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Scenario YAML root must be a mapping: {path}")
    return payload


def _parse_slice_snssai(slices_payload: List[Dict[str, Any]]) -> Tuple[List[Slice], Dict[str, str], List[str]]:
    slices: List[Slice] = []
    label_to_snssai: Dict[str, str] = {}
    all_snssai: List[str] = []

    for item in slices_payload:
        sst = int(item.get("sst") or 0)
        sd = str(item.get("sd") or "000000")
        label = str(item.get("label") or f"slice-{sst}-{sd}").strip() or f"slice-{sst}-{sd}"
        resource = item.get("resource") if isinstance(item.get("resource"), dict) else {}
        qos = item.get("qos") if isinstance(item.get("qos"), dict) else {}
        slice_obj = Slice(
            name=label,
            sst=sst,
            sd=sd,
            capacity=SliceCapacity(
                total_bandwidth_ul=float(resource.get("capacity_ul_mbps", 0.0) or 0.0),
                total_bandwidth_dl=float(resource.get("capacity_dl_mbps", 0.0) or 0.0),
                guaranteed_bandwidth_ul=float(resource.get("guaranteed_ul_mbps", 0.0) or 0.0),
                guaranteed_bandwidth_dl=float(resource.get("guaranteed_dl_mbps", 0.0) or 0.0),
            ),
            load=SliceLoad(),
            qos=SliceQos(
                latency=float(qos.get("latency_ms", qos.get("latency", 0.0)) or 0.0),
                processing_delay=float(qos.get("processing_delay_ms", qos.get("processing_delay", 0.0)) or 0.0),
                jitter=float(qos.get("jitter_ms", qos.get("jitter", 0.0)) or 0.0),
                loss_rate=float(qos.get("loss_rate", qos.get("loss", 0.0)) or 0.0),
            ),
        )
        slices.append(slice_obj)
        label_to_snssai[label] = slice_obj.snssai
        all_snssai.append(slice_obj.snssai)

    return slices, label_to_snssai, sorted(set(all_snssai))


def _build_nodes_from_yaml(
    *,
    gnbs_payload: List[Dict[str, Any]],
    upfs_payload: List[Dict[str, Any]],
    label_to_snssai: Dict[str, str],
    all_snssai: List[str],
) -> List[Node]:
    nodes: List[Node] = []

    for index, item in enumerate(gnbs_payload):
        hosted = [label_to_snssai[label] for label in item.get("slices", []) or [] if label in label_to_snssai]
        nodes.append(
            Node(
                id=index,
                name=str(item.get("name") or f"AN_gNB_{index}"),
                node_type="AN",
                capacity=NodeCapacity(cpu=1200.0, memory=250.0, mec=1200.0, prb=2500.0),
                hosted_slice_snssais=sorted(set(hosted)),
            )
        )

    for index, item in enumerate(upfs_payload):
        nodes.append(
            Node(
                id=100 + index,
                name=str(item.get("name") or f"CN_UPF_{index}"),
                node_type="CN",
                capacity=NodeCapacity(cpu=4200.0, memory=2024.0, mec=1800.0, prb=0.0),
                hosted_slice_snssais=list(all_snssai),
            )
        )

    return nodes


def _build_apps_from_yaml(
    *,
    apps_payload: List[Dict[str, Any]],
    flows_payload: List[Dict[str, Any]],
    ues_payload: List[Dict[str, Any]],
    label_to_snssai: Dict[str, str],
) -> List[App]:
    used_suffixes: set = set()
    ue_supi_by_name = {
        str(item.get("name") or "").strip(): _normalize_supi(item.get("supi"))
        for item in ues_payload
        if str(item.get("name") or "").strip()
    }
    flows_by_app_id: Dict[str, List[Dict[str, Any]]] = {}
    for flow in flows_payload:
        app_id = str(flow.get("app_id") or "").strip()
        if not app_id:
            continue
        flows_by_app_id.setdefault(app_id, []).append(flow)

    apps: List[App] = []
    seen_app_ids: set[str] = set()
    for app_payload in apps_payload:
        raw_app_id = str(app_payload.get("app_id") or "").strip()
        if raw_app_id:
            seen_app_ids.add(raw_app_id)
        app_supi = _normalize_supi(app_payload.get("supi")) or ue_supi_by_name.get(str(app_payload.get("ue_name") or "").strip())
        flow_objects: List[Flow] = []
        for flow_payload in flows_by_app_id.get(raw_app_id, []):
            service_type = str(flow_payload.get("service_type") or "").strip() or _service_type_id_to_name(
                int(flow_payload.get("service_type_id") or 1)
            )
            service_type_id = int(flow_payload.get("service_type_id") or _service_type_name_to_id(service_type) or 1)
            sla_target = flow_payload.get("sla_target") if isinstance(flow_payload.get("sla_target"), dict) else {}
            current_slice_snssai = str(flow_payload.get("current_slice_snssai") or "").strip()
            if not current_slice_snssai:
                current_slice_snssai = label_to_snssai.get(str(flow_payload.get("slice_ref") or "").strip(), "")
            flow_objects.append(
                Flow(
                    id=_normalize_or_generate_id(flow_payload.get("flow_id"), "flow", used_suffixes),
                    name=str(flow_payload.get("name") or flow_payload.get("flow_id") or "flow_unknown"),
                    service=FlowService(service_type=service_type, service_type_id=service_type_id),
                    sla=FlowSLA(
                        bandwidth_ul=float(
                            sla_target.get("bandwidth_ul_mbps", flow_payload.get("allocated_bandwidth_ul_mbps", 0.0)) or 0.0
                        ),
                        bandwidth_dl=float(
                            sla_target.get("bandwidth_dl_mbps", flow_payload.get("allocated_bandwidth_dl_mbps", 0.0)) or 0.0
                        ),
                        guaranteed_bandwidth_ul=float(
                            sla_target.get("guaranteed_bandwidth_ul_mbps", flow_payload.get("allocated_bandwidth_ul_mbps", 0.0)) or 0.0
                        ),
                        guaranteed_bandwidth_dl=float(
                            sla_target.get("guaranteed_bandwidth_dl_mbps", flow_payload.get("allocated_bandwidth_dl_mbps", 0.0)) or 0.0
                        ),
                        latency=float(sla_target.get("latency_ms", 0.0) or 0.0),
                        jitter=float(sla_target.get("jitter_ms", 0.0) or 0.0),
                        loss_rate=float(sla_target.get("loss_rate", 0.0) or 0.0),
                        priority=int(sla_target.get("priority", 1) or 1),
                    ),
                    traffic=FlowTraffic(
                        packet_size=float(flow_payload.get("packet_size_bytes", 0.0) or 0.0),
                        arrival_rate=float(flow_payload.get("arrival_rate_pps", 0.0) or 0.0),
                    ),
                    allocation=FlowAllocation(
                        current_slice_snssai=current_slice_snssai or None,
                        allocated_bandwidth_ul=float(flow_payload.get("allocated_bandwidth_ul_mbps", 0.0) or 0.0),
                        allocated_bandwidth_dl=float(flow_payload.get("allocated_bandwidth_dl_mbps", 0.0) or 0.0),
                        optimize_requested=bool(flow_payload.get("optimize_requested", False)),
                    ),
                )
            )

        apps.append(
            App(
                id=_normalize_or_generate_id(raw_app_id, "app", used_suffixes),
                name=str(app_payload.get("name") or raw_app_id or "app_unknown"),
                flows=flow_objects,
                supi=app_supi,
            )
        )

    referenced_app_ids = [app_id for app_id in flows_by_app_id if app_id not in seen_app_ids]
    for raw_app_id in referenced_app_ids:
        flow_objects: List[Flow] = []
        app_name = raw_app_id or "app_unknown"
        app_supi = None
        for flow_payload in flows_by_app_id.get(raw_app_id, []):
            service_type = str(flow_payload.get("service_type") or "").strip() or _service_type_id_to_name(
                int(flow_payload.get("service_type_id") or 1)
            )
            service_type_id = int(flow_payload.get("service_type_id") or _service_type_name_to_id(service_type) or 1)
            sla_target = flow_payload.get("sla_target") if isinstance(flow_payload.get("sla_target"), dict) else {}
            current_slice_snssai = str(flow_payload.get("current_slice_snssai") or "").strip()
            if not current_slice_snssai:
                current_slice_snssai = label_to_snssai.get(str(flow_payload.get("slice_ref") or "").strip(), "")
            if app_supi is None:
                app_supi = _normalize_supi(flow_payload.get("supi"))
            app_name = str(flow_payload.get("app_name") or app_name)
            flow_objects.append(
                Flow(
                    id=_normalize_or_generate_id(flow_payload.get("flow_id"), "flow", used_suffixes),
                    name=str(flow_payload.get("name") or flow_payload.get("flow_id") or "flow_unknown"),
                    service=FlowService(service_type=service_type, service_type_id=service_type_id),
                    sla=FlowSLA(
                        bandwidth_ul=float(
                            sla_target.get("bandwidth_ul_mbps", flow_payload.get("allocated_bandwidth_ul_mbps", 0.0)) or 0.0
                        ),
                        bandwidth_dl=float(
                            sla_target.get("bandwidth_dl_mbps", flow_payload.get("allocated_bandwidth_dl_mbps", 0.0)) or 0.0
                        ),
                        guaranteed_bandwidth_ul=float(
                            sla_target.get("guaranteed_bandwidth_ul_mbps", flow_payload.get("allocated_bandwidth_ul_mbps", 0.0)) or 0.0
                        ),
                        guaranteed_bandwidth_dl=float(
                            sla_target.get("guaranteed_bandwidth_dl_mbps", flow_payload.get("allocated_bandwidth_dl_mbps", 0.0)) or 0.0
                        ),
                        latency=float(sla_target.get("latency_ms", 0.0) or 0.0),
                        jitter=float(sla_target.get("jitter_ms", 0.0) or 0.0),
                        loss_rate=float(sla_target.get("loss_rate", 0.0) or 0.0),
                        priority=int(sla_target.get("priority", 1) or 1),
                    ),
                    traffic=FlowTraffic(
                        packet_size=float(flow_payload.get("packet_size_bytes", 0.0) or 0.0),
                        arrival_rate=float(flow_payload.get("arrival_rate_pps", 0.0) or 0.0),
                    ),
                    allocation=FlowAllocation(
                        current_slice_snssai=current_slice_snssai or None,
                        allocated_bandwidth_ul=float(flow_payload.get("allocated_bandwidth_ul_mbps", 0.0) or 0.0),
                        allocated_bandwidth_dl=float(flow_payload.get("allocated_bandwidth_dl_mbps", 0.0) or 0.0),
                        optimize_requested=bool(flow_payload.get("optimize_requested", False)),
                    ),
                )
            )
        apps.append(
            App(
                id=_normalize_or_generate_id(raw_app_id, "app", used_suffixes),
                name=app_name,
                flows=flow_objects,
                supi=app_supi,
            )
        )

    return apps


def _create_scenario_from_yaml(path: Path) -> Tuple[List[App], List[Slice], List[Node]]:
    payload = _load_yaml_payload(path)
    slices_payload = list(payload.get("slices") or [])
    gnbs_payload = list(payload.get("gnbs") or [])
    upfs_payload = list(payload.get("upfs") or [])
    ues_payload = list(payload.get("ues") or [])
    apps_payload = list(payload.get("apps") or [])
    flows_payload = list(payload.get("flows") or [])
    if not slices_payload:
        raise ValueError(f"Scenario YAML does not define any slices: {path}")
    if not gnbs_payload:
        raise ValueError(f"Scenario YAML does not define any gNBs: {path}")
    if not upfs_payload:
        raise ValueError(f"Scenario YAML does not define any UPFs: {path}")
    if not apps_payload and not flows_payload:
        raise ValueError(f"Scenario YAML does not define any apps/flows: {path}")

    slices, label_to_snssai, all_snssai = _parse_slice_snssai(slices_payload)
    nodes = _build_nodes_from_yaml(
        gnbs_payload=gnbs_payload,
        upfs_payload=upfs_payload,
        label_to_snssai=label_to_snssai,
        all_snssai=all_snssai,
    )
    apps = _build_apps_from_yaml(
        apps_payload=apps_payload,
        flows_payload=flows_payload,
        ues_payload=ues_payload,
        label_to_snssai=label_to_snssai,
    )
    return apps, slices, nodes


def deserialize_scenario_payload(payload: Union[str, Dict[str, Any]]) -> Optional[Tuple[List[App], List[Slice], List[Node]]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None

    if not isinstance(payload, dict):
        return None

    scenario = payload.get("scenario", payload)
    if not isinstance(scenario, dict):
        return None

    if not all(k in scenario for k in ["apps", "slices", "nodes"]):
        return None

    return _deserialize_scenario(scenario)


