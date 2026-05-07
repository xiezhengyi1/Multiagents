import re
import secrets
import json
from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union, Dict, Any, Optional, Set

from .models import (
    App,
    Flow,
    FlowAllocation,
    FlowService,
    FlowSLA,
    FlowTelemetry,
    FlowTraffic,
    OptimizationConfig,
    AMPolicyState,
)
from .engine import SliceOptimizationEngine
from ..scenario.yaml_loader import deserialize_scenario_payload
from ..storage.session_store import get_snapshot_data_by_id
from shared.logging import setup_logger

logger = setup_logger(__name__)

_OPTIMIZER_TRACE_PATH = (
    Path(__file__).resolve().parents[4]
    / "training"
    / "optimizer"
    / "raw_traces"
    / "optimizer.jsonl"
)


def _is_executable_solver_status(status: Any) -> bool:
    text = str(status or "").strip().lower()
    return text.startswith("optimal")


def _write_optimizer_trace(payload: Dict[str, Any]) -> None:
    _OPTIMIZER_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OPTIMIZER_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_optimizer_scenario(snapshot_id: str) -> tuple[List[App], list[Any], list[Any]]:
    normalized_snapshot_id = str(snapshot_id or "").strip()
    if not normalized_snapshot_id:
        raise ValueError("optimizer requires snapshot_id")
    snapshot = get_snapshot_data_by_id(normalized_snapshot_id)
    if not isinstance(snapshot, dict) or not snapshot:
        raise LookupError(f"optimizer snapshot not found: snapshot_id={normalized_snapshot_id}")
    scenario = deserialize_scenario_payload(snapshot)
    if scenario is None:
        raise RuntimeError(f"optimizer snapshot is not a valid scenario payload: snapshot_id={normalized_snapshot_id}")
    return scenario


def _slice_trace_view(slices: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for slice_obj in slices:
        rows.append(
            {
                "name": slice_obj.name,
                "snssai": slice_obj.snssai,
                "sst": slice_obj.sst,
                "sd": slice_obj.sd,
                "capacity": {
                    "total_bandwidth_ul": slice_obj.capacity.total_bandwidth_ul,
                    "total_bandwidth_dl": slice_obj.capacity.total_bandwidth_dl,
                    "guaranteed_bandwidth_ul": slice_obj.capacity.guaranteed_bandwidth_ul,
                    "guaranteed_bandwidth_dl": slice_obj.capacity.guaranteed_bandwidth_dl,
                },
                "load": {
                    "current_bandwidth_ul": slice_obj.load.current_bandwidth_ul,
                    "current_bandwidth_dl": slice_obj.load.current_bandwidth_dl,
                },
                "qos": {
                    "latency": slice_obj.qos.latency,
                    "processing_delay": slice_obj.qos.processing_delay,
                    "jitter": slice_obj.qos.jitter,
                    "loss_rate": slice_obj.qos.loss_rate,
                },
            }
        )
    return rows


def _node_trace_view(nodes: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node in nodes:
        rows.append(
            {
                "id": node.id,
                "name": node.name,
                "node_type": node.node_type,
                "hosted_slice_snssais": list(node.hosted_slice_snssais or []),
                "capacity": {
                    "cpu": node.capacity.cpu,
                    "memory": node.capacity.memory,
                    "mec": node.capacity.mec,
                    "prb": node.capacity.prb,
                },
                "telemetry": {
                    "cpu_utilization": node.telemetry.cpu_utilization,
                    "mec_utilization": node.telemetry.mec_utilization,
                    "memory_utilization": node.telemetry.memory_utilization,
                    "prb_utilization": node.telemetry.prb_utilization,
                },
            }
        )
    return rows


def _flow_trace_view(flow: Flow) -> Dict[str, Any]:
    return {
        "flow_id": flow.id,
        "flow_name": flow.name,
        "service_type": flow.service.service_type,
        "service_type_id": flow.service.service_type_id,
        "current_slice_snssai": flow.allocation.current_slice_snssai,
        "allocated_bandwidth_ul": flow.allocation.allocated_bandwidth_ul,
        "allocated_bandwidth_dl": flow.allocation.allocated_bandwidth_dl,
        "optimize_requested": flow.allocation.optimize_requested,
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
    }


def _collect_frozen_flows(apps: List[App]) -> List[Dict[str, Any]]:
    frozen: List[Dict[str, Any]] = []
    for app in apps:
        for flow in app.flows:
            if flow.allocation.optimize_requested:
                continue
            if not str(flow.allocation.current_slice_snssai or "").strip():
                continue
            frozen.append(
                {
                    "app_id": app.id,
                    "app_name": app.name,
                    "supi": app.supi,
                    **_flow_trace_view(flow),
                }
            )
    return frozen


def _summarize_policy_state(debug_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context = debug_context or {}
    policy_state = context.get("policy_state")
    target_ues = list(context.get("target_ues") or [])
    if not isinstance(policy_state, dict) or not target_ues:
        return {}
    target_supi = str(target_ues[0] or "").strip()
    ue_ctx = policy_state.get(target_supi)
    if not isinstance(ue_ctx, dict):
        return {}
    am_ctx = (
        ue_ctx.get("amPolicyContext")
        or ue_ctx.get("am_policy")
        or ue_ctx.get("accessMobilityContext")
        or {}
    )
    if not isinstance(am_ctx, dict):
        return {}
    return {
        "supi": target_supi,
        "allowed_snssais": am_ctx.get("allowedSnssais") or am_ctx.get("allowed_snssais") or [],
        "target_snssais": am_ctx.get("targetSnssais") or am_ctx.get("target_snssais") or [],
        "rfsp": am_ctx.get("rfsp") or am_ctx.get("rfspIndex"),
        "triggers": am_ctx.get("triggers") or [],
    }


def _normalize_supi(raw_supi: Any) -> Optional[str]:
    if raw_supi is None:
        return None
    supi = str(raw_supi).strip()
    return supi or None


def _service_type_name_to_id(service_type: Any, default: int = 1) -> int:
    mapping = {
        "embb": 1,
        "urllc": 2,
        "mmtc": 3,
    }
    return mapping.get(str(service_type or "").strip().lower(), default)


_ID_FORMAT = {
    "app": re.compile(r"^app-(\d{4})$"),
    "flow": re.compile(r"^flow-(\d{4})$"),
}


def _extract_suffix(raw_id: Any, prefix: str) -> Optional[str]:
    if raw_id is None:
        return None
    match = _ID_FORMAT[prefix].fullmatch(str(raw_id).strip())
    if not match:
        return None
    return match.group(1)


def _collect_used_id_suffixes(apps: List[App]) -> set:
    used: set = set()
    for app in apps:
        app_suffix = _extract_suffix(app.id, "app")
        if app_suffix:
            used.add(app_suffix)
        for flow in app.flows:
            flow_suffix = _extract_suffix(flow.id, "flow")
            if flow_suffix:
                used.add(flow_suffix)
    return used


def _allocate_unique_suffix(used_suffixes: set) -> str:
    for _ in range(1000):
        suffix = f"{secrets.randbelow(10000):04d}"
        if suffix not in used_suffixes:
            used_suffixes.add(suffix)
            return suffix
    raise RuntimeError("无法分配唯一ID后缀，请检查ID空间使用情况")


def _normalize_or_generate_id(raw_id: Any, prefix: str, used_suffixes: set) -> str:
    pattern = _ID_FORMAT[prefix]
    if raw_id is not None:
        candidate = str(raw_id).strip()
        match = pattern.fullmatch(candidate)
        if match:
            suffix = match.group(1)
            if suffix not in used_suffixes:
                used_suffixes.add(suffix)
                return candidate

    suffix = _allocate_unique_suffix(used_suffixes)
    return f"{prefix}-{suffix}"


def _normalize_flow_list(raw_flows: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_flows, list):
        raise TypeError("new_app_data['flows'] must be a list of flow dictionaries")

    normalized_flows: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_flows):
        if not isinstance(item, dict):
            raise TypeError(f"flow at index {index} must be a dictionary")
        normalized_flows.append(dict(item))
    return normalized_flows


def _collapse_payload_to_target_payload(raw_payload: Any) -> Dict[str, Any]:
    if isinstance(raw_payload, list):
        payload: Dict[str, Any] = {"flows": _normalize_flow_list(raw_payload)}
    elif isinstance(raw_payload, dict):
        payload = dict(raw_payload)
        if "app_name" in payload and "name" not in payload:
            payload["name"] = payload["app_name"]
        payload["flows"] = _normalize_flow_list(payload.get("flows", []))
    else:
        raise TypeError("new_app_data must be a dict or a list of flow dictionaries")

    flows = payload["flows"]
    if not flows:
        raise ValueError("new_app_data must include at least one flow")

    target_groups: List[Dict[str, Any]] = []
    by_target: Dict[tuple[str, str], Dict[str, Any]] = {}
    for flow in flows:
        app_id = str(flow.get("app_id") or "").strip()
        app_name = str(flow.get("app_name") or flow.get("name") or "").strip()
        if not app_id and not app_name:
            raise ValueError("each flow in new_app_data must include app_id or app_name")
        target_key = (app_id, app_name)
        group = by_target.get(target_key)
        if group is None:
            group = {
                "app_id": app_id,
                "name": app_name,
                "supi": str(flow.get("supi") or "").strip(),
                "flows": [],
            }
            by_target[target_key] = group
            target_groups.append(group)
        elif not group.get("supi") and str(flow.get("supi") or "").strip():
            group["supi"] = str(flow.get("supi") or "").strip()
        group["flows"].append(flow)

    if not target_groups:
        raise ValueError("new_app_data must resolve to at least one target app")

    if len(target_groups) == 1:
        only_group = target_groups[0]
        if "app_id" not in payload and only_group.get("app_id"):
            payload["app_id"] = only_group["app_id"]
        if "supi" not in payload and only_group.get("supi"):
            payload["supi"] = only_group["supi"]
        if "name" not in payload and only_group.get("name"):
            payload["name"] = only_group["name"]
    payload["target_apps"] = target_groups

    return payload


def _clone_flow(flow: Flow) -> Flow:
    return Flow(
        id=flow.id,
        name=flow.name,
        service=FlowService(
            service_type=flow.service.service_type,
            service_type_id=flow.service.service_type_id,
        ),
        sla=FlowSLA(
            bandwidth_ul=flow.sla.bandwidth_ul,
            bandwidth_dl=flow.sla.bandwidth_dl,
            guaranteed_bandwidth_ul=flow.sla.guaranteed_bandwidth_ul,
            guaranteed_bandwidth_dl=flow.sla.guaranteed_bandwidth_dl,
            latency=flow.sla.latency,
            jitter=flow.sla.jitter,
            loss_rate=flow.sla.loss_rate,
            priority=flow.sla.priority,
        ),
        traffic=FlowTraffic(
            packet_size=flow.traffic.packet_size,
            arrival_rate=flow.traffic.arrival_rate,
            five_tuple=flow.traffic.five_tuple,
        ),
        allocation=FlowAllocation(
            current_slice_snssai=flow.allocation.current_slice_snssai,
            allocated_bandwidth_ul=flow.allocation.allocated_bandwidth_ul,
            allocated_bandwidth_dl=flow.allocation.allocated_bandwidth_dl,
            optimize_requested=flow.allocation.optimize_requested,
        ),
        telemetry=FlowTelemetry(
            throughput_ul=flow.telemetry.throughput_ul,
            throughput_dl=flow.telemetry.throughput_dl,
            latency=flow.telemetry.latency,
            jitter=flow.telemetry.jitter,
            loss_rate=flow.telemetry.loss_rate,
            packet_sent=flow.telemetry.packet_sent,
            packet_received=flow.telemetry.packet_received,
        ),
    )


def _build_flow_from_payload(flow_payload: Dict[str, Any], matched_old_flow: Optional[Flow], flow_id: str, *, index: int) -> Flow:
    service_type = flow_payload.get(
        "service_type",
        matched_old_flow.service.service_type if matched_old_flow else "eMBB",
    )
    service_type_id = _service_type_name_to_id(
        service_type,
        default=int(
            flow_payload.get(
                "service_type_id",
                matched_old_flow.service.service_type_id if matched_old_flow else 1,
            ) or 1
        ),
    )
    bw_ul_val = float(flow_payload.get("bw_ul", matched_old_flow.sla.bandwidth_ul if matched_old_flow else 0.0) or 0.0)
    bw_dl_val = float(flow_payload.get("bw_dl", matched_old_flow.sla.bandwidth_dl if matched_old_flow else 0.0) or 0.0)
    gbr_ul_val = float(
        flow_payload.get(
            "gbr_ul",
            matched_old_flow.sla.guaranteed_bandwidth_ul if matched_old_flow else bw_ul_val,
        ) or 0.0
    )
    gbr_dl_val = float(
        flow_payload.get(
            "gbr_dl",
            matched_old_flow.sla.guaranteed_bandwidth_dl if matched_old_flow else bw_dl_val,
        ) or 0.0
    )
    latency_val = float(flow_payload.get("lat", matched_old_flow.sla.latency if matched_old_flow else 100.0) or 0.0)
    loss_val = float(flow_payload.get("loss_req", matched_old_flow.sla.loss_rate if matched_old_flow else 0.05) or 0.0)
    jitter_val = float(flow_payload.get("jitter_req", matched_old_flow.sla.jitter if matched_old_flow else 50.0) or 0.0)
    priority_val = int(flow_payload.get("priority", matched_old_flow.sla.priority if matched_old_flow else 10) or 10)

    return Flow(
        id=flow_id,
        name=str(flow_payload.get("name") or (matched_old_flow.name if matched_old_flow else f"flow_{index}")),
        service=FlowService(
            service_type=str(service_type),
            service_type_id=service_type_id,
        ),
        sla=FlowSLA(
            bandwidth_ul=bw_ul_val,
            bandwidth_dl=bw_dl_val,
            guaranteed_bandwidth_ul=gbr_ul_val,
            guaranteed_bandwidth_dl=gbr_dl_val,
            latency=latency_val,
            jitter=jitter_val,
            loss_rate=loss_val,
            priority=priority_val,
        ),
        traffic=FlowTraffic(
            packet_size=float(flow_payload.get("packet_size", matched_old_flow.traffic.packet_size if matched_old_flow else 0.0) or 0.0),
            arrival_rate=float(flow_payload.get("arrival_rate", matched_old_flow.traffic.arrival_rate if matched_old_flow else 0.0) or 0.0),
            five_tuple=tuple(flow_payload["five_tuple"]) if isinstance(flow_payload.get("five_tuple"), (list, tuple)) else (matched_old_flow.traffic.five_tuple if matched_old_flow else None),
        ),
        allocation=FlowAllocation(
            current_slice_snssai=matched_old_flow.allocation.current_slice_snssai if matched_old_flow else None,
            allocated_bandwidth_ul=matched_old_flow.allocation.allocated_bandwidth_ul if matched_old_flow else None,
            allocated_bandwidth_dl=matched_old_flow.allocation.allocated_bandwidth_dl if matched_old_flow else None,
            optimize_requested=True,
        ),
        telemetry=FlowTelemetry(
            throughput_ul=matched_old_flow.telemetry.throughput_ul if matched_old_flow else None,
            throughput_dl=matched_old_flow.telemetry.throughput_dl if matched_old_flow else None,
            latency=matched_old_flow.telemetry.latency if matched_old_flow else None,
            jitter=matched_old_flow.telemetry.jitter if matched_old_flow else None,
            loss_rate=matched_old_flow.telemetry.loss_rate if matched_old_flow else None,
            packet_sent=matched_old_flow.telemetry.packet_sent if matched_old_flow else None,
            packet_received=matched_old_flow.telemetry.packet_received if matched_old_flow else None,
        ),
    )


def _serialize_flow_output(flow: Flow, *, strategy: Optional[str] = None) -> Dict[str, Any]:
    payload = asdict(flow)
    if strategy:
        payload["strategies"] = strategy
    return payload


def _build_target_app_summary(app: App, requested_flow_ids: Set[str], strategy_by_flow_id: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": app.id,
        "name": app.name,
        "supi": app.supi,
        "flows": [
            _serialize_flow_output(flow, strategy=strategy_by_flow_id.get(flow.id) or None)
            for flow in app.flows
            if flow.id in requested_flow_ids
        ],
    }


def _build_unassigned_flow_diagnostics(
    *,
    engine: SliceOptimizationEngine,
    apps: List[App],
    slices: List[Any],
    nodes: List[Any],
    target_apps: List[App],
    requested_flow_ids: Set[str],
) -> List[Dict[str, Any]]:
    modeled_baseline_loads = engine._slice_modeled_baseline_loads(apps)
    assigned_by_slice: Dict[str, Dict[str, float]] = {}
    for app in apps:
        for flow in app.flows:
            slice_snssai = str(flow.allocation.current_slice_snssai or "").strip()
            if not slice_snssai:
                continue
            bucket = assigned_by_slice.setdefault(slice_snssai, {"ul": 0.0, "dl": 0.0})
            bucket["ul"] += float(flow.allocation.allocated_bandwidth_ul or 0.0)
            bucket["dl"] += float(flow.allocation.allocated_bandwidth_dl or 0.0)

    slice_by_snssai = {slice_obj.snssai: slice_obj for slice_obj in slices}
    diagnostics: List[Dict[str, Any]] = []

    for app in target_apps:
        for flow in app.flows:
            if flow.id not in requested_flow_ids:
                continue
            if str(flow.allocation.current_slice_snssai or "").strip():
                continue

            candidate_slices: List[Dict[str, Any]] = []
            for slice_obj in slices:
                slice_latency, slice_processing_delay, slice_jitter, slice_loss = engine._slice_kpis_for_constraints(slice_obj)
                slice_load = assigned_by_slice.get(slice_obj.snssai, {"ul": 0.0, "dl": 0.0})
                background_ul, background_dl = engine._effective_background_load(slice_obj, modeled_baseline_loads)
                remaining_ul = float(slice_obj.capacity.total_bandwidth_ul or 0.0) - background_ul - float(slice_load["ul"] or 0.0)
                remaining_dl = float(slice_obj.capacity.total_bandwidth_dl or 0.0) - background_dl - float(slice_load["dl"] or 0.0)

                violations: List[Dict[str, Any]] = []
                if slice_latency + slice_processing_delay > flow.sla.latency:
                    violations.append(
                        {
                            "constraint": "latency_bound",
                            "required_max": float(flow.sla.latency or 0.0),
                            "actual": round(slice_latency + slice_processing_delay, 6),
                        }
                    )
                if flow.sla.jitter > 0 and slice_jitter > flow.sla.jitter:
                    violations.append(
                        {
                            "constraint": "jitter_bound",
                            "required_max": float(flow.sla.jitter or 0.0),
                            "actual": round(slice_jitter, 6),
                        }
                    )
                if flow.sla.loss_rate > 0 and slice_loss > flow.sla.loss_rate:
                    violations.append(
                        {
                            "constraint": "loss_bound",
                            "required_max": float(flow.sla.loss_rate or 0.0),
                            "actual": round(slice_loss, 9),
                        }
                    )
                if remaining_ul + 1e-9 < float(flow.sla.bandwidth_ul or 0.0):
                    violations.append(
                        {
                            "constraint": "slice_capacity_ul",
                            "required_min": float(flow.sla.bandwidth_ul or 0.0),
                            "actual": round(max(remaining_ul, 0.0), 6),
                        }
                    )
                if remaining_dl + 1e-9 < float(flow.sla.bandwidth_dl or 0.0):
                    violations.append(
                        {
                            "constraint": "slice_capacity_dl",
                            "required_min": float(flow.sla.bandwidth_dl or 0.0),
                            "actual": round(max(remaining_dl, 0.0), 6),
                        }
                    )

                flow_total_traffic = float(flow.sla.bandwidth_ul or 0.0) + float(flow.sla.bandwidth_dl or 0.0)
                hosted_nodes = [
                    node for node in nodes
                    if slice_obj.snssai in set(node.hosted_slice_snssais or [])
                    or slice_obj.name in set(node.hosted_slice_snssais or [])
                ]
                node_violations: List[Dict[str, Any]] = []
                for node in hosted_nodes:
                    hosted_set = set(node.hosted_slice_snssais or [])
                    hosted_snssais = [
                        current_slice.snssai
                        for current_slice in slices
                        if current_slice.snssai in hosted_set or current_slice.name in hosted_set
                    ]
                    existing_traffic = 0.0
                    for hosted_snssai in hosted_snssais:
                        load = assigned_by_slice.get(hosted_snssai, {"ul": 0.0, "dl": 0.0})
                        existing_traffic += float(load["ul"] or 0.0) + float(load["dl"] or 0.0)

                    if node.node_type == "CN":
                        available_cpu = float(node.capacity.cpu or 0.0) - float(node.telemetry.cpu_utilization or 0.0) * float(node.capacity.cpu or 0.0) - existing_traffic * float(engine.config.alpha_cn or 0.0)
                        required_cpu = flow_total_traffic * float(engine.config.alpha_cn or 0.0)
                        if available_cpu + 1e-9 < required_cpu:
                            node_violations.append(
                                {
                                    "constraint": "cn_cpu",
                                    "node": node.name,
                                    "required_min": round(required_cpu, 6),
                                    "actual": round(max(available_cpu, 0.0), 6),
                                }
                            )
                    if node.node_type == "AN":
                        available_cpu = float(node.capacity.cpu or 0.0) - float(node.telemetry.cpu_utilization or 0.0) * float(node.capacity.cpu or 0.0) - existing_traffic * float(engine.config.alpha_an or 0.0)
                        required_cpu = flow_total_traffic * float(engine.config.alpha_an or 0.0)
                        if available_cpu + 1e-9 < required_cpu:
                            node_violations.append(
                                {
                                    "constraint": "an_cpu",
                                    "node": node.name,
                                    "required_min": round(required_cpu, 6),
                                    "actual": round(max(available_cpu, 0.0), 6),
                                }
                            )
                        available_prb = float(node.capacity.prb or 0.0) - float(node.telemetry.prb_utilization or 0.0) * float(node.capacity.prb or 0.0) - existing_traffic * float(engine.config.prb or 0.0)
                        required_prb = flow_total_traffic * float(engine.config.prb or 0.0)
                        if available_prb + 1e-9 < required_prb:
                            node_violations.append(
                                {
                                    "constraint": "an_prb",
                                    "node": node.name,
                                    "required_min": round(required_prb, 6),
                                    "actual": round(max(available_prb, 0.0), 6),
                                }
                            )
                    if float(node.capacity.mec or 0.0) > 0:
                        available_mec = (1 - float(node.telemetry.mec_utilization or 0.0)) * float(node.capacity.mec or 0.0) - existing_traffic * float(engine._mec_overhead_by_sst(slice_obj.sst) or 0.0)
                        required_mec = flow_total_traffic * float(engine._mec_overhead_by_sst(slice_obj.sst) or 0.0)
                        if available_mec + 1e-9 < required_mec:
                            node_violations.append(
                                {
                                    "constraint": "mec_capacity",
                                    "node": node.name,
                                    "required_min": round(required_mec, 6),
                                    "actual": round(max(available_mec, 0.0), 6),
                                }
                            )

                candidate_slices.append(
                    {
                        "snssai": slice_obj.snssai,
                        "name": slice_obj.name,
                        "sst": slice_obj.sst,
                        "capacity_snapshot": {
                            "remaining_ul_mbps": round(max(remaining_ul, 0.0), 6),
                            "remaining_dl_mbps": round(max(remaining_dl, 0.0), 6),
                        },
                        "kpi_snapshot": {
                            "latency_ms": round(slice_latency + slice_processing_delay, 6),
                            "jitter_ms": round(slice_jitter, 6),
                            "loss_rate": round(slice_loss, 9),
                        },
                        "violations": violations,
                        "node_violations": node_violations,
                        "feasible": not violations and not node_violations,
                    }
                )

            diagnostics.append(
                {
                    "flow_id": flow.id,
                    "app_id": app.id,
                    "app_name": app.name,
                    "flow_name": flow.name,
                    "requested_sla": {
                        "bandwidth_ul_mbps": float(flow.sla.bandwidth_ul or 0.0),
                        "bandwidth_dl_mbps": float(flow.sla.bandwidth_dl or 0.0),
                        "gbr_ul_mbps": float(flow.sla.guaranteed_bandwidth_ul or 0.0),
                        "gbr_dl_mbps": float(flow.sla.guaranteed_bandwidth_dl or 0.0),
                        "latency_ms": float(flow.sla.latency or 0.0),
                        "jitter_ms": float(flow.sla.jitter or 0.0),
                        "loss_rate": float(flow.sla.loss_rate or 0.0),
                        "priority": int(flow.sla.priority or 0),
                    },
                    "summary": "no candidate slice satisfies all hard QoS/resource constraints",
                    "candidate_slices": candidate_slices,
                }
            )

    return diagnostics


def optimize_network_slices(
    new_app_data: Union[dict, List[dict]],
    w1: float,
    w2: float,
    w3: float,
    w4: float = 0.0,
    mode: str = "full",
    return_json: bool = True,
    *,
    am_policy_state: Optional[AMPolicyState] = None,
    mobility_risk_weight: float = 0.0,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Execute slice optimization for one or more target apps.

    Accepted input shapes:
    - {"app_id": ..., "name": ..., "supi": ..., "flows": [...]}
    - [{flow...}, {flow...}, ...] for one or more apps, grouped by flow-level app_id/app_name
    """
    trace_record: Dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "session_id": str((debug_context or {}).get("session_id") or ""),
        "snapshot_id": str((debug_context or {}).get("snapshot_id") or ""),
        "requested_domains": list((debug_context or {}).get("requested_domains") or []),
        "target_ues": list((debug_context or {}).get("target_ues") or []),
        "mobility_policy_summary": _summarize_policy_state(debug_context),
        "slice_kpi_source": str((debug_context or {}).get("slice_kpi_source") or "qos"),
        "qos_relaxation_ratio": float((debug_context or {}).get("qos_relaxation_ratio") or 0.2),
        "solver_mode": mode,
        "weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
        "mobility_risk_weight": mobility_risk_weight,
        "am_policy_state": (
            {
                "old_allowed_snssais": list(am_policy_state.old_allowed_snssais or []),
                "old_target_snssais": list(am_policy_state.old_target_snssais or []),
                "old_rfsp": am_policy_state.old_rfsp,
                "old_triggers": list(am_policy_state.old_triggers or []),
                "old_ue_ambr_ul": am_policy_state.old_ue_ambr_ul,
                "old_ue_ambr_dl": am_policy_state.old_ue_ambr_dl,
                "mobility_risk_score": am_policy_state.mobility_risk_score,
            }
            if am_policy_state is not None
            else None
        ),
    }
    try:
        # Operate on a detached copy so optimizer previews never mutate the shared
        # in-memory baseline before the workflow is actually committed.
        apps, slices, nodes = deepcopy(_load_optimizer_scenario(trace_record["snapshot_id"]))
        used_suffixes = _collect_used_id_suffixes(apps)
        normalized_payload = _collapse_payload_to_target_payload(new_app_data)
        requested_flows_payload = normalized_payload["flows"]
        target_app_payloads = list(normalized_payload.get("target_apps") or [])

        requested_flow_ids: Set[str] = set()
        target_apps: List[App] = []
        target_app_ids: List[str] = []

        for app_index, target_app_payload in enumerate(target_app_payloads):
            target_app_id = target_app_payload.get("app_id")
            target_app_name = target_app_payload.get("name")
            existing_app = next(
                (app for app in apps if (target_app_id and app.id == target_app_id) or (target_app_name and app.name == target_app_name)),
                None,
            )
            if existing_app is not None:
                final_app_id = existing_app.id
            else:
                if not target_app_id and not target_app_name:
                    raise ValueError("new app payload must include app_id or app_name when the target app does not already exist")
                final_app_id = _normalize_or_generate_id(target_app_id, "app", used_suffixes)

            if not target_app_name:
                target_app_name = existing_app.name if existing_app is not None else f"NewApp{app_index + 1}"

            app_supi = _normalize_supi(target_app_payload.get("supi"))
            if app_supi is None and existing_app is not None:
                app_supi = _normalize_supi(existing_app.supi)

            existing_flow_map: Dict[str, Flow] = {}
            if existing_app is not None:
                for old_flow in existing_app.flows:
                    existing_flow_map[old_flow.id] = old_flow

            merged_flows: List[Flow] = []
            matched_existing_flow_ids: Set[str] = set()

            for flow_index, flow_payload in enumerate(target_app_payload.get("flows") or []):
                raw_fid = str(flow_payload.get("flow_id") or "").strip() or None
                matched_old_flow = existing_flow_map.get(raw_fid) if raw_fid is not None else None

                if matched_old_flow is not None:
                    current_f_id = matched_old_flow.id
                    matched_existing_flow_ids.add(matched_old_flow.id)
                else:
                    current_f_id = _normalize_or_generate_id(raw_fid, "flow", used_suffixes)
                requested_flow_ids.add(current_f_id)

                merged_flows.append(
                    _build_flow_from_payload(
                        flow_payload,
                        matched_old_flow,
                        current_f_id,
                        index=flow_index,
                    )
                )

            if existing_app is not None:
                for old_flow in existing_app.flows:
                    if old_flow.id not in matched_existing_flow_ids:
                        preserved_flow = _clone_flow(old_flow)
                        preserved_flow.allocation.optimize_requested = False
                        merged_flows.append(preserved_flow)

            target_apps.append(
                App(
                    id=final_app_id,
                    name=target_app_name,
                    supi=app_supi,
                    flows=merged_flows,
                )
            )
            target_app_ids.append(final_app_id)

        config = OptimizationConfig(
            w1=w1, w2=w2, w3=w3, w4=w4,
            w8=max(0.0, float(mobility_risk_weight or 0.0) * 20.0),
            enable_am_optimization=am_policy_state is not None,
            am_policy_state=am_policy_state,
            qos_relaxation_ratio=float((debug_context or {}).get("qos_relaxation_ratio") or 0.2),
            slice_kpi_source=str((debug_context or {}).get("slice_kpi_source") or "qos"),
        )
        engine = SliceOptimizationEngine(config)

        active_apps = [app for app in apps if app.id not in set(target_app_ids)]
        updated_apps_list = active_apps + target_apps
        trace_record["target_apps"] = [
            {
                "app_id": target_app.id,
                "app_name": target_app.name,
                "supi": target_app.supi,
                "requested_flows": [
                    _flow_trace_view(flow)
                    for flow in target_app.flows
                    if flow.id in requested_flow_ids
                ],
                "preserved_flows": [
                    _flow_trace_view(flow)
                    for flow in target_app.flows
                    if flow.id not in requested_flow_ids
                ],
            }
            for target_app in target_apps
        ]
        trace_record["target_app"] = (
            trace_record["target_apps"][0]
            if len(trace_record["target_apps"]) == 1
            else {
                "app_id": "",
                "app_name": "MULTI_APP_TARGET",
                "supi": "",
                "requested_flows": [
                    flow
                    for app_payload in trace_record["target_apps"]
                    for flow in app_payload["requested_flows"]
                ],
                "preserved_flows": [
                    flow
                    for app_payload in trace_record["target_apps"]
                    for flow in app_payload["preserved_flows"]
                ],
            }
        )
        trace_record["frozen_flows"] = _collect_frozen_flows(updated_apps_list)
        trace_record["slice_state"] = _slice_trace_view(slices)
        trace_record["node_state"] = _node_trace_view(nodes)

        if mode == "incremental":
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve_incremental(updated_apps_list, slices, nodes)
        elif mode == "hybrid":
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve_hybrid(updated_apps_list, slices, nodes)
        else:
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve(updated_apps_list, slices, nodes)
        trace_record["solver_result"] = {
            "status": status_str,
            "objective_value": objective_val,
            "breakdown": breakdown,
            "result_rows": results_df.to_dict(orient="records") if not results_df.empty else [],
            "slice_stats": slice_stats_df.to_dict(orient="records") if not slice_stats_df.empty else [],
        }

        if not _is_executable_solver_status(status_str):
            _write_optimizer_trace(trace_record)
            return (
                {"error": f"Infeasible solver result: {status_str}"}
                if return_json
                else f"求解器未得到可执行最优解: {status_str}"
            )

        if results_df.empty:
            _write_optimizer_trace(trace_record)
            return (
                {"error": f"Solver returned no executable result rows (status={status_str})"}
                if return_json
                else f"求解器未返回可执行结果 (status={status_str})。"
            )

        strategy_by_flow_id: Dict[str, str] = {}
        for _, row in results_df.iterrows():
            r_app_id = row["App ID"]
            r_flow_id = row["Flow ID"]
            r_new_slice = row["New Slice"]
            r_act_bw_ul = row["Act BW UL"]
            r_act_bw_dl = row["Act BW DL"]
            strategy_by_flow_id[str(r_flow_id)] = str(row.get("Strategies") or "")

            cached_app = next((app for app in updated_apps_list if app.id == r_app_id), None)
            if cached_app is None:
                continue
            cached_flow = next((flow for flow in cached_app.flows if flow.id == r_flow_id), None)
            if cached_flow is None:
                continue
            cached_flow.allocation.current_slice_snssai = r_new_slice
            cached_flow.allocation.allocated_bandwidth_ul = float(r_act_bw_ul or 0.0)
            cached_flow.allocation.allocated_bandwidth_dl = float(r_act_bw_dl or 0.0)
            cached_flow.telemetry.throughput_ul = float(r_act_bw_ul or 0.0)
            cached_flow.telemetry.throughput_dl = float(r_act_bw_dl or 0.0)
            cached_flow.telemetry.latency = max(0.1, float(cached_flow.sla.latency or 0.0) * 0.85)
            cached_flow.telemetry.jitter = max(0.1, float(cached_flow.sla.jitter or 0.0) * 0.8)
            cached_flow.telemetry.loss_rate = max(0.0, float(cached_flow.sla.loss_rate or 0.0) * 0.75)

        slice_status = []
        target_ssts = {
            _service_type_name_to_id(flow.get("service_type"), default=int(flow.get("service_type_id", 1) or 1))
            for flow in requested_flows_payload
        } or {1}
        for slice_row in slice_stats_df.to_dict(orient="records"):
            try:
                slice_sst = int(str(slice_row.get("SNSSAI", ""))[:2], 16)
            except Exception:
                slice_sst = None
            if slice_sst in target_ssts:
                slice_status.append(slice_row)

        target_apps_after = [app for app in updated_apps_list if app.id in set(target_app_ids)]
        target_app_outputs = [
            _build_target_app_summary(app, requested_flow_ids, strategy_by_flow_id)
            for app in target_apps_after
        ]
        unassigned_flow_diagnostics = _build_unassigned_flow_diagnostics(
            engine=engine,
            apps=updated_apps_list,
            slices=slices,
            nodes=nodes,
            target_apps=target_apps_after,
            requested_flow_ids=requested_flow_ids,
        )
        aggregate_target_app = (
            target_app_outputs[0]
            if len(target_app_outputs) == 1
            else {
                "id": "",
                "name": "MULTI_APP_TARGET",
                "supi": "",
                "flows": [
                    flow
                    for app_payload in target_app_outputs
                    for flow in app_payload["flows"]
                ],
            }
        )
        impacted_flow_outputs = [
            {
                "app_id": app.id,
                "app_name": app.name,
                "supi": app.supi,
                "flow": _serialize_flow_output(flow, strategy=strategy_by_flow_id.get(flow.id) or None),
            }
            for app in updated_apps_list
            for flow in app.flows
            if flow.id not in requested_flow_ids and strategy_by_flow_id.get(flow.id) and strategy_by_flow_id.get(flow.id) != "保持"
        ]

        if return_json:
            response_payload = {
                "meta": {
                    "status": status_str,
                    "objective_value": float(objective_val) if objective_val is not None else None,
                    "mode": mode,
                    "params": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
                    "breakdown": breakdown,
                    "infeasibility_details": unassigned_flow_diagnostics,
                },
                "target_app": aggregate_target_app,
                "target_apps": target_app_outputs,
                "impacted_flows": impacted_flow_outputs,
                "slice_stats": slice_status,
            }
            trace_record["response_meta"] = response_payload["meta"]
            _write_optimizer_trace(trace_record)
            return response_payload

        output = []
        output.append("--- Optimization Report (Flow Level) ---")
        output.append(f"Params: w1={w1}, w2={w2}, w3={w3}, w4={w4}")
        output.append(f"Mode: {mode}")
        output.append(f"Status: {status_str}")
        output.append(f"Objective: {objective_val if objective_val is not None else 'N/A'}")
        if breakdown:
            output.append(
                "Breakdown: "
                f"load={breakdown.get('load_norm', 0):.6f}, "
                f"signal={breakdown.get('signal_norm', 0):.6f}, "
                f"exp={breakdown.get('exp', 0):.6f}, "
                f"qos_core={breakdown.get('qos_core', 0):.6f}, "
                f"qos_aux={breakdown.get('qos_aux', 0):.6f}"
            )

        my_results = results_df[results_df["Flow ID"].isin(requested_flow_ids)]
        if not my_results.empty:
            for target_app in target_apps_after:
                app_results = my_results[my_results["App ID"] == target_app.id]
                if app_results.empty:
                    continue
                output.append(f"Target app '{target_app.name}' (ID: {target_app.id}):")
                for _, row in app_results.iterrows():
                    strategy_note = f", strategy: {row['Strategies']}" if row["Strategies"] != "保持" else ""
                    output.append(
                        f"  - flow [{row['Flow Name']}] (ID: {row['Flow ID']}) -> slice: {row['New Slice'] or 'unassigned'} "
                        f"(UL: {row['Act BW UL']}/{row['Req BW UL']}M, DL: {row['Act BW DL']}/{row['Req BW DL']}M{strategy_note})"
                    )

        impacted_df = results_df[(~results_df["Flow ID"].isin(requested_flow_ids)) & (results_df["Strategies"] != "保持")]
        if not impacted_df.empty:
            output.append("Impacted flows:")
            for _, row in impacted_df.iterrows():
                output.append(
                    f"  - {row['App']} / [{row['Flow Name']}] -> {row['New Slice']} "
                    f"(strategy: {row['Strategies']})"
                )

        output.append("Slice stats:")
        for _, row in slice_stats_df.iterrows():
            output.append(
                f"  - {row['Slice']} ({row['SNSSAI']}): UL load {row['Load UL (%)']}% / DL load {row['Load DL (%)']}%"
            )
        trace_record["response_meta"] = {
            "status": status_str,
            "objective_value": float(objective_val) if objective_val is not None else None,
            "mode": mode,
            "params": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
            "breakdown": breakdown,
        }
        _write_optimizer_trace(trace_record)
        return "\n".join(output)
    except Exception as e:
        trace_record["exception"] = {"type": type(e).__name__, "message": str(e)}
        _write_optimizer_trace(trace_record)
        logger.error(f"优化过程发生异常: {e}", exc_info=True)
        return {"error": str(e)} if return_json else f"系统错误: 优化求解失败 - {str(e)}"
