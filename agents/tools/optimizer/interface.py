import re
import secrets
from dataclasses import asdict
from copy import deepcopy
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
from agents.tools.init_scenario import (
    get_current_scenario,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


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


def _collapse_payload_to_app_dict(raw_payload: Any) -> Dict[str, Any]:
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

    flow_app_ids = {str(flow.get("app_id") or "").strip() for flow in flows if str(flow.get("app_id") or "").strip()}
    if len(flow_app_ids) > 1:
        raise ValueError("all flows in one optimization request must belong to the same app_id")
    if "app_id" not in payload and flow_app_ids:
        payload["app_id"] = next(iter(flow_app_ids))

    flow_supis = {str(flow.get("supi") or "").strip() for flow in flows if str(flow.get("supi") or "").strip()}
    if len(flow_supis) > 1:
        raise ValueError("all flows in one optimization request must belong to the same supi")
    if "supi" not in payload and flow_supis:
        payload["supi"] = next(iter(flow_supis))

    flow_app_names = {str(flow.get("app_name") or "").strip() for flow in flows if str(flow.get("app_name") or "").strip()}
    if len(flow_app_names) > 1:
        raise ValueError("all flows in one optimization request must belong to the same app_name")
    if "name" not in payload and flow_app_names:
        payload["name"] = next(iter(flow_app_names))

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
) -> Union[str, Dict[str, Any]]:
    """
    Execute slice optimization for one target app.

    Accepted input shapes:
    - {"app_id": ..., "name": ..., "supi": ..., "flows": [...]}
    - [{flow...}, {flow...}, ...] for multiple flows under one app
    """
    try:
        # Operate on a detached copy so optimizer previews never mutate the shared
        # in-memory baseline before the workflow is actually committed.
        apps, slices, nodes = deepcopy(get_current_scenario())
        used_suffixes = _collect_used_id_suffixes(apps)
        normalized_payload = _collapse_payload_to_app_dict(new_app_data)
        requested_flows_payload = normalized_payload["flows"]

        target_app_id = normalized_payload.get("app_id")
        target_app_name = normalized_payload.get("name")

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
            target_app_name = existing_app.name if existing_app is not None else "NewApp"

        app_supi = _normalize_supi(normalized_payload.get("supi"))
        if app_supi is None and existing_app is not None:
            app_supi = _normalize_supi(existing_app.supi)

        existing_flow_map: Dict[str, Flow] = {}
        if existing_app is not None:
            for old_flow in existing_app.flows:
                existing_flow_map[old_flow.id] = old_flow

        merged_flows: List[Flow] = []
        requested_flow_ids: Set[str] = set()
        matched_existing_flow_ids: Set[str] = set()

        for index, flow_payload in enumerate(requested_flows_payload):
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
                    index=index,
                )
            )

        if existing_app is not None:
            for old_flow in existing_app.flows:
                if old_flow.id not in matched_existing_flow_ids:
                    preserved_flow = _clone_flow(old_flow)
                    preserved_flow.allocation.optimize_requested = False
                    merged_flows.append(preserved_flow)

        target_app = App(
            id=final_app_id,
            name=target_app_name,
            supi=app_supi,
            flows=merged_flows,
        )

        config = OptimizationConfig(
            w1=w1, w2=w2, w3=w3, w4=w4,
            w8=max(0.0, float(mobility_risk_weight or 0.0) * 20.0),
            enable_am_optimization=am_policy_state is not None,
            am_policy_state=am_policy_state,
        )
        engine = SliceOptimizationEngine(config)

        if existing_app is not None:
            active_apps = [app for app in apps if app.id != existing_app.id]
        else:
            active_apps = list(apps)
        updated_apps_list = active_apps + [target_app]

        if mode == "incremental":
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve_incremental(updated_apps_list, slices, nodes)
        elif mode == "hybrid":
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve_hybrid(updated_apps_list, slices, nodes)
        else:
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve(updated_apps_list, slices, nodes)

        if results_df.empty:
            return {"error": "Empty Result"} if return_json else "求解器未返回结果 (Empty Result)。"

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

        target_app_after = next(app for app in updated_apps_list if app.id == final_app_id)
        target_flow_outputs = [
            _serialize_flow_output(flow, strategy=strategy_by_flow_id.get(flow.id) or None)
            for flow in target_app_after.flows
            if flow.id in requested_flow_ids
        ]
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
            return {
                "meta": {
                    "status": status_str,
                    "objective_value": float(objective_val) if objective_val is not None else None,
                    "mode": mode,
                    "params": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
                    "breakdown": breakdown,
                },
                "target_app": {
                    "id": final_app_id,
                    "name": target_app_name,
                    "supi": app_supi,
                    "flows": target_flow_outputs,
                },
                "impacted_flows": impacted_flow_outputs,
                "slice_stats": slice_status,
            }

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

        my_results = results_df[(results_df["App ID"] == final_app_id) & results_df["Flow ID"].isin(requested_flow_ids)]
        if not my_results.empty:
            output.append(f"Target app '{target_app_name}' (ID: {final_app_id}):")
            for _, row in my_results.iterrows():
                strategy_note = f", strategy: {row['Strategies']}" if row["Strategies"] != "保持" else ""
                output.append(
                    f"  - flow [{row['Flow Name']}] (ID: {row['Flow ID']}) -> slice: {row['New Slice'] or 'unassigned'} "
                    f"(UL: {row['Act BW UL']}/{row['Req BW UL']}M, DL: {row['Act BW DL']}/{row['Req BW DL']}M{strategy_note})"
                )

        impacted_df = results_df[(results_df["App ID"] != final_app_id) & (results_df["Strategies"] != "保持")]
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
        return "\n".join(output)
    except Exception as e:
        logger.error(f"优化过程发生异常: {e}", exc_info=True)
        return {"error": str(e)} if return_json else f"系统错误: 优化求解失败 - {str(e)}"
