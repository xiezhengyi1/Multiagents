import re
import secrets
from typing import List, Union, Dict, Any, Optional, Set
from .models import App, Flow, OptimizationConfig, AMPolicyState, ServiceType, SLAProfile, Link, Path, Node
from .engine import SliceOptimizationEngine, IBNSOptimizationEngine
from agents.tools.init_scenario import (
    cache_scenario,
    get_cached_control_scenario,
    get_current_scenario,
    serialize_scenario_for_api,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _to_model_list(data_list: List[dict], model_cls, extra_transform=None) -> List[Any]:
    """关键步骤：统一完成 dict -> dataclass 的字段过滤与实例化。"""
    result = []
    valid_keys = set(model_cls.__annotations__.keys())
    for item in data_list:
        filtered = {k: v for k, v in item.items() if k in valid_keys}
        if extra_transform:
            filtered = extra_transform(filtered)
        result.append(model_cls(**filtered))
    return result


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
    m = _ID_FORMAT[prefix].fullmatch(str(raw_id).strip())
    if not m:
        return None
    return m.group(1)


def _collect_used_id_suffixes(apps: List[App]) -> set:
    used: set = set()
    for app in apps:
        app_suffix = _extract_suffix(getattr(app, "app_id", None), "app")
        if app_suffix:
            used.add(app_suffix)
        for flow in getattr(app, "flows", []):
            flow_suffix = _extract_suffix(getattr(flow, "flow_id", None), "flow")
            if flow_suffix:
                used.add(flow_suffix)
    return used


def _discard_suffix(raw_id: Any, prefix: str, used_suffixes: set) -> None:
    suffix = _extract_suffix(raw_id, prefix)
    if suffix:
        used_suffixes.discard(suffix)


def _allocate_unique_suffix(used_suffixes: set) -> str:
    for _ in range(1000):
        suffix = f"{secrets.randbelow(10000):04d}"
        if suffix not in used_suffixes:
            used_suffixes.add(suffix)
            return suffix
    raise RuntimeError("无法分配唯一ID后缀，请检查ID空间使用情况")


def _normalize_or_generate_id(raw_id: Any, prefix: str, used_suffixes: set) -> str:
    """关键步骤：仅接受新ID格式；旧ID一律重生，并保证8位数字后缀唯一。"""
    pattern = _ID_FORMAT[prefix]
    if raw_id is not None:
        candidate = str(raw_id).strip()
        m = pattern.fullmatch(candidate)
        if m:
            suffix = m.group(1)
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

    flow_app_ids = {str(flow.get("app_id")).strip() for flow in flows if str(flow.get("app_id") or "").strip()}
    if len(flow_app_ids) > 1:
        raise ValueError("all flows in one optimization request must belong to the same app_id")
    if "app_id" not in payload and flow_app_ids:
        payload["app_id"] = next(iter(flow_app_ids))

    flow_supis = {str(flow.get("supi")).strip() for flow in flows if str(flow.get("supi") or "").strip()}
    if len(flow_supis) > 1:
        raise ValueError("all flows in one optimization request must belong to the same supi")
    if "supi" not in payload and flow_supis:
        payload["supi"] = next(iter(flow_supis))

    flow_app_names = {str(flow.get("app_name")).strip() for flow in flows if str(flow.get("app_name") or "").strip()}
    if len(flow_app_names) > 1:
        raise ValueError("all flows in one optimization request must belong to the same app_name")
    if "name" not in payload and flow_app_names:
        payload["name"] = next(iter(flow_app_names))

    return payload


def optimize_network_slices(new_app_data: Union[dict, List[dict]], w1: float, w2: float, w3: float, w4: float = 0.0, mode: str = 'full', return_json: bool = True, *, am_policy_state: Optional[AMPolicyState] = None) -> Union[str, Dict[str, Any]]:
    """
    Execute slice optimization for one target app.

    Accepted input shapes:
    - {"app_id": ..., "name": ..., "supi": ..., "flows": [...]}
    - [{flow...}, {flow...}, ...] for multiple flows under one app
    """
    try:
        apps, slices, nodes = get_current_scenario()
        used_suffixes = _collect_used_id_suffixes(apps)
        normalized_payload = _collapse_payload_to_app_dict(new_app_data)
        requested_flows_payload = normalized_payload["flows"]

        target_app_id = normalized_payload.get("app_id")
        target_app_name = normalized_payload.get("name")

        existing_app = next(
            (app for app in apps if (target_app_id and app.app_id == target_app_id) or (target_app_name and app.name == target_app_name)),
            None,
        )
        if existing_app is not None:
            final_app_id = existing_app.app_id
        else:
            if not target_app_id and not target_app_name:
                raise ValueError("new app payload must include app_id or app_name when the target app does not already exist")
            final_app_id = _normalize_or_generate_id(target_app_id, "app", used_suffixes)

        if not target_app_name:
            target_app_name = existing_app.name if existing_app is not None else "NewApp"

        app_supi = _normalize_supi(normalized_payload.get("supi"))
        if app_supi is None and existing_app is not None:
            app_supi = _normalize_supi(getattr(existing_app, "supi", None))

        existing_flow_map: Dict[str, Flow] = {}
        if existing_app is not None:
            for old_flow in existing_app.flows:
                raw_existing_fid = getattr(old_flow, "flow_id", None)
                if raw_existing_fid is not None:
                    existing_flow_map[str(raw_existing_fid).strip()] = old_flow

        merged_flows: List[Flow] = []
        requested_flow_ids: Set[str] = set()
        matched_existing_flow_ids: Set[str] = set()

        for index, flow_payload in enumerate(requested_flows_payload):
            raw_fid = str(flow_payload.get("flow_id") or "").strip() or None
            matched_old_flow = existing_flow_map.get(raw_fid) if raw_fid is not None else None

            if matched_old_flow is not None:
                current_f_id = matched_old_flow.flow_id
                matched_existing_flow_ids.add(matched_old_flow.flow_id)
            else:
                current_f_id = _normalize_or_generate_id(raw_fid, "flow", used_suffixes)
            requested_flow_ids.add(current_f_id)

            bw_ul_val = flow_payload.get("bw_ul", matched_old_flow.bw_ul if matched_old_flow else 0)
            bw_dl_val = flow_payload.get("bw_dl", matched_old_flow.bw_dl if matched_old_flow else 0)
            gbr_ul_val = flow_payload.get("gbr_ul", matched_old_flow.gbr_ul if matched_old_flow else bw_ul_val)
            gbr_dl_val = flow_payload.get("gbr_dl", matched_old_flow.gbr_dl if matched_old_flow else bw_dl_val)
            lat_val = flow_payload.get("lat", matched_old_flow.lat if matched_old_flow else 100)
            loss_val = flow_payload.get("loss_req", matched_old_flow.loss_req if matched_old_flow else 0.05)
            jitter_val = flow_payload.get("jitter_req", matched_old_flow.jitter_req if matched_old_flow else 50)
            priority_val = flow_payload.get("priority", matched_old_flow.priority if matched_old_flow else 10)
            service_type = flow_payload.get("service_type", matched_old_flow.service_type if matched_old_flow else "eMBB")
            service_type_id = _service_type_name_to_id(
                service_type,
                default=int(
                    flow_payload.get(
                        "service_type_id",
                        matched_old_flow.service_type_id if matched_old_flow else 1,
                    ) or 1
                ),
            )

            merged_flows.append(
                Flow(
                    name=flow_payload.get("name", f"flow_{index}"),
                    flow_id=current_f_id,
                    service_type=service_type,
                    bw_ul=bw_ul_val,
                    bw_dl=bw_dl_val,
                    gbr_ul=gbr_ul_val,
                    gbr_dl=gbr_dl_val,
                    lat=lat_val,
                    loss_req=loss_val,
                    jitter_req=jitter_val,
                    priority=priority_val,
                    service_type_id=service_type_id,
                    old_slice=matched_old_flow.old_slice if matched_old_flow else None,
                    old_allocated_bw_ul=matched_old_flow.old_allocated_bw_ul if matched_old_flow else None,
                    old_allocated_bw_dl=matched_old_flow.old_allocated_bw_dl if matched_old_flow else None,
                    optimize_requested=True,
                )
            )

        if existing_app is not None:
            for old_flow in existing_app.flows:
                if old_flow.flow_id not in matched_existing_flow_ids:
                    old_flow.optimize_requested = False
                    merged_flows.append(old_flow)

        target_app = App(
            name=target_app_name,
            app_id=final_app_id,
            supi=app_supi,
            flows=merged_flows,
        )

        # 关键步骤：配置优化参数，可选启用 AM 联合优化
        config = OptimizationConfig(
            w1=w1, w2=w2, w3=w3, w4=w4,
            enable_am_optimization=am_policy_state is not None,
            am_policy_state=am_policy_state,
        )
        engine = SliceOptimizationEngine(config)

        if existing_app is not None:
            active_apps = [app for app in apps if app.app_id != existing_app.app_id]
        else:
            active_apps = list(apps)
        updated_apps_list = active_apps + [target_app]

        if mode == 'incremental':
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve_incremental(updated_apps_list, slices, nodes)
        elif mode == 'hybrid':
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve_hybrid(updated_apps_list, slices, nodes)
        else:
            results_df, slice_stats_df, status_str, objective_val, breakdown = engine.solve(updated_apps_list, slices, nodes)

        if results_df.empty:
            return {"error": "Empty Result"} if return_json else "求解器未返回结果 (Empty Result)。"

        for _, row in results_df.iterrows():
            r_app_id = row["App ID"]
            r_flow_id = row["Flow ID"]
            r_new_slice = row["New Slice"]
            r_act_bw_ul = row["Act BW UL"]
            r_act_bw_dl = row["Act BW DL"]

            cached_app = next((app for app in updated_apps_list if app.app_id == r_app_id), None)
            if cached_app is None:
                continue
            cached_flow = next((flow for flow in cached_app.flows if flow.flow_id == r_flow_id), None)
            if cached_flow is None:
                continue
            cached_flow.old_slice = r_new_slice
            cached_flow.old_allocated_bw_ul = r_act_bw_ul
            cached_flow.old_allocated_bw_dl = r_act_bw_dl
            cached_flow.sim_throughput_ul = float(r_act_bw_ul or 0.0)
            cached_flow.sim_throughput_dl = float(r_act_bw_dl or 0.0)
            cached_flow.sim_latency = max(0.1, float(cached_flow.lat or 0.0) * 0.85)
            cached_flow.sim_jitter = max(0.1, float(cached_flow.jitter_req or 0.0) * 0.8)
            cached_flow.sim_loss_rate = max(0.0, float(cached_flow.loss_req or 0.0) * 0.75)

        cached_control = get_cached_control_scenario()
        cache_scenario(
            updated_apps_list,
            slices,
            nodes,
            cached_control.get("mobility") or [],
            cached_control.get("policy_state") or {},
        )

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

        is_target_mask = (results_df["App ID"] == final_app_id) & results_df["Flow ID"].isin(requested_flow_ids)
        target_flow_records = results_df[is_target_mask].to_dict(orient="records")
        impacted_records = results_df[(~is_target_mask) & (results_df["Strategies"] != "淇濇寔")].to_dict(orient="records")

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
                    "app_id": final_app_id,
                    "name": target_app_name,
                    "flows": target_flow_records,
                },
                "impacted_flows": impacted_records,
                "slice_stats": slice_status,
            }

        output = []
        output.append("--- Optimization Report (Flow Level) ---")
        output.append(f"Params: w1={w1}, w2={w2}, w3={w3}, w4={w4}")
        mode_map = {"full": "full", "incremental": "incremental", "hybrid": "hybrid"}
        output.append(f"Mode: {mode_map.get(mode, mode)}")
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

        my_results = results_df[is_target_mask]
        if not my_results.empty:
            output.append(f"Target app '{target_app_name}' (ID: {final_app_id}):")
            for _, row in my_results.iterrows():
                strategy_note = f", strategy: {row['Strategies']}" if row["Strategies"] != "淇濇寔" else ""
                output.append(
                    f"  - flow [{row['Flow Name']}] (ID: {row['Flow ID']}) -> slice: {row['New Slice'] or 'unassigned'} "
                    f"(UL: {row['Act BW UL']}/{row['Req BW UL']}M, DL: {row['Act BW DL']}/{row['Req BW DL']}M{strategy_note})"
                )

        impacted_df = results_df[(~is_target_mask) & (results_df["Strategies"] != "淇濇寔")]
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

def optimize_ibns_network(
    flows_data: List[dict],
    nodes_data: List[dict],
    links_data: List[dict],
    paths_data: List[dict],
    service_types_data: List[dict],
    sla_profiles_data: List[dict] = None,
    mode: str = 'p2',
    slices_data: List[dict] = None # Optional: Check existing slices first
) -> Dict[str, Any]:
    """
    Interface for IBNS Optimization (P1/P2).
    Accepts raw dictionaries, converts to internal models, and runs the engine.
    Also supports checking if flow can be accommodated by existing slices.
    """
    try:
        from .models import Slice
        used_suffixes: set = set()
        for flow_dict in flows_data:
            if not isinstance(flow_dict, dict):
                continue
            suffix = _extract_suffix(flow_dict.get("flow_id"), "flow")
            if suffix:
                used_suffixes.add(suffix)

        def _normalize_slice_payload(raw_slice: dict) -> dict:
            """关键步骤：兼容 snssai 输入并补全 Slice 构造字段。"""
            payload = dict(raw_slice)
            snssai = payload.get('snssai')
            if snssai and ('sst' not in payload or 'sd' not in payload):
                try:
                    payload['sst'] = int(snssai[:2], 16)
                    payload['sd'] = snssai[2:]
                except Exception:
                    # 无法从 snssai 解析时，交由后续 dataclass 校验报错
                    pass
            return payload

        def _effective_throughput_bps(flow_obj: Flow) -> float:
            """关键步骤：与 IBNS 引擎保持一致的吞吐口径。"""
            if flow_obj.packet_size > 0 and flow_obj.arrival_rate > 0:
                return float(flow_obj.packet_size * flow_obj.arrival_rate)

            r_ul = max(0.0, float(flow_obj.bw_ul) * 1e6)
            r_dl = max(0.0, float(flow_obj.bw_dl) * 1e6)
            if r_ul > 0 and r_dl > 0:
                return (2.0 * r_ul * r_dl) / (r_ul + r_dl)
            return max(r_ul, r_dl)

        def _build_ibns_slice_stats(slices_for_stats: List[Slice]) -> List[Dict[str, Any]]:
            stats = []
            for s in slices_for_stats:
                s_name = getattr(s, 'name', f"Slice_{s.snssai}")
                s_max_ul = getattr(s, 'total_bw_ul', getattr(s, 'max_bw_ul', 1000))
                s_max_dl = getattr(s, 'total_bw_dl', getattr(s, 'max_bw_dl', 1000))
                s_load_ul = getattr(s, 'current_load_bw_ul', 0)
                s_load_dl = getattr(s, 'current_load_bw_dl', 0)
                stats.append({
                    "Slice": s_name,
                    "SNSSAI": s.snssai,
                    "Load UL (%)": round((s_load_ul / s_max_ul * 100), 2) if s_max_ul > 0 else 0,
                    "Load DL (%)": round((s_load_dl / s_max_dl * 100), 2) if s_max_dl > 0 else 0,
                    "Rem UL (M)": round(s_max_ul - s_load_ul, 2),
                    "Rem DL (M)": round(s_max_dl - s_load_dl, 2)
                })
            return stats

        def _safe_float(val, default: float = 0.0) -> float:
            try:
                return float(val)
            except Exception:
                return float(default)

        def _merge_sla_kpis(base: Dict[str, float], incoming: Dict[str, float]) -> Dict[str, float]:
            """关键步骤：按语义合并KPI，latency取更小，throughput/reliability取更大。"""
            merged = dict(base or {})
            for k, v in (incoming or {}).items():
                if k not in merged:
                    merged[k] = v
                elif str(k) == "1":
                    merged[k] = min(_safe_float(merged[k], merged[k]), _safe_float(v, v))
                else:
                    merged[k] = max(_safe_float(merged[k], merged[k]), _safe_float(v, v))
            return merged

        def _flow_sla_from_ue_context(ue_ctx: Dict[str, Any], flow_id: str) -> Dict[str, float]:
            """关键步骤：从UE上下文提取flow级SLA并转换为IBNS KPI阈值。"""
            if not isinstance(ue_ctx, dict) or not flow_id:
                return {}

            kpis: Dict[str, float] = {}

            qos_decs = ue_ctx.get("qosDecs")
            if isinstance(qos_decs, dict):
                for _, qos_map in qos_decs.items():
                    if not isinstance(qos_map, dict):
                        continue
                    for qos_id, qos_obj in qos_map.items():
                        if not isinstance(qos_obj, dict):
                            continue
                        qos_id_str = str(qos_id)
                        if flow_id in qos_id_str or qos_id_str.endswith(str(flow_id)):
                            if "packetDelayBudget" in qos_obj:
                                kpis["1"] = _safe_float(qos_obj.get("packetDelayBudget"), 0.0)
                            if "packetErrorRate" in qos_obj:
                                per = _safe_float(qos_obj.get("packetErrorRate"), 0.0)
                                kpis["3"] = max(0.0, min(1.0, 1.0 - per))
                            if "gbrDl" in qos_obj:
                                kpis["2"] = _safe_float(qos_obj.get("gbrDl"), 0.0)
                            elif "maxbrDl" in qos_obj:
                                kpis["2"] = _safe_float(qos_obj.get("maxbrDl"), 0.0)

            sm_policy_data = ue_ctx.get("smPolicyData")
            if isinstance(sm_policy_data, dict):
                for _, sm_obj in sm_policy_data.items():
                    if not isinstance(sm_obj, dict):
                        continue
                    flow_sla = None
                    if isinstance(sm_obj.get("flow_sla_profiles"), dict):
                        flow_sla = sm_obj.get("flow_sla_profiles", {}).get(flow_id)
                    if flow_sla is None and isinstance(sm_obj.get(flow_id), dict):
                        flow_sla = sm_obj.get(flow_id, {}).get("sla_profile")
                    if isinstance(flow_sla, dict):
                        if "latency_ms" in flow_sla:
                            kpis["1"] = _safe_float(flow_sla.get("latency_ms"), kpis.get("1", 0.0))
                        if "throughput_mbps" in flow_sla:
                            kpis["2"] = _safe_float(flow_sla.get("throughput_mbps"), kpis.get("2", 0.0))
                        if "packet_loss_rate" in flow_sla:
                            loss = _safe_float(flow_sla.get("packet_loss_rate"), 0.0)
                            kpis["3"] = max(0.0, min(1.0, 1.0 - loss))

            return {k: v for k, v in kpis.items() if v > 0}

        # 关键步骤1：输入数据转为内部模型对象
        flow_supi_map: Dict[str, str] = {}
        for flow_dict in flows_data:
            if not isinstance(flow_dict, dict):
                continue
            flow_id = flow_dict.get('flow_id')
            supi = _normalize_supi(flow_dict.get('supi'))
            if flow_id and supi:
                flow_supi_map[str(flow_id)] = supi

        flows = _to_model_list(flows_data, Flow)
        
        # Optional: Load Existing Slices for pre-check
        existing_slices = []
        if slices_data:
            for s in slices_data:
                valid_keys = Slice.__annotations__.keys()
                normalized_s = _normalize_slice_payload(s)
                filtered_s = {k: v for k, v in normalized_s.items() if k in valid_keys}
                try:
                    obj = Slice(**filtered_s)
                    existing_slices.append(obj)
                except Exception as e:
                    logger.warning(f"Skipping invalid slice data: {s}, error: {e}")

        # Helper: Determine Strategy (Consistent with SliceOptimizationEngine)
        def _get_strategies(flow_obj, mapped_slice_snssai, act_ul, act_dl):
            if not flow_obj: return "未知"
            strategies = []
            TOL = 0.01
            
            # 策略B: 重路由
            # 注意: 如果是新业务(old_slice is None), 任何分配都会触发不等于判断。
            # 根据 engine.py 逻辑: if mapped_slice != flow.old_slice
            if mapped_slice_snssai != flow_obj.old_slice:
                strategies.append("策略B(重路由)")

            # 策略A: 拒绝/被抢占
            if act_ul < TOL and act_dl < TOL:
                strategies.append("策略A(拒绝/被抢占 UL)")

            # 策略C: 修改
            if flow_obj.old_allocated_bw_ul is not None and flow_obj.old_allocated_bw_dl is not None:
                if abs(act_ul - flow_obj.old_allocated_bw_ul) > TOL or abs(act_dl - flow_obj.old_allocated_bw_dl) > TOL:
                    strategies.append("策略C(修改)")

            # 策略D: 降级
            if act_ul < flow_obj.bw_ul - TOL and act_ul > TOL:
                strategies.append("策略D(降级 UL)")
            if act_dl < flow_obj.bw_dl - TOL and act_dl > TOL:
                strategies.append("策略D(降级 DL)")

            if not strategies:
                strategies.append("保持")
            return ", ".join(strategies)

        # --- Logic: Check if existing slices can accommodate flows ---
        flows_for_optimization = []
        pre_allocated_results = []
        
        # Map for quick access
        flow_map = {f.flow_id: f for f in flows}

        if existing_slices:
            logger.info(f"Checking {len(flows)} flows against {len(existing_slices)} existing slices...")
            for flow in flows:
                allocated = False
                for s in existing_slices:
                    # Check if Slice can accommodate Flow (Service Type Match + Resources)
                    # We assume Slice.sst matches Flow.service_type_id
                    if s.sst == flow.service_type_id:
                        if s.can_accommodate(flow):
                            # Allocate logically
                            s.current_load_bw_ul += flow.bw_ul
                            s.current_load_bw_dl += flow.bw_dl
                            allocated = True
                            
                            strat = _get_strategies(flow, s.snssai, flow.bw_ul, flow.bw_dl)
                            
                            pre_allocated_results.append({
                                "Flow ID": flow.flow_id,
                                "Service Type": f"ID-{flow.service_type_id}",
                                "Allocated AN": "Existing Slice (Logical)",
                                "Status": "Served (Existing Slice)",
                                "Throughput": _effective_throughput_bps(flow),
                                "Slice SNSSAI": s.snssai,
                                "Strategies": strat,
                                "Act BW UL": flow.bw_ul,
                                "Act BW DL": flow.bw_dl,
                                "Req BW UL": flow.bw_ul,
                                "Req BW DL": flow.bw_dl,
                                "App ID": _normalize_or_generate_id("IBNS_Batch", "app", used_suffixes),
                                "Flow Name": flow.name
                            })
                            break
                if not allocated:
                    flows_for_optimization.append(flow)
        else:
            flows_for_optimization = flows
            
        if not flows_for_optimization and not pre_allocated_results:
             # Edge case: No flows?
             pass
            
        if not flows_for_optimization:
            logger.info("All flows served by existing slices. Skipping IBNS optimization.")
            ibns_slice_stats = _build_ibns_slice_stats(existing_slices)
            return {
                "meta": {
                    "status": "success",
                    "objective_value": None,
                    "mode": mode,
                    "params": {"mode": mode},
                    "breakdown": {}
                },
                "target_app": {
                    "app_id": _normalize_or_generate_id("IBNS_Batch_Request", "app", used_suffixes),
                    "name": "IBNS Optimization Batch",
                    "flows": pre_allocated_results
                },
                "impacted_flows": [],
                "slice_stats": ibns_slice_stats,
                "scenario": {}
            }

        logger.info(f"Proceeding to IBNS Optimization for {len(flows_for_optimization)} flows...")

        # 关键步骤2：构造拓扑与路径模型
        nodes = _to_model_list(nodes_data, Node)
        links = _to_model_list(links_data, Link)
        paths = _to_model_list(
            paths_data,
            Path,
            extra_transform=lambda p: {
                **p,
                "links": [tuple(x) for x in p.get("links", [])] if isinstance(p.get("links"), list) else p.get("links")
            },
        )

        # 关键步骤3：构造服务类型与 SLA 映射
        service_types = {}
        for st in service_types_data:
            valid_keys = ServiceType.__annotations__.keys()
            filtered_st = {k: v for k, v in st.items() if k in valid_keys}
            obj = ServiceType(**filtered_st)
            service_types[obj.id] = obj
        
        sla_profiles = {}
        if sla_profiles_data:
            for sla in sla_profiles_data:
                valid_keys = SLAProfile.__annotations__.keys()
                filtered_sla = {k: v for k, v in sla.items() if k in valid_keys}
                if 'kpi_thresholds' in filtered_sla and isinstance(filtered_sla['kpi_thresholds'], dict):
                     filtered_sla['kpi_thresholds'] = {int(k): float(v) for k, v in filtered_sla['kpi_thresholds'].items()}
                obj = SLAProfile(**filtered_sla)
                sla_profiles[obj.service_type_id] = obj

        # 关键步骤：P2且未显式传SLA时，从UEContext按flow兜底构造SLA。
        if str(mode).lower() == 'p2' and not sla_profiles:
            try:
                from agents.tools.db_tool import get_ue_context_by_supi
            except Exception:
                get_ue_context_by_supi = None

            if get_ue_context_by_supi is not None:
                ue_ctx_cache: Dict[str, Optional[Dict[str, Any]]] = {}
                sla_by_st: Dict[int, Dict[str, float]] = {}

                for flow in flows_for_optimization:
                    flow_id = getattr(flow, 'flow_id', None)
                    st_id = getattr(flow, 'service_type_id', None)
                    supi = flow_supi_map.get(str(flow_id)) if flow_id is not None else None
                    if not supi:
                        supi = getattr(flow, 'supi', None)
                    if not supi or not flow_id or st_id is None:
                        continue

                    st_id_int = int(st_id)
                    if supi not in ue_ctx_cache:
                        ue_ctx_cache[supi] = get_ue_context_by_supi(str(supi))
                    flow_kpis = _flow_sla_from_ue_context(ue_ctx_cache.get(supi) or {}, str(flow_id))
                    if flow_kpis:
                        sla_by_st[st_id_int] = _merge_sla_kpis(sla_by_st.get(st_id_int, {}), flow_kpis)

                for st_id, kpis in sla_by_st.items():
                    sla_profiles[st_id] = SLAProfile(
                        service_type_id=st_id,
                        kpi_thresholds={int(k): float(v) for k, v in kpis.items()}
                    )

            if not sla_profiles:
                logger.warning("P2模式未找到SLA配置（显式参数和UEContext均为空），将按无SLA阈值继续。")

        # 关键步骤4：初始化引擎并按模式求解
        engine = IBNSOptimizationEngine()
        
        # 3. Solve (Only for remaining flows)
        if mode == 'p1':
            result_df = engine.solve_p1_fulfillment(flows_for_optimization, nodes, links, paths, service_types)
        else:
            result_df = engine.solve_p2_assurance(flows_for_optimization, nodes, links, paths, service_types, sla_profiles)
            
        # 关键步骤5：合并已有切片分配结果与优化结果及策略计算
        optimized_results = []
        if not result_df.empty:
             raw_list = result_df.to_dict(orient='records')
             for item in raw_list:
                 # Check if strategy is already calculated by engine (if modeled after SliceOptimizationEngine)
                 if "Strategies" not in item:
                     # Attempt to calculate strategy if missing
                     fid = item.get("Flow ID")
                     f_obj = flow_map.get(fid)
                     if f_obj:
                         act_ul = float(item.get("Act BW UL", f_obj.bw_ul)) # Default to req if missing
                         act_dl = float(item.get("Act BW DL", f_obj.bw_dl))
                         mapped_snssai = item.get("Slice SNSSAI") or item.get("New Slice")
                         
                         item["Strategies"] = _get_strategies(f_obj, mapped_snssai, act_ul, act_dl)
                         # Fill missing keys for consistency
                         if "App ID" not in item:
                             item["App ID"] = _normalize_or_generate_id("IBNS_Batch", "app", used_suffixes)
                         if "Flow Name" not in item: item["Flow Name"] = f_obj.name
                         if "Req BW UL" not in item: item["Req BW UL"] = f_obj.bw_ul
                         if "Req BW DL" not in item: item["Req BW DL"] = f_obj.bw_dl
                 optimized_results.append(item)

        final_results = pre_allocated_results + optimized_results

        # 构造切片统计 (for consistency)
        ibns_slice_stats = []
        if existing_slices:
            ibns_slice_stats = _build_ibns_slice_stats(existing_slices)

        # 格式化输出 (一致性结构)
        return {
            "meta": {
                "status": "success",
                "objective_value": None,
                "mode": mode,
                "params": {"mode": mode},
                "breakdown": {}
            },
            "target_app": {
                "app_id": _normalize_or_generate_id("IBNS_Batch_Request", "app", used_suffixes),
                "name": "IBNS Optimization Batch",
                "flows": final_results
            },
            "impacted_flows": [],  # IBNS assumes dedicated/isolated optimization
            "slice_stats": ibns_slice_stats,
            "scenario": {}
        }
    except Exception as e:
        logger.error(f"IBNS Optimization failed: {e}", exc_info=True)
        return {"status": "error", "message": f"IBNS Optimization Failed: {str(e)}"}
