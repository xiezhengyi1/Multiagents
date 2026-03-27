"""Scenario initialization entrypoint.

Usage:
  python tools/init_scenario.py
  python tools/init_scenario.py --reset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import secrets
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from tools.db_tool import (
    build_flow_info_from_five_tuple,
    get_latest_snapshot_data,
    sync_latest_snapshot_flow_catalog_to_ue_context,
    update_scenario_in_db,
    upsert_ue_context,
)
from tools.optimizer.models import App, Flow, Node, Slice


def _filter_dataclass_kwargs(src: Dict[str, Any], cls, exclude: Optional[set] = None) -> Dict[str, Any]:
    """按 dataclass 定义过滤字段，避免脏数据影响反序列化。"""
    exclude = exclude or set()
    valid_keys = set(cls.__annotations__.keys()) - exclude
    return {k: v for k, v in src.items() if k in valid_keys}


def _asdict_list(items: List[Any]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in items]


def _service_type_id_to_name(service_type_id: int) -> str:
    """service_type_id 到 service_type 的可解释映射。"""
    mapping = {
        1: "eMBB",
        2: "URLLC",
        3: "mMTC",
    }
    return mapping.get(service_type_id, f"type_{service_type_id}")


def _service_type_name_to_id(service_type: str) -> int:
    """service_type 到 service_type_id 的兼容映射。"""
    reverse_mapping = {
        "embb": 1,
        "urllc": 2,
        "mmtc": 3,
    }
    return reverse_mapping.get(str(service_type).strip().lower(), 1)


def _build_flow_from_dict(f_dict: Dict[str, Any], used_suffixes: Optional[set] = None) -> Flow:
    """按最新 Flow 模型构建对象，自动兼容新增仿真结果字段。"""
    flow_kwargs = _filter_dataclass_kwargs(f_dict, Flow)

    # 关键步骤：兼容 service_type/service_type_id 双字段
    if "service_type_id" not in flow_kwargs:
        raw_st_id = f_dict.get("service_type_id", None)
        if raw_st_id is not None:
            try:
                flow_kwargs["service_type_id"] = int(raw_st_id)
            except (TypeError, ValueError):
                flow_kwargs["service_type_id"] = 1

    if "service_type" not in flow_kwargs:
        if "service_type" in f_dict and f_dict.get("service_type") is not None:
            flow_kwargs["service_type"] = str(f_dict.get("service_type"))
        else:
            flow_kwargs["service_type"] = _service_type_id_to_name(int(flow_kwargs.get("service_type_id", 1)))

    if "service_type_id" not in flow_kwargs:
        flow_kwargs["service_type_id"] = _service_type_name_to_id(flow_kwargs["service_type"])

    # 关键步骤：补齐 Flow 必填字段，避免历史/脏数据导致初始化失败
    flow_kwargs.setdefault("name", str(f_dict.get("name", f_dict.get("flow_id", "flow_unknown"))))
    flow_kwargs.setdefault("flow_id", str(f_dict.get("flow_id", f_dict.get("name", "flow_unknown"))))
    flow_kwargs.setdefault("bw_ul", 0.0)
    flow_kwargs.setdefault("bw_dl", 0.0)
    flow_kwargs.setdefault("gbr_ul", 0.0)
    flow_kwargs.setdefault("gbr_dl", 0.0)
    flow_kwargs.setdefault("lat", 0.0)
    flow_kwargs.setdefault("loss_req", 0.0)
    flow_kwargs.setdefault("jitter_req", 0.0)
    flow_kwargs.setdefault("priority", 1)

    flow_kwargs["flow_id"] = _normalize_or_generate_id(
        flow_kwargs.get("flow_id"),
        "flow",
        used_suffixes,
    )

    return Flow(**flow_kwargs)


def _build_slice_from_dict(s_dict: Dict[str, Any]) -> Slice:
    """按最新 Slice 模型构建对象，兼容缺失字段与新增仿真结果字段。"""
    sst = s_dict.get("sst", 0)
    sd = s_dict.get("sd", "000000")
    generated_name = f"Slice_{sst}_{sd}"

    slice_kwargs = _filter_dataclass_kwargs(s_dict, Slice, exclude={"snssai"})

    # 关键步骤：补齐 Slice 必填字段，避免反序列化时报 missing required args
    slice_kwargs.setdefault("name", generated_name)
    slice_kwargs.setdefault("sst", 0)
    slice_kwargs.setdefault("sd", "000000")
    slice_kwargs.setdefault("total_bw_ul", 0.0)
    slice_kwargs.setdefault("total_bw_dl", 0.0)
    slice_kwargs.setdefault("current_load_bw_ul", 0.0)
    slice_kwargs.setdefault("current_load_bw_dl", 0.0)
    slice_kwargs.setdefault("latency", 0.0)
    slice_kwargs.setdefault("proc_delay", 0.0)
    slice_kwargs.setdefault("loss", 0.0)
    slice_kwargs.setdefault("jitter", 0.0)
    slice_kwargs.setdefault("reserved_bw", 0.0)

    return Slice(**slice_kwargs)


def _build_node_from_dict(n_dict: Dict[str, Any]) -> Node:
    """按最新 Node 模型构建对象，自动兼容新增仿真结果字段。"""
    node_kwargs = _filter_dataclass_kwargs(n_dict, Node)

    # 关键步骤：补齐 Node 必填字段
    node_kwargs.setdefault("name", str(n_dict.get("id", "node_unknown")))
    node_kwargs.setdefault("cpu_capacity", 0.0)
    node_kwargs.setdefault("memory_capacity", 0.0)
    node_kwargs.setdefault("slices_hosted", [])

    return Node(**node_kwargs)

# Default scenario structure provided by configuration
DEFAULT_SCENARIO_APPS_JSON = [
    {"name": "Remote_Drive", "supi": "imsi-20893001", "flows": [{"lat": 12.0, "name": "Remote_Drive_video_1", "bw_dl": 18.0, "bw_ul": 16.0, "gbr_dl": 14.0, "gbr_ul": 12.0, "flow_id": "app_remote_drive_f1", "loss_req": 0.0005, "priority": 1, "old_slice": "01000001", "jitter_req": 2.0, "packet_size": 12000.0, "arrival_rate": 500.0, "service_type_id": 1, "old_allocated_bw_dl": 18.0, "old_allocated_bw_ul": 16.0}], "app_id": "app_remote_drive", "min_lat": 62.44, "max_prio": 4, "total_bw_dl": 35.97, "total_bw_ul": 33.91},
    {"name": "4K_Video", "supi": "imsi-20893002", "flows": [{"lat": 35.0, "name": "4K_Video_stream_1", "bw_dl": 45.0, "bw_ul": 6.0, "gbr_dl": 36.0, "gbr_ul": 3.0, "flow_id": "app_4k_video_f1", "loss_req": 0.003, "priority": 5, "old_slice": "01000002", "jitter_req": 12.0, "packet_size": 60000.0, "arrival_rate": 800.0, "service_type_id": 1, "old_allocated_bw_dl": 45.0, "old_allocated_bw_ul": 6.0}, {"lat": 25.0, "name": "4K_Video_control_2", "supi": "imsi-20893002", "bw_dl": 2.5, "bw_ul": 2.0, "gbr_dl": 1.2, "gbr_ul": 1.0, "flow_id": "app_4k_video_f2", "loss_req": 0.0015, "priority": 4, "old_slice": "02000001", "jitter_req": 6.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 2, "old_allocated_bw_dl": 2.5, "old_allocated_bw_ul": 2.0}], "app_id": "app_4k_video", "min_lat": 5.69, "max_prio": 3, "total_bw_dl": 11.850000000000001, "total_bw_ul": 7.359999999999999},
    {"name": "IoT_Sensor", "supi": "imsi-20893003", "flows": [{"lat": 120.0, "name": "IoT_Sensor_video_1", "bw_dl": 8.0, "bw_ul": 5.0, "gbr_dl": 5.0, "gbr_ul": 3.0, "flow_id": "app_iot_sensor_f1", "loss_req": 0.02, "priority": 8, "old_slice": "01000001", "jitter_req": 30.0, "packet_size": 24000.0, "arrival_rate": 50.0, "service_type_id": 1, "old_allocated_bw_dl": 8.0, "old_allocated_bw_ul": 5.0}, {"lat": 80.0, "name": "IoT_Sensor_control_2", "supi": "imsi-20893003", "bw_dl": 0.8, "bw_ul": 1.2, "gbr_dl": 0.4, "gbr_ul": 0.8, "flow_id": "app_iot_sensor_f2", "loss_req": 0.01, "priority": 7, "old_slice": "01000002", "jitter_req": 20.0, "packet_size": 8000.0, "arrival_rate": 800.0, "service_type_id": 1, "old_allocated_bw_dl": 0.8, "old_allocated_bw_ul": 1.2}], "app_id": "app_iot_sensor", "min_lat": 10.82, "max_prio": 7, "total_bw_dl": 45.11, "total_bw_ul": 49.269999999999996},
    {"name": "Web_Browse", "supi": "imsi-20893004", "flows": [{"lat": 90.0, "name": "Web_Browse_control_1", "bw_dl": 6.0, "bw_ul": 1.0, "gbr_dl": 2.0, "gbr_ul": 0.5, "flow_id": "app_web_browse_f1", "loss_req": 0.03, "priority": 10, "old_slice": "02000002", "jitter_req": 35.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 2, "old_allocated_bw_dl": 6.0, "old_allocated_bw_ul": 1.0}, {"lat": 110.0, "name": "Web_Browse_control_2", "supi": "imsi-20893004", "bw_dl": 4.0, "bw_ul": 0.8, "gbr_dl": 1.5, "gbr_ul": 0.4, "flow_id": "app_web_browse_f2", "loss_req": 0.04, "priority": 11, "old_slice": "03000001", "jitter_req": 45.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 3, "old_allocated_bw_dl": 4.0, "old_allocated_bw_ul": 0.8}], "app_id": "app_web_browse", "min_lat": 6.55, "max_prio": 3, "total_bw_dl": 7.039999999999999, "total_bw_ul": 7.93},
    {"name": "AR_Gaming", "supi": "imsi-20893005", "flows": [{"lat": 8.0, "name": "AR_Gaming_control_1", "bw_dl": 12.0, "bw_ul": 8.0, "gbr_dl": 10.0, "gbr_ul": 6.0, "flow_id": "app_ar_gaming_f1", "loss_req": 0.0008, "priority": 2, "old_slice": "01000001", "jitter_req": 2.0, "packet_size": 8000.0, "arrival_rate": 50.0, "service_type_id": 1, "old_allocated_bw_dl": 12.0, "old_allocated_bw_ul": 8.0}, {"lat": 15.0, "name": "AR_Gaming_video_2", "supi": "imsi-20893005", "bw_dl": 55.0, "bw_ul": 18.0, "gbr_dl": 45.0, "gbr_ul": 12.0, "flow_id": "app_ar_gaming_f2", "loss_req": 0.0015, "priority": 3, "old_slice": "03000001", "jitter_req": 4.0, "packet_size": 24000.0, "arrival_rate": 100.0, "service_type_id": 3, "old_allocated_bw_dl": 55.0, "old_allocated_bw_ul": 18.0}], "app_id": "app_ar_gaming", "min_lat": 10.83, "max_prio": 9, "total_bw_dl": 36.21, "total_bw_ul": 16.18},
    {"name": "Factory_Robot", "supi": "imsi-20893006", "flows": [{"lat": 10.0, "name": "Factory_Robot_video_1", "bw_dl": 20.0, "bw_ul": 20.0, "gbr_dl": 18.0, "gbr_ul": 18.0, "flow_id": "app_factory_robot_f1", "loss_req": 0.0003, "priority": 1, "old_slice": "01000001", "jitter_req": 1.5, "packet_size": 12000.0, "arrival_rate": 500.0, "service_type_id": 1, "old_allocated_bw_dl": 20.0, "old_allocated_bw_ul": 20.0}], "app_id": "app_factory_robot", "min_lat": 66.46, "max_prio": 7, "total_bw_dl": 27.77, "total_bw_ul": 19.52},
    {"name": "Smart_Meter", "supi": "imsi-20893007", "flows": [{"lat": 300.0, "name": "Smart_Meter_telemetry_1", "bw_dl": 0.2, "bw_ul": 0.6, "gbr_dl": 0.1, "gbr_ul": 0.3, "flow_id": "app_smart_meter_f1", "loss_req": 0.05, "priority": 12, "old_slice": "02000001", "jitter_req": 80.0, "packet_size": 2000.0, "arrival_rate": 50.0, "service_type_id": 2, "old_allocated_bw_dl": 0.2, "old_allocated_bw_ul": 0.6}], "app_id": "app_smart_meter", "min_lat": 77.34, "max_prio": 11, "total_bw_dl": 0.94, "total_bw_ul": 2.84},
    {"name": "Telemedicine", "supi": "imsi-20893008", "flows": [{"lat": 15.0, "name": "Telemedicine_control_1", "bw_dl": 25.0, "bw_ul": 12.0, "gbr_dl": 20.0, "gbr_ul": 9.0, "flow_id": "app_telemedicine_f1", "loss_req": 0.0005, "priority": 1, "old_slice": "03000001", "jitter_req": 2.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 3, "old_allocated_bw_dl": 25.0, "old_allocated_bw_ul": 12.0}], "app_id": "app_telemedicine", "min_lat": 9.28, "max_prio": 1, "total_bw_dl": 5.39, "total_bw_ul": 2.05},
    {"name": "Drone_Control", "supi": "imsi-20893009", "flows": [{"lat": 20.0, "name": "Drone_Control_video_1", "bw_dl": 30.0, "bw_ul": 10.0, "gbr_dl": 24.0, "gbr_ul": 7.0, "flow_id": "app_drone_control_f1", "loss_req": 0.001, "priority": 2, "old_slice": "01000002", "jitter_req": 5.0, "packet_size": 16000.0, "arrival_rate": 100.0, "service_type_id": 1, "old_allocated_bw_dl": 30.0, "old_allocated_bw_ul": 10.0}], "app_id": "app_drone_control", "min_lat": 29.09, "max_prio": 9, "total_bw_dl": 27.55, "total_bw_ul": 39.35},
    {"name": "Cloud_Render", "supi": "imsi-20893010", "flows": [{"lat": 12.0, "name": "Cloud_Render_control_1", "bw_dl": 60.0, "bw_ul": 8.0, "gbr_dl": 48.0, "gbr_ul": 5.0, "flow_id": "app_cloud_render_f1", "loss_req": 0.002, "priority": 4, "old_slice": "02000001", "jitter_req": 6.0, "packet_size": 60000.0, "arrival_rate": 800.0, "service_type_id": 2, "old_allocated_bw_dl": 60.0, "old_allocated_bw_ul": 8.0}, {"lat": 180.0, "name": "Cloud_Render_telemetry_2", "supi": "imsi-20893010", "bw_dl": 0.5, "bw_ul": 1.2, "gbr_dl": 0.2, "gbr_ul": 0.7, "flow_id": "app_cloud_render_f2", "loss_req": 0.02, "priority": 9, "old_slice": "01000002", "jitter_req": 50.0, "packet_size": 4000.0, "arrival_rate": 100.0, "service_type_id": 1, "old_allocated_bw_dl": 0.5, "old_allocated_bw_ul": 1.2}], "app_id": "app_cloud_render", "min_lat": 9.99, "max_prio": 6, "total_bw_dl": 4.930000000000001, "total_bw_ul": 5.260000000000001}]

# 简单缓存：仅作为优化-下发-提交之间的短期桥接
_SCENARIO_CACHE: Dict[str, Optional[List[Any]]] = {
    "apps": None,
    "slices": None,
    "nodes": None,
}


def cache_scenario(apps: List[App], slices: List[Slice], nodes: List[Node]) -> None:
    _SCENARIO_CACHE["apps"] = apps
    _SCENARIO_CACHE["slices"] = slices
    _SCENARIO_CACHE["nodes"] = nodes


def get_cached_scenario() -> Tuple[Optional[List[App]], Optional[List[Slice]], Optional[List[Node]]]:
    return _SCENARIO_CACHE.get("apps"), _SCENARIO_CACHE.get("slices"), _SCENARIO_CACHE.get("nodes")


def clear_cached_scenario() -> None:
    _SCENARIO_CACHE["apps"] = None
    _SCENARIO_CACHE["slices"] = None
    _SCENARIO_CACHE["nodes"] = None


def serialize_scenario_for_api(apps: List[App], slices: List[Slice], nodes: List[Node]) -> Dict[str, Any]:
    return {
        "apps": _asdict_list(apps),
        "slices": _asdict_list(slices),
        "nodes": _asdict_list(nodes),
    }

def _map_5qi_by_service_type(service_type_id: int) -> int:
    # 关键步骤：按业务类型给一个可解释的默认 5QI
    mapping = {
        1: 9,   # eMBB / 一般业务
        2: 7,   # 低时延增强
        3: 65,  # mMTC / 海量连接
    }
    return mapping.get(service_type_id, 9)


def _normalize_supi(raw_supi: Any) -> Optional[str]:
    """关键步骤：统一 SUPI 形态，确保同一 UE 多流按同一 key 聚合。"""
    if raw_supi is None:
        return None

    supi = str(raw_supi).strip()
    if not supi:
        return None

    # 兼容历史初始化数据: imsi-20893001-0000 -> imsi-20893001
    m = re.fullmatch(r"(imsi-\d+)-\d{4}", supi)
    if m:
        return m.group(1)

    return supi


_ID_FORMAT = {
    "app": re.compile(r"^app-(\d{4})$"),
    "flow": re.compile(r"^flow-(\d{4})$"),
}


_GLOBAL_USED_SUFFIXES: set = set()


def _allocate_unique_suffix(used_suffixes: set) -> str:
    for _ in range(1000):
        suffix = f"{secrets.randbelow(10000):04d}"
        if suffix not in used_suffixes:
            used_suffixes.add(suffix)
            return suffix
    raise RuntimeError("无法分配唯一ID后缀，请检查ID空间使用情况")


def _normalize_or_generate_id(raw_id: Any, prefix: str, used_suffixes: Optional[set] = None) -> str:
    """关键步骤：仅接受新ID格式；旧ID一律重生，并保证后缀唯一。"""
    used_suffixes = _GLOBAL_USED_SUFFIXES if used_suffixes is None else used_suffixes
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


def _extract_app_supi(app_dict: Dict[str, Any]) -> Optional[str]:
    """关键步骤：优先使用 app.supi，兼容历史 flow.supi 兜底。"""
    app_supi = _normalize_supi(app_dict.get("supi"))
    if app_supi:
        return app_supi
    return None


def _seed_ue_contexts_from_apps(apps: List[App]) -> int:
    """
    根据场景中的 flow 为每个 UE 生成并写入 UeContext 关键字段。
    返回成功 upsert 的 UE 数量。
    """
    ue_payload: Dict[str, Dict[str, Any]] = {}

    for app in apps:
        supi = _normalize_supi(getattr(app, "supi", None))
        for flow in app.flows:
            flow_id = getattr(flow, "flow_id", None)
            if not supi or not flow_id:
                continue

            try:
                service_type_id = int(getattr(flow, "service_type_id", 1))
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
                }

            if not any(item.get("app_id") == getattr(app, "app_id", None) for item in ue_payload[supi]["app_catalog"]):
                ue_payload[supi]["app_catalog"].append(
                    {
                        "supi": supi,
                        "app_name": getattr(app, "name", None),
                        "app_id": getattr(app, "app_id", None),
                        "flow_count": len(getattr(app, "flows", []) or []),
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
                    "app_name": getattr(app, "name", None),
                    "app_id": getattr(app, "app_id", None),
                    "flow_name": getattr(flow, "name", None),
                    "flow_id": flow_id,
                    "service_type": getattr(flow, "service_type", None),
                    "service_type_id": service_type_id,
                    "bw_ul": getattr(flow, "bw_ul", None),
                    "bw_dl": getattr(flow, "bw_dl", None),
                    "gbr_ul": getattr(flow, "gbr_ul", None),
                    "gbr_dl": getattr(flow, "gbr_dl", None),
                    "lat": getattr(flow, "lat", None),
                    "loss_req": getattr(flow, "loss_req", None),
                    "jitter_req": getattr(flow, "jitter_req", None),
                    "priority": getattr(flow, "priority", None),
                    "current_bw_ul": getattr(flow, "old_allocated_bw_ul", None)
                    if getattr(flow, "old_allocated_bw_ul", None) is not None
                    else getattr(flow, "bw_ul", None),
                    "current_bw_dl": getattr(flow, "old_allocated_bw_dl", None)
                    if getattr(flow, "old_allocated_bw_dl", None) is not None
                    else getattr(flow, "bw_dl", None),
                    "five_tuple": list(getattr(flow, "five_tuple", None))
                    if isinstance(getattr(flow, "five_tuple", None), (list, tuple))
                    else None,
                }
            )

            flow_info = build_flow_info_from_five_tuple(getattr(flow, "five_tuple", None))
            if not flow_info:
                flow_info = {
                    "flowDescription": f"permit out ip from {supi} to any",
                    "flowDirection": "BIDIRECTIONAL",
                }

            ue_payload[supi]["pcc_rules"][sm_policy_id][pcc_rule_id] = {
                "pccRuleId": pcc_rule_id,
                "precedence": int(flow.priority),
                "flowInfos": [flow_info],
                "refQosData": [qos_id],
                # 补充 TSCAI 信息 (如果 flow 有相关字段)
                "tscaiInputDl": [{
                    "periodicity": int(1000.0 / getattr(flow, 'arrival_rate', 50.0) * 1e6) if getattr(flow, 'arrival_rate', 0) > 0 else None,
                    "surTimeInNumMsg": int(getattr(flow, 'packet_size', 0))
                }],
                "tscaiInputUl": [{
                    "periodicity": int(1000.0 / getattr(flow, 'arrival_rate', 50.0) * 1e6) if getattr(flow, 'arrival_rate', 0) > 0 else None,
                    "surTimeInNumMsg": int(getattr(flow, 'packet_size', 0))
                }]
            }

            ue_payload[supi]["qos_decs"][sm_policy_id][qos_id] = {
                "qosId": qos_id,
                "5qi": _map_5qi_by_service_type(service_type_id),
                "gbrUl": str(flow.gbr_ul),
                "gbrDl": str(flow.gbr_dl),
                "maxbrUl": str(flow.bw_ul),
                "maxbrDl": str(flow.bw_dl),
                "packetDelayBudget": int(flow.lat),
                "packetErrorRate": str(flow.loss_req),
                "priorityLevel": int(flow.priority),
            }

            ue_payload[supi]["sess_rules"][sm_policy_id][sess_rule_id] = {
                "sessRuleId": sess_rule_id,
                "authDefQos": {
                    "5qi": _map_5qi_by_service_type(service_type_id),
                    "priorityLevel": int(flow.priority),
                    "gbrUl": str(flow.gbr_ul),
                    "gbrDl": str(flow.gbr_dl),
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
        )
        if ok:
            success += 1

    return success


def sync_latest_flow_five_tuples_to_ue_context() -> Dict[str, int]:
    """
    Rebuild UE flow catalogs from the latest snapshot and refresh PCC flowInfos
    so flowDescription matches the authoritative five_tuple when available.
    """
    return sync_latest_snapshot_flow_catalog_to_ue_context()

def _deserialize_scenario(data: Dict[str, Any]) -> Tuple[List[App], List[Slice], List[Node]]:
    used_suffixes: set = set()
    apps: List[App] = []
    for app_dict in data.get("apps", []):
        flows = []
        for f_dict in app_dict.get("flows", []):
            flows.append(_build_flow_from_dict(f_dict, used_suffixes=used_suffixes))

        app_kwargs = {k: app_dict.get(k) for k in ("name", "app_id")}
        app_kwargs["app_id"] = _normalize_or_generate_id(
            app_kwargs.get("app_id"),
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


def _create_default_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Generate deterministic default scenario:
    - 5 AN
    - 4 CN (UPF)
    - 10 APP
    - each APP has 1~2 flows
    - 5 slices
    """

    rng = random.Random(20260315)

    slices_data = [
        Slice("S1_Gold", sst=2, sd="000001", total_bw_ul=120, total_bw_dl=120, current_load_bw_ul=0, current_load_bw_dl=0, latency=3, proc_delay=1, loss=0.001, jitter=1.5, reserved_bw=0),
        Slice("S2_Silver", sst=1, sd="000001", total_bw_ul=220, total_bw_dl=220, current_load_bw_ul=0, current_load_bw_dl=0, latency=10, proc_delay=2, loss=0.01, jitter=8, reserved_bw=0),
        Slice("S3_Public", sst=1, sd="000002", total_bw_ul=180, total_bw_dl=180, current_load_bw_ul=0, current_load_bw_dl=0, latency=40, proc_delay=5, loss=0.03, jitter=25, reserved_bw=0),
        Slice("S4_Platinum", sst=2, sd="000002", total_bw_ul=100, total_bw_dl=80, current_load_bw_ul=0, current_load_bw_dl=0, latency=2, proc_delay=0.8, loss=0.0005, jitter=1.0, reserved_bw=0),
        Slice("S5_Massive", sst=3, sd="000001", total_bw_ul=160, total_bw_dl=160, current_load_bw_ul=0, current_load_bw_dl=0, latency=100, proc_delay=10, loss=0.05, jitter=60, reserved_bw=0),
    ]

    slice_by_snssai = {f"{s.sst:02X}{s.sd}": s for s in slices_data}
    all_slice_snssai = [f"{s.sst:02X}{s.sd}" for s in slices_data]
    
    # 关键步骤：保证“节点托管切片并集”覆盖全部切片，且 MEC 在节点间分布
    def _build_hosted_slices(node_idx: int, node_count: int, target_k: int) -> List[str]:
        hosted = {
            all_slice_snssai[node_idx % len(all_slice_snssai)],
            all_slice_snssai[(node_idx + 1) % len(all_slice_snssai)],
        }
        remain = [n for n in all_slice_snssai if n not in hosted]
        extra_k = max(0, min(target_k - len(hosted), len(remain)))
        if extra_k > 0:
            hosted.update(rng.sample(remain, k=extra_k))
        return sorted(hosted)

    nodes_data: List[Node] = []
    for i in range(5):
        hosted = _build_hosted_slices(node_idx=i, node_count=5, target_k=3)
        # uRLLC(sst=2) 在 AN 侧更偏 MEC，下式体现“不同 AN 节点的 MEC 分布差异”
        an_pref_count = sum(1 for n in hosted if slice_by_snssai[n].sst == 2)
        nodes_data.append(
            Node(
                name=f"AN_gNB_{i}",
                cpu_capacity=1200 + i * 10,
                memory_capacity=250 + i * 32,
                slices_hosted=hosted,
                id=i,
                type="AN",
                mec_capacity=1200 + an_pref_count * 45 + i * 8,
                prb_capacity=2500 + i * 50,
            )
        )

    for i in range(4):
        hosted = _build_hosted_slices(node_idx=i, node_count=4, target_k=3)
        # eMBB/mMTC(sst=1/3) 在 CN 侧更偏 MEC，下式体现“不同 CN 节点的 MEC 分布差异”
        cn_pref_count = sum(1 for n in hosted if slice_by_snssai[n].sst in (1, 3))
        nodes_data.append(
            Node(
                name=f"CN_UPF_{i}",
                cpu_capacity=4200 + i * 60,
                memory_capacity=2024 + i * 128,
                slices_hosted=hosted,
                id=100 + i,
                type="CN",
                mec_capacity=1800 + cn_pref_count * 35 + i * 15,
                prb_capacity=0,
            )
        )

    apps_data: List[App] = []
    used_suffixes: set = set()
    
    # 使用预定义的 DEFAULT_SCENARIO_APPS_JSON 替换随机生成
    for app_dict in DEFAULT_SCENARIO_APPS_JSON:
        flows: List[Flow] = []
        for f_dict in app_dict.get("flows", []):
            flows.append(_build_flow_from_dict(f_dict, used_suffixes=used_suffixes))

        apps_data.append(
            App(
                name=app_dict["name"],
                app_id=_normalize_or_generate_id(
                    app_dict.get("app_id"),
                    "app",
                    used_suffixes,
                ),
                supi=_extract_app_supi(app_dict),
                flows=flows,
            )
        )

    return apps_data, slices_data, nodes_data


def get_initial_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Priority: latest DB snapshot -> default generation (+save)."""
    snapshot_data = get_latest_snapshot_data()
    if snapshot_data:
        apps, slices, nodes = _deserialize_scenario(snapshot_data)
        seeded = _seed_ue_contexts_from_apps(apps)
        sync_summary = sync_latest_flow_five_tuples_to_ue_context()
        print(f"[UEContext] seeded {seeded} UE records (from snapshot)")
        print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from snapshot)")
        cache_scenario(apps, slices, nodes)
        return apps, slices, nodes

    apps, slices, nodes = _create_default_scenario()
    update_scenario_in_db(apps, slices, nodes, trigger="System-Init")
    seeded = _seed_ue_contexts_from_apps(apps)
    sync_summary = sync_latest_flow_five_tuples_to_ue_context()
    print(f"[UEContext] seeded {seeded} UE records (from default)")
    print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from default)")
    cache_scenario(apps, slices, nodes)
    return apps, slices, nodes


def get_current_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    apps, slices, nodes = get_cached_scenario()
    if apps is not None and slices is not None and nodes is not None:
        return apps, slices, nodes
    return get_initial_scenario()


def initialize_scenario(reset: bool = False) -> dict:
    """Initialize scenario in cache/DB and return summary.

    Args:
        reset: If True, force-generate default scenario and persist as a new snapshot.
               If False, load latest snapshot or create defaults when missing.
    """
    if reset:
        # 关键步骤：强制生成默认场景并保存
        apps, slices, nodes = _create_default_scenario()
        cache_scenario(apps, slices, nodes)
        ok = update_scenario_in_db(apps, slices, nodes, trigger="Manual-Reset")
        if not ok:
            raise RuntimeError("Failed to persist reset scenario into DB")
        seeded = _seed_ue_contexts_from_apps(apps)
        sync_summary = sync_latest_flow_five_tuples_to_ue_context()
        print(f"[UEContext] seeded {seeded} UE records (from reset)")
        print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from reset)")
    else:
        apps, slices, nodes = get_initial_scenario()

    an_count = sum(1 for n in nodes if getattr(n, "type", "") == "AN")
    cn_count = sum(1 for n in nodes if getattr(n, "type", "") == "CN")
    min_flows = min((len(a.flows) for a in apps), default=0)
    max_flows = max((len(a.flows) for a in apps), default=0)

    return {
        "apps": len(apps),
        "slices": len(slices),
        "nodes": len(nodes),
        "an_nodes": an_count,
        "cn_nodes": cn_count,
        "flow_range_per_app": [min_flows, max_flows],
        "mode": "reset" if reset else "load-or-init",
    }


def init_main() -> None:
    parser = argparse.ArgumentParser(description="Initialize scenario cache and DB snapshot")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force reset to generated default scenario and persist a new snapshot",
    )
    args = parser.parse_args()

    summary = initialize_scenario(reset=args.reset)
    print("Scenario initialized:")
    print(summary)


if __name__ == "__main__":
    init_main()
