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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from utils.logger import setup_logger
from database.models import (
    UeAmPolicyAssociationRecord,
    UeContextRecord,
    UeMobilityEventRecord,
    UeServingNfBindingRecord,
)

from agents.tools.db_tool import (
    build_flow_info_from_five_tuple,
    get_latest_snapshot_data,
    record_mobility_event,
    session_scope,
    sync_latest_snapshot_flow_catalog_to_ue_context,
    update_scenario_in_db,
    upsert_am_policy_association,
    upsert_ue_context,
    upsert_serving_nf_binding,
)
from agents.tools.optimizer.models import App, Flow, Node, Slice
from agents.tools.optimizer.models import (
    FlowAllocation,
    FlowService,
    FlowSLA,
    FlowTelemetry,
    FlowTraffic,
    NodeCapacity,
    NodeTelemetry,
    SliceCapacity,
    SliceLoad,
    SliceQos,
    SliceTelemetry,
)


logger = setup_logger(__name__)


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
    """按最新 Flow 模型构建对象。默认快照必须是新 schema；种子数据允许内部转换。"""
    flow_id = _normalize_or_generate_id(
        f_dict.get("id", f_dict.get("flow_id")),
        "flow",
        used_suffixes,
    )

    service_payload = f_dict.get("service") if isinstance(f_dict.get("service"), dict) else {}
    sla_payload = f_dict.get("sla") if isinstance(f_dict.get("sla"), dict) else {}
    traffic_payload = f_dict.get("traffic") if isinstance(f_dict.get("traffic"), dict) else {}
    allocation_payload = f_dict.get("allocation") if isinstance(f_dict.get("allocation"), dict) else {}
    telemetry_payload = f_dict.get("telemetry") if isinstance(f_dict.get("telemetry"), dict) else {}

    service_type = str(
        service_payload.get("service_type")
        or f_dict.get("service_type")
        or _service_type_id_to_name(int(service_payload.get("service_type_id", f_dict.get("service_type_id", 1)) or 1))
    )
    service_type_id = int(
        service_payload.get("service_type_id")
        or f_dict.get("service_type_id")
        or _service_type_name_to_id(service_type)
    )

    bandwidth_ul = float(sla_payload.get("bandwidth_ul", f_dict.get("bw_ul", 0.0)) or 0.0)
    bandwidth_dl = float(sla_payload.get("bandwidth_dl", f_dict.get("bw_dl", 0.0)) or 0.0)
    guaranteed_bandwidth_ul = float(sla_payload.get("guaranteed_bandwidth_ul", f_dict.get("gbr_ul", bandwidth_ul)) or 0.0)
    guaranteed_bandwidth_dl = float(sla_payload.get("guaranteed_bandwidth_dl", f_dict.get("gbr_dl", bandwidth_dl)) or 0.0)
    latency = float(sla_payload.get("latency", f_dict.get("lat", 0.0)) or 0.0)
    jitter = float(sla_payload.get("jitter", f_dict.get("jitter_req", 0.0)) or 0.0)
    loss_rate = float(sla_payload.get("loss_rate", f_dict.get("loss_req", 0.0)) or 0.0)
    priority = int(sla_payload.get("priority", f_dict.get("priority", 1)) or 1)

    five_tuple = traffic_payload.get("five_tuple")
    if not isinstance(five_tuple, (list, tuple)):
        five_tuple = f_dict.get("five_tuple")

    return Flow(
        id=flow_id,
        name=str(f_dict.get("name", f_dict.get("flow_id", f_dict.get("id", "flow_unknown")))),
        service=FlowService(
            service_type=service_type,
            service_type_id=service_type_id,
        ),
        sla=FlowSLA(
            bandwidth_ul=bandwidth_ul,
            bandwidth_dl=bandwidth_dl,
            guaranteed_bandwidth_ul=guaranteed_bandwidth_ul,
            guaranteed_bandwidth_dl=guaranteed_bandwidth_dl,
            latency=latency,
            jitter=jitter,
            loss_rate=loss_rate,
            priority=priority,
        ),
        traffic=FlowTraffic(
            packet_size=float(traffic_payload.get("packet_size", f_dict.get("packet_size", 0.0)) or 0.0),
            arrival_rate=float(traffic_payload.get("arrival_rate", f_dict.get("arrival_rate", 0.0)) or 0.0),
            five_tuple=tuple(five_tuple) if isinstance(five_tuple, (list, tuple)) else None,
        ),
        allocation=FlowAllocation(
            current_slice_snssai=str(
                allocation_payload.get("current_slice_snssai", f_dict.get("old_slice", "")) or ""
            ).strip() or None,
            allocated_bandwidth_ul=allocation_payload.get("allocated_bandwidth_ul", f_dict.get("old_allocated_bw_ul")),
            allocated_bandwidth_dl=allocation_payload.get("allocated_bandwidth_dl", f_dict.get("old_allocated_bw_dl")),
            optimize_requested=bool(allocation_payload.get("optimize_requested", f_dict.get("optimize_requested", False))),
        ),
        telemetry=FlowTelemetry(
            throughput_ul=telemetry_payload.get("throughput_ul", f_dict.get("sim_throughput_ul")),
            throughput_dl=telemetry_payload.get("throughput_dl", f_dict.get("sim_throughput_dl")),
            latency=telemetry_payload.get("latency", f_dict.get("sim_latency")),
            jitter=telemetry_payload.get("jitter", f_dict.get("sim_jitter")),
            loss_rate=telemetry_payload.get("loss_rate", f_dict.get("sim_loss_rate")),
            packet_sent=telemetry_payload.get("packet_sent", f_dict.get("sim_packet_sent")),
            packet_received=telemetry_payload.get("packet_received", f_dict.get("sim_packet_received")),
        ),
    )


def _build_slice_from_dict(s_dict: Dict[str, Any]) -> Slice:
    """按最新 Slice 模型构建对象。"""
    sst = s_dict.get("sst", 0)
    sd = s_dict.get("sd", "000000")
    generated_name = f"Slice_{sst}_{sd}"
    capacity_payload = s_dict.get("capacity") if isinstance(s_dict.get("capacity"), dict) else {}
    load_payload = s_dict.get("load") if isinstance(s_dict.get("load"), dict) else {}
    qos_payload = s_dict.get("qos") if isinstance(s_dict.get("qos"), dict) else {}
    telemetry_payload = s_dict.get("telemetry") if isinstance(s_dict.get("telemetry"), dict) else {}

    return Slice(
        name=str(s_dict.get("name", generated_name)),
        sst=int(sst or 0),
        sd=str(sd or "000000"),
        capacity=SliceCapacity(
            total_bandwidth_ul=float(capacity_payload.get("total_bandwidth_ul", s_dict.get("total_bw_ul", 0.0)) or 0.0),
            total_bandwidth_dl=float(capacity_payload.get("total_bandwidth_dl", s_dict.get("total_bw_dl", 0.0)) or 0.0),
            reserved_bandwidth_ul=float(capacity_payload.get("reserved_bandwidth_ul", s_dict.get("reserved_bw", 0.0)) or 0.0),
            reserved_bandwidth_dl=float(capacity_payload.get("reserved_bandwidth_dl", s_dict.get("reserved_bw", 0.0)) or 0.0),
        ),
        load=SliceLoad(
            current_bandwidth_ul=float(load_payload.get("current_bandwidth_ul", s_dict.get("current_load_bw_ul", 0.0)) or 0.0),
            current_bandwidth_dl=float(load_payload.get("current_bandwidth_dl", s_dict.get("current_load_bw_dl", 0.0)) or 0.0),
        ),
        qos=SliceQos(
            latency=float(qos_payload.get("latency", s_dict.get("latency", 0.0)) or 0.0),
            processing_delay=float(qos_payload.get("processing_delay", s_dict.get("proc_delay", 0.0)) or 0.0),
            jitter=float(qos_payload.get("jitter", s_dict.get("jitter", 0.0)) or 0.0),
            loss_rate=float(qos_payload.get("loss_rate", s_dict.get("loss", 0.0)) or 0.0),
        ),
        telemetry=SliceTelemetry(
            utilization_ul=telemetry_payload.get("utilization_ul", s_dict.get("sim_utilization_ul")),
            utilization_dl=telemetry_payload.get("utilization_dl", s_dict.get("sim_utilization_dl")),
            latency=telemetry_payload.get("latency", s_dict.get("sim_latency")),
            jitter=telemetry_payload.get("jitter", s_dict.get("sim_jitter")),
            loss_rate=telemetry_payload.get("loss_rate", s_dict.get("sim_loss_rate")),
        ),
    )


def _build_node_from_dict(n_dict: Dict[str, Any]) -> Node:
    """按最新 Node 模型构建对象。"""
    node_type = str(n_dict.get("node_type", n_dict.get("type", "Generic")))
    capacity_payload = n_dict.get("capacity") if isinstance(n_dict.get("capacity"), dict) else {}
    telemetry_payload = n_dict.get("telemetry") if isinstance(n_dict.get("telemetry"), dict) else {}
    default_capacity = {
        "AN": {"cpu": 1200.0, "memory": 250.0, "mec": 1200.0, "prb": 2500.0},
        "CN": {"cpu": 4200.0, "memory": 2024.0, "mec": 1800.0, "prb": 0.0},
    }.get(node_type.upper(), {"cpu": 0.0, "memory": 0.0, "mec": 0.0, "prb": 0.0})

    raw_node_id = n_dict.get("id", -1)
    try:
        normalized_node_id = int(raw_node_id)
    except (TypeError, ValueError):
        node_id_match = re.search(r"(\d+)$", str(raw_node_id or n_dict.get("name") or "").strip())
        if node_id_match:
            ordinal = max(int(node_id_match.group(1)) - 1, 0)
            normalized_node_id = 100 + ordinal if node_type.upper() == "CN" else ordinal
        else:
            fallback_key = str(raw_node_id or n_dict.get("name") or "node").strip()
            checksum = sum((index + 1) * ord(char) for index, char in enumerate(fallback_key))
            normalized_node_id = 1000 + (checksum % 9000) if node_type.upper() == "CN" else checksum % 1000

    return Node(
        id=normalized_node_id,
        name=str(n_dict.get("name", n_dict.get("id", "node_unknown"))),
        node_type=node_type,
        capacity=NodeCapacity(
            cpu=float(capacity_payload.get("cpu", n_dict.get("cpu_capacity", default_capacity["cpu"])) or 0.0),
            memory=float(capacity_payload.get("memory", n_dict.get("memory_capacity", default_capacity["memory"])) or 0.0),
            mec=float(capacity_payload.get("mec", n_dict.get("mec_capacity", default_capacity["mec"])) or 0.0),
            prb=float(capacity_payload.get("prb", n_dict.get("prb_capacity", default_capacity["prb"])) or 0.0),
        ),
        hosted_slice_snssais=list(n_dict.get("hosted_slice_snssais", n_dict.get("slices_hosted", [])) or []),
        telemetry=NodeTelemetry(
            cpu_utilization=telemetry_payload.get("cpu_utilization", n_dict.get("sim_cpu_utilization", 0.0)),
            mec_utilization=telemetry_payload.get("mec_utilization", n_dict.get("sim_mec_utilization", 0.0)),
            memory_utilization=telemetry_payload.get("memory_utilization", n_dict.get("sim_mem_utilization", 0.0)),
            prb_utilization=telemetry_payload.get("prb_utilization", n_dict.get("sim_prb_utilization", 0.0)),
        ),
    )

# Default scenario structure provided by configuration
DEFAULT_SCENARIO_APPS_JSON = [
    {"name": "Remote_Drive", "supi": "imsi-208930000000001", "flows": [{"lat": 12.0, "name": "Remote_Drive_video_1", "bw_dl": 18.0, "bw_ul": 16.0, "gbr_dl": 14.0, "gbr_ul": 12.0, "flow_id": "app_remote_drive_f1", "loss_req": 0.0005, "priority": 1, "old_slice": "01000001", "jitter_req": 2.0, "packet_size": 12000.0, "arrival_rate": 500.0, "service_type_id": 1, "old_allocated_bw_dl": 18.0, "old_allocated_bw_ul": 16.0}], "app_id": "app_remote_drive", "min_lat": 62.44, "max_prio": 4, "total_bw_dl": 35.97, "total_bw_ul": 33.91},
    {"name": "4K_Video", "supi": "imsi-208930000000002", "flows": [{"lat": 35.0, "name": "4K_Video_stream_1", "bw_dl": 45.0, "bw_ul": 6.0, "gbr_dl": 36.0, "gbr_ul": 3.0, "flow_id": "app_4k_video_f1", "loss_req": 0.003, "priority": 5, "old_slice": "01000002", "jitter_req": 12.0, "packet_size": 60000.0, "arrival_rate": 800.0, "service_type_id": 1, "old_allocated_bw_dl": 45.0, "old_allocated_bw_ul": 6.0}, {"lat": 25.0, "name": "4K_Video_control_2", "supi": "imsi-20893002", "bw_dl": 2.5, "bw_ul": 2.0, "gbr_dl": 1.2, "gbr_ul": 1.0, "flow_id": "app_4k_video_f2", "loss_req": 0.0015, "priority": 4, "old_slice": "02000001", "jitter_req": 6.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 2, "old_allocated_bw_dl": 2.5, "old_allocated_bw_ul": 2.0}], "app_id": "app_4k_video", "min_lat": 5.69, "max_prio": 3, "total_bw_dl": 11.850000000000001, "total_bw_ul": 7.359999999999999},
    {"name": "IoT_Sensor", "supi": "imsi-208930000000003", "flows": [{"lat": 120.0, "name": "IoT_Sensor_video_1", "bw_dl": 8.0, "bw_ul": 5.0, "gbr_dl": 5.0, "gbr_ul": 3.0, "flow_id": "app_iot_sensor_f1", "loss_req": 0.02, "priority": 8, "old_slice": "01000001", "jitter_req": 30.0, "packet_size": 24000.0, "arrival_rate": 50.0, "service_type_id": 1, "old_allocated_bw_dl": 8.0, "old_allocated_bw_ul": 5.0}, {"lat": 80.0, "name": "IoT_Sensor_control_2", "supi": "imsi-20893003", "bw_dl": 0.8, "bw_ul": 1.2, "gbr_dl": 0.4, "gbr_ul": 0.8, "flow_id": "app_iot_sensor_f2", "loss_req": 0.01, "priority": 7, "old_slice": "01000002", "jitter_req": 20.0, "packet_size": 8000.0, "arrival_rate": 800.0, "service_type_id": 1, "old_allocated_bw_dl": 0.8, "old_allocated_bw_ul": 1.2}], "app_id": "app_iot_sensor", "min_lat": 10.82, "max_prio": 7, "total_bw_dl": 45.11, "total_bw_ul": 49.269999999999996},
    {"name": "Web_Browse", "supi": "imsi-208930000000004", "flows": [{"lat": 90.0, "name": "Web_Browse_control_1", "bw_dl": 6.0, "bw_ul": 1.0, "gbr_dl": 2.0, "gbr_ul": 0.5, "flow_id": "app_web_browse_f1", "loss_req": 0.03, "priority": 10, "old_slice": "02000002", "jitter_req": 35.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 2, "old_allocated_bw_dl": 6.0, "old_allocated_bw_ul": 1.0}, {"lat": 110.0, "name": "Web_Browse_control_2", "supi": "imsi-20893004", "bw_dl": 4.0, "bw_ul": 0.8, "gbr_dl": 1.5, "gbr_ul": 0.4, "flow_id": "app_web_browse_f2", "loss_req": 0.04, "priority": 11, "old_slice": "03000001", "jitter_req": 45.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 3, "old_allocated_bw_dl": 4.0, "old_allocated_bw_ul": 0.8}], "app_id": "app_web_browse", "min_lat": 6.55, "max_prio": 3, "total_bw_dl": 7.039999999999999, "total_bw_ul": 7.93},
    {"name": "AR_Gaming", "supi": "imsi-208930000000005", "flows": [{"lat": 8.0, "name": "AR_Gaming_control_1", "bw_dl": 12.0, "bw_ul": 8.0, "gbr_dl": 10.0, "gbr_ul": 6.0, "flow_id": "app_ar_gaming_f1", "loss_req": 0.0008, "priority": 2, "old_slice": "01000001", "jitter_req": 2.0, "packet_size": 8000.0, "arrival_rate": 50.0, "service_type_id": 1, "old_allocated_bw_dl": 12.0, "old_allocated_bw_ul": 8.0}, {"lat": 15.0, "name": "AR_Gaming_video_2", "supi": "imsi-20893005", "bw_dl": 55.0, "bw_ul": 18.0, "gbr_dl": 45.0, "gbr_ul": 12.0, "flow_id": "app_ar_gaming_f2", "loss_req": 0.0015, "priority": 3, "old_slice": "03000001", "jitter_req": 4.0, "packet_size": 24000.0, "arrival_rate": 100.0, "service_type_id": 3, "old_allocated_bw_dl": 55.0, "old_allocated_bw_ul": 18.0}], "app_id": "app_ar_gaming", "min_lat": 10.83, "max_prio": 9, "total_bw_dl": 36.21, "total_bw_ul": 16.18},
    {"name": "Factory_Robot", "supi": "imsi-208930000000006", "flows": [{"lat": 10.0, "name": "Factory_Robot_video_1", "bw_dl": 20.0, "bw_ul": 20.0, "gbr_dl": 18.0, "gbr_ul": 18.0, "flow_id": "app_factory_robot_f1", "loss_req": 0.0003, "priority": 1, "old_slice": "01000001", "jitter_req": 1.5, "packet_size": 12000.0, "arrival_rate": 500.0, "service_type_id": 1, "old_allocated_bw_dl": 20.0, "old_allocated_bw_ul": 20.0}], "app_id": "app_factory_robot", "min_lat": 66.46, "max_prio": 7, "total_bw_dl": 27.77, "total_bw_ul": 19.52},
    {"name": "Smart_Meter", "supi": "imsi-208930000000007", "flows": [{"lat": 300.0, "name": "Smart_Meter_telemetry_1", "bw_dl": 0.2, "bw_ul": 0.6, "gbr_dl": 0.1, "gbr_ul": 0.3, "flow_id": "app_smart_meter_f1", "loss_req": 0.05, "priority": 12, "old_slice": "02000001", "jitter_req": 80.0, "packet_size": 2000.0, "arrival_rate": 50.0, "service_type_id": 2, "old_allocated_bw_dl": 0.2, "old_allocated_bw_ul": 0.6}], "app_id": "app_smart_meter", "min_lat": 77.34, "max_prio": 11, "total_bw_dl": 0.94, "total_bw_ul": 2.84},
    {"name": "Telemedicine", "supi": "imsi-208930000000008", "flows": [{"lat": 15.0, "name": "Telemedicine_control_1", "bw_dl": 25.0, "bw_ul": 12.0, "gbr_dl": 20.0, "gbr_ul": 9.0, "flow_id": "app_telemedicine_f1", "loss_req": 0.0005, "priority": 1, "old_slice": "03000001", "jitter_req": 2.0, "packet_size": 12000.0, "arrival_rate": 200.0, "service_type_id": 3, "old_allocated_bw_dl": 25.0, "old_allocated_bw_ul": 12.0}], "app_id": "app_telemedicine", "min_lat": 9.28, "max_prio": 1, "total_bw_dl": 5.39, "total_bw_ul": 2.05},
    {"name": "Drone_Control", "supi": "imsi-208930000000009", "flows": [{"lat": 20.0, "name": "Drone_Control_video_1", "bw_dl": 30.0, "bw_ul": 10.0, "gbr_dl": 24.0, "gbr_ul": 7.0, "flow_id": "app_drone_control_f1", "loss_req": 0.001, "priority": 2, "old_slice": "01000002", "jitter_req": 5.0, "packet_size": 16000.0, "arrival_rate": 100.0, "service_type_id": 1, "old_allocated_bw_dl": 30.0, "old_allocated_bw_ul": 10.0}], "app_id": "app_drone_control", "min_lat": 29.09, "max_prio": 9, "total_bw_dl": 27.55, "total_bw_ul": 39.35},
    {"name": "Cloud_Render", "supi": "imsi-208930000000010", "flows": [{"lat": 12.0, "name": "Cloud_Render_control_1", "bw_dl": 60.0, "bw_ul": 8.0, "gbr_dl": 48.0, "gbr_ul": 5.0, "flow_id": "app_cloud_render_f1", "loss_req": 0.002, "priority": 4, "old_slice": "02000001", "jitter_req": 6.0, "packet_size": 60000.0, "arrival_rate": 800.0, "service_type_id": 2, "old_allocated_bw_dl": 60.0, "old_allocated_bw_ul": 8.0}, {"lat": 180.0, "name": "Cloud_Render_telemetry_2", "supi": "imsi-208930000000010", "bw_dl": 0.5, "bw_ul": 1.2, "gbr_dl": 0.2, "gbr_ul": 0.7, "flow_id": "app_cloud_render_f2", "loss_req": 0.02, "priority": 9, "old_slice": "01000002", "jitter_req": 50.0, "packet_size": 4000.0, "arrival_rate": 100.0, "service_type_id": 1, "old_allocated_bw_dl": 0.5, "old_allocated_bw_ul": 1.2}], "app_id": "app_cloud_render", "min_lat": 9.99, "max_prio": 6, "total_bw_dl": 4.930000000000001, "total_bw_ul": 5.260000000000001}]

# 简单缓存：仅作为优化-下发-提交之间的短期桥接
_SCENARIO_CACHE: Dict[str, Optional[List[Any]]] = {
    "apps": None,
    "slices": None,
    "nodes": None,
    "mobility": None,
    "policy_state": None,
}


def cache_scenario(
    apps: List[App],
    slices: List[Slice],
    nodes: List[Node],
    mobility: Optional[List[Dict[str, Any]]] = None,
    policy_state: Optional[Dict[str, Any]] = None,
) -> None:
    _SCENARIO_CACHE["apps"] = apps
    _SCENARIO_CACHE["slices"] = slices
    _SCENARIO_CACHE["nodes"] = nodes
    _SCENARIO_CACHE["mobility"] = mobility or []
    _SCENARIO_CACHE["policy_state"] = policy_state or {}


def get_cached_scenario() -> Tuple[Optional[List[App]], Optional[List[Slice]], Optional[List[Node]]]:
    return _SCENARIO_CACHE.get("apps"), _SCENARIO_CACHE.get("slices"), _SCENARIO_CACHE.get("nodes")


def get_cached_control_scenario() -> Dict[str, Any]:
    return {
        "apps": _SCENARIO_CACHE.get("apps"),
        "slices": _SCENARIO_CACHE.get("slices"),
        "nodes": _SCENARIO_CACHE.get("nodes"),
        "mobility": _SCENARIO_CACHE.get("mobility") or [],
        "policy_state": _SCENARIO_CACHE.get("policy_state") or {},
    }


def clear_cached_scenario() -> None:
    _SCENARIO_CACHE["apps"] = None
    _SCENARIO_CACHE["slices"] = None
    _SCENARIO_CACHE["nodes"] = None
    _SCENARIO_CACHE["mobility"] = None
    _SCENARIO_CACHE["policy_state"] = None


def serialize_scenario_for_api(apps: List[App], slices: List[Slice], nodes: List[Node]) -> Dict[str, Any]:
    return {
        "apps": _asdict_list(apps),
        "slices": _asdict_list(slices),
        "nodes": _asdict_list(nodes),
        "mobility": _SCENARIO_CACHE.get("mobility") or [],
        "policy_state": _SCENARIO_CACHE.get("policy_state") or {},
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

    # 历史种子数据错误地把 IMSI 缩成了 imsi-20893001 这种短格式。
    # 这里显式迁移为 3GPP 期望的完整 SUPI: imsi-208930000000001。
    m = re.fullmatch(r"(?i)imsi-(20893\d+)(?:-\d{4})?", supi)
    if m:
        digits = m.group(1)
        if len(digits) < 15:
            suffix = digits[5:]
            digits = f"20893{suffix.zfill(10)}"
        return f"imsi-{digits}"

    return supi


for _app in DEFAULT_SCENARIO_APPS_JSON:
    _app["supi"] = _normalize_supi(_app.get("supi"))
    for _flow in _app.get("flows", []) or []:
        if "supi" in _flow:
            _flow["supi"] = _normalize_supi(_flow.get("supi"))


_ID_FORMAT = {
    "app": re.compile(r"^app-(\d{4})$"),
    "flow": re.compile(r"^flow-(\d{4})$"),
}


_GLOBAL_USED_SUFFIXES: set = set()


def _allocate_unique_suffix(prefix: str, used_suffixes: set) -> str:
    for _ in range(1000):
        suffix = f"{secrets.randbelow(10000):04d}"
        key = f"{prefix}:{suffix}"
        if key not in used_suffixes:
            used_suffixes.add(key)
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
            key = f"{prefix}:{suffix}"
            if key not in used_suffixes:
                used_suffixes.add(key)
                return candidate

    suffix = _allocate_unique_suffix(prefix, used_suffixes)
    return f"{prefix}-{suffix}"


def _extract_app_supi(app_dict: Dict[str, Any]) -> Optional[str]:
    """关键步骤：优先使用 app.supi，兼容历史 flow.supi 兜底。"""
    app_supi = _normalize_supi(app_dict.get("supi"))
    if app_supi:
        return app_supi
    return None


def _snssai_from_parts(sst: Any, sd: Any) -> str:
    return f"{int(sst):02X}{str(sd).strip().zfill(6)}"


def _build_deterministic_five_tuple(*, supi: str, flow_ordinal: int, service_type_id: int) -> Tuple[str, str, int, int, str]:
    digits = re.sub(r"\D+", "", str(supi or ""))
    subscriber_index = int(digits[-3:]) if digits else flow_ordinal + 1
    host_octet = max(1, (flow_ordinal % 200) + 1)
    src_ip = f"10.{service_type_id}.{subscriber_index % 250}.{host_octet}"
    dst_ip = f"172.16.{service_type_id}.{10 + host_octet}"
    src_port = 20000 + ((subscriber_index * 11 + flow_ordinal) % 20000)
    dst_port = 4000 + (service_type_id * 100) + flow_ordinal
    protocol = "udp" if service_type_id in (2, 3) else "tcp"
    return src_ip, dst_ip, src_port, dst_port, protocol


def _require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Scenario field '{key}' must be a mapping")
    return value


def _require_list(parent: Dict[str, Any], key: str) -> List[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise TypeError(f"Scenario field '{key}' must be a list")
    return value


def _load_experiment_scenario_payload(path: Union[str, Path]) -> Dict[str, Any]:
    scenario_path = Path(path).expanduser().resolve()
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Scenario file must decode to a mapping: {scenario_path}")
    payload["_scenario_file"] = str(scenario_path)
    return payload


def _deserialize_experiment_scenario(payload: Dict[str, Any]) -> Tuple[List[App], List[Slice], List[Node]]:
    required_top_keys = [
        "name",
        "scenario_id",
        "slices",
        "upfs",
        "gnbs",
        "ues",
        "apps",
        "flows",
        "free5gc",
        "ns3",
        "writer",
        "topology",
        "bridge",
    ]
    missing_keys = [key for key in required_top_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"Scenario payload missing required keys: {missing_keys}")

    slice_rows = _require_list(payload, "slices")
    upf_rows = _require_list(payload, "upfs")
    gnb_rows = _require_list(payload, "gnbs")
    ue_rows = _require_list(payload, "ues")
    app_rows = _require_list(payload, "apps")
    flow_rows = _require_list(payload, "flows")

    slice_specs: Dict[str, Dict[str, Any]] = {}
    slice_flows: Dict[str, List[Dict[str, Any]]] = {}
    for slice_row in slice_rows:
        if not isinstance(slice_row, dict):
            raise TypeError("Each slice entry must be a mapping")
        label = str(slice_row.get("label") or "").strip()
        if not label:
            raise ValueError("Each slice entry must define a non-empty label")
        if label in slice_specs:
            raise ValueError(f"Duplicate slice label: {label}")
        resource = _require_mapping(slice_row, "resource")
        sst = int(slice_row.get("sst"))
        sd = str(slice_row.get("sd") or "").strip().zfill(6)
        slice_specs[label] = {
            "label": label,
            "sst": sst,
            "sd": sd,
            "resource": resource,
            "snssai": _snssai_from_parts(sst, sd),
        }
        slice_flows[label] = []

    upf_by_name: Dict[str, Dict[str, Any]] = {}
    for upf_row in upf_rows:
        if not isinstance(upf_row, dict):
            raise TypeError("Each upf entry must be a mapping")
        name = str(upf_row.get("name") or "").strip()
        if not name:
            raise ValueError("Each upf entry must define name")
        if name in upf_by_name:
            raise ValueError(f"Duplicate upf name: {name}")
        role = str(upf_row.get("role") or "").strip()
        if role not in {"branching-upf", "anchor-upf"}:
            raise ValueError(f"UPF {name} has unsupported role: {role}")
        upf_by_name[name] = upf_row

    gnb_by_name: Dict[str, Dict[str, Any]] = {}
    gnb_slice_refs_by_upf: Dict[str, set[str]] = {name: set() for name in upf_by_name}
    for gnb_row in gnb_rows:
        if not isinstance(gnb_row, dict):
            raise TypeError("Each gnb entry must be a mapping")
        name = str(gnb_row.get("name") or "").strip()
        if not name:
            raise ValueError("Each gnb entry must define name")
        if name in gnb_by_name:
            raise ValueError(f"Duplicate gnb name: {name}")
        slice_refs = [str(item).strip() for item in (gnb_row.get("slices") or []) if str(item).strip()]
        if not slice_refs:
            raise ValueError(f"gNB {name} must host at least one slice")
        for slice_ref in slice_refs:
            if slice_ref not in slice_specs:
                raise ValueError(f"gNB {name} references unknown slice: {slice_ref}")
        backhaul_upf = str(gnb_row.get("backhaul_upf") or "").strip()
        if backhaul_upf not in upf_by_name:
            raise ValueError(f"gNB {name} references unknown backhaul UPF: {backhaul_upf}")
        gnb_by_name[name] = gnb_row
        gnb_slice_refs_by_upf[backhaul_upf].update(slice_refs)

    ue_by_name: Dict[str, Dict[str, Any]] = {}
    ue_by_supi: Dict[str, Dict[str, Any]] = {}
    session_refs: Dict[str, Dict[str, Any]] = {}
    for ue_row in ue_rows:
        if not isinstance(ue_row, dict):
            raise TypeError("Each ue entry must be a mapping")
        ue_name = str(ue_row.get("name") or "").strip()
        if not ue_name:
            raise ValueError("Each ue entry must define name")
        if ue_name in ue_by_name:
            raise ValueError(f"Duplicate ue name: {ue_name}")
        supi = _normalize_supi(ue_row.get("supi"))
        if not supi:
            raise ValueError(f"UE {ue_name} must define a valid supi")
        if supi in ue_by_supi:
            raise ValueError(f"Duplicate UE supi: {supi}")
        gnb_name = str(ue_row.get("gnb") or "").strip()
        if gnb_name not in gnb_by_name:
            raise ValueError(f"UE {ue_name} references unknown gNB: {gnb_name}")
        sessions = ue_row.get("sessions") or []
        if not isinstance(sessions, list) or not sessions:
            raise ValueError(f"UE {ue_name} must define at least one session")
        for session_row in sessions:
            if not isinstance(session_row, dict):
                raise TypeError(f"UE {ue_name} session entries must be mappings")
            session_ref = str(session_row.get("session_ref") or "").strip()
            if not session_ref:
                raise ValueError(f"UE {ue_name} session is missing session_ref")
            if session_ref in session_refs:
                raise ValueError(f"Duplicate session_ref: {session_ref}")
            slice_ref = str(session_row.get("slice_ref") or "").strip()
            if slice_ref not in slice_specs:
                raise ValueError(f"UE {ue_name} session references unknown slice: {slice_ref}")
            session_refs[session_ref] = {
                "ue_name": ue_name,
                "supi": supi,
                "slice_ref": slice_ref,
            }
        ue_by_name[ue_name] = ue_row
        ue_by_supi[supi] = ue_row

    app_by_id: Dict[str, Dict[str, Any]] = {}
    for app_row in app_rows:
        if not isinstance(app_row, dict):
            raise TypeError("Each app entry must be a mapping")
        app_id = str(app_row.get("app_id") or "").strip()
        if not app_id:
            raise ValueError("Each app entry must define app_id")
        if app_id in app_by_id:
            raise ValueError(f"Duplicate app_id: {app_id}")
        ue_name = str(app_row.get("ue_name") or "").strip()
        if ue_name not in ue_by_name:
            raise ValueError(f"App {app_id} references unknown UE: {ue_name}")
        supi = _normalize_supi(app_row.get("supi"))
        if supi != _normalize_supi(ue_by_name[ue_name].get("supi")):
            raise ValueError(f"App {app_id} supi does not match UE {ue_name}")
        flow_ids = [str(item).strip() for item in (app_row.get("flow_ids") or []) if str(item).strip()]
        if not flow_ids:
            raise ValueError(f"App {app_id} must declare non-empty flow_ids")
        app_by_id[app_id] = app_row

    flow_by_id: Dict[str, Dict[str, Any]] = {}
    for flow_index, flow_row in enumerate(flow_rows):
        if not isinstance(flow_row, dict):
            raise TypeError("Each flow entry must be a mapping")
        flow_id = str(flow_row.get("flow_id") or "").strip()
        if not flow_id:
            raise ValueError("Each flow entry must define flow_id")
        if flow_id in flow_by_id:
            raise ValueError(f"Duplicate flow_id: {flow_id}")
        app_id = str(flow_row.get("app_id") or "").strip()
        if app_id not in app_by_id:
            raise ValueError(f"Flow {flow_id} references unknown app_id: {app_id}")
        ue_name = str(flow_row.get("ue_name") or "").strip()
        if ue_name not in ue_by_name:
            raise ValueError(f"Flow {flow_id} references unknown UE: {ue_name}")
        supi = _normalize_supi(flow_row.get("supi"))
        if supi != _normalize_supi(app_by_id[app_id].get("supi")):
            raise ValueError(f"Flow {flow_id} supi does not match app {app_id}")
        slice_ref = str(flow_row.get("slice_ref") or "").strip()
        if slice_ref not in slice_specs:
            raise ValueError(f"Flow {flow_id} references unknown slice: {slice_ref}")
        session_ref = str(flow_row.get("session_ref") or "").strip()
        session_spec = session_refs.get(session_ref)
        if session_spec is None:
            raise ValueError(f"Flow {flow_id} references unknown session_ref: {session_ref}")
        if session_spec["ue_name"] != ue_name or session_spec["slice_ref"] != slice_ref:
            raise ValueError(f"Flow {flow_id} session_ref is inconsistent with its UE or slice")
        expected_snssai = slice_specs[slice_ref]["snssai"]
        current_snssai = str(flow_row.get("current_slice_snssai") or expected_snssai).strip()
        if current_snssai != expected_snssai:
            raise ValueError(
                f"Flow {flow_id} current_slice_snssai {current_snssai} does not match slice_ref {slice_ref} ({expected_snssai})"
            )
        flow_by_id[flow_id] = flow_row
        slice_flows[slice_ref].append(flow_row)

    for app_id, app_row in app_by_id.items():
        flow_ids = [str(item).strip() for item in (app_row.get("flow_ids") or []) if str(item).strip()]
        missing_flow_ids = [flow_id for flow_id in flow_ids if flow_id not in flow_by_id]
        if missing_flow_ids:
            raise ValueError(f"App {app_id} references unknown flow_ids: {missing_flow_ids}")
        orphan_flow_ids = [
            flow_id
            for flow_id, flow_row in flow_by_id.items()
            if str(flow_row.get("app_id") or "").strip() == app_id and flow_id not in flow_ids
        ]
        if orphan_flow_ids:
            raise ValueError(f"App {app_id} is missing flow_ids declared by flows table: {orphan_flow_ids}")

    used_suffixes: set = set()
    flow_object_by_id: Dict[str, Flow] = {}
    for flow_index, flow_row in enumerate(flow_rows):
        app_id = str(flow_row["app_id"]).strip()
        slice_ref = str(flow_row["slice_ref"]).strip()
        slice_spec = slice_specs[slice_ref]
        service_type = str(flow_row.get("service_type") or "").strip()
        service_type_id = int(flow_row.get("service_type_id") or _service_type_name_to_id(service_type))
        sla_target = _require_mapping(flow_row, "sla_target")
        supi = _normalize_supi(flow_row.get("supi"))
        flow_object_by_id[str(flow_row["flow_id"]).strip()] = Flow(
            id=_normalize_or_generate_id(flow_row.get("flow_id"), "flow", used_suffixes),
            name=str(flow_row.get("name") or flow_row.get("flow_id") or "").strip(),
            service=FlowService(
                service_type=service_type,
                service_type_id=service_type_id,
            ),
            sla=FlowSLA(
                bandwidth_ul=float(sla_target.get("bandwidth_ul_mbps") or 0.0),
                bandwidth_dl=float(sla_target.get("bandwidth_dl_mbps") or 0.0),
                guaranteed_bandwidth_ul=float(sla_target.get("guaranteed_bandwidth_ul_mbps") or 0.0),
                guaranteed_bandwidth_dl=float(sla_target.get("guaranteed_bandwidth_dl_mbps") or 0.0),
                latency=float(sla_target.get("latency_ms") or 0.0),
                jitter=float(sla_target.get("jitter_ms") or 0.0),
                loss_rate=float(sla_target.get("loss_rate") or 0.0),
                priority=int(sla_target.get("priority") or 1),
            ),
            traffic=FlowTraffic(
                packet_size=float(flow_row.get("packet_size_bytes") or 0.0),
                arrival_rate=float(flow_row.get("arrival_rate_pps") or 0.0),
                five_tuple=_build_deterministic_five_tuple(
                    supi=supi or app_id,
                    flow_ordinal=flow_index + 1,
                    service_type_id=service_type_id,
                ),
            ),
            allocation=FlowAllocation(
                current_slice_snssai=slice_spec["snssai"],
                allocated_bandwidth_ul=float(flow_row.get("allocated_bandwidth_ul_mbps") or 0.0),
                allocated_bandwidth_dl=float(flow_row.get("allocated_bandwidth_dl_mbps") or 0.0),
                optimize_requested=bool(flow_row.get("optimize_requested", False)),
            ),
        )

    apps: List[App] = []
    for app_row in app_rows:
        app_id = str(app_row["app_id"]).strip()
        app_flow_ids = [str(item).strip() for item in app_row.get("flow_ids") or []]
        apps.append(
            App(
                id=_normalize_or_generate_id(app_id, "app", used_suffixes),
                name=str(app_row.get("name") or app_id).strip(),
                supi=_normalize_supi(app_row.get("supi")),
                flows=[flow_object_by_id[flow_id] for flow_id in app_flow_ids],
            )
        )

    slices: List[Slice] = []
    for slice_row in slice_rows:
        label = str(slice_row["label"]).strip()
        slice_spec = slice_specs[label]
        resource = slice_spec["resource"]
        flow_items = slice_flows[label]
        slice_latencies = [float(_require_mapping(flow_item, "sla_target").get("latency_ms") or 0.0) for flow_item in flow_items]
        slice_jitters = [float(_require_mapping(flow_item, "sla_target").get("jitter_ms") or 0.0) for flow_item in flow_items]
        slice_losses = [float(_require_mapping(flow_item, "sla_target").get("loss_rate") or 0.0) for flow_item in flow_items]
        slices.append(
            Slice(
                name=label,
                sst=slice_spec["sst"],
                sd=slice_spec["sd"],
                capacity=SliceCapacity(
                    total_bandwidth_ul=float(resource.get("capacity_ul_mbps") or 0.0),
                    total_bandwidth_dl=float(resource.get("capacity_dl_mbps") or 0.0),
                    reserved_bandwidth_ul=float(resource.get("guaranteed_ul_mbps") or 0.0),
                    reserved_bandwidth_dl=float(resource.get("guaranteed_dl_mbps") or 0.0),
                ),
                load=SliceLoad(
                    current_bandwidth_ul=sum(float(flow_item.get("allocated_bandwidth_ul_mbps") or 0.0) for flow_item in flow_items),
                    current_bandwidth_dl=sum(float(flow_item.get("allocated_bandwidth_dl_mbps") or 0.0) for flow_item in flow_items),
                ),
                qos=SliceQos(
                    latency=min(slice_latencies) if slice_latencies else 0.0,
                    processing_delay=max(0.5, round((min(slice_latencies) if slice_latencies else 0.0) * 0.1, 2)),
                    jitter=min(slice_jitters) if slice_jitters else 0.0,
                    loss_rate=min(slice_losses) if slice_losses else 0.0,
                ),
            )
        )

    nodes: List[Node] = []
    all_slice_snssais = sorted(spec["snssai"] for spec in slice_specs.values())
    for gnb_index, gnb_row in enumerate(gnb_rows):
        gnb_name = str(gnb_row["name"]).strip()
        hosted_slice_refs = [str(item).strip() for item in gnb_row.get("slices") or []]
        hosted_slice_snssais = sorted(slice_specs[slice_ref]["snssai"] for slice_ref in hosted_slice_refs)
        attached_ues = [
            ue_row
            for ue_row in ue_rows
            if str(ue_row.get("gnb") or "").strip() == gnb_name
        ]
        low_latency_slice_count = sum(1 for slice_ref in hosted_slice_refs if slice_specs[slice_ref]["sst"] == 2)
        nodes.append(
            Node(
                id=gnb_index,
                name=gnb_name,
                node_type="AN",
                capacity=NodeCapacity(
                    cpu=900.0 + (len(attached_ues) * 120.0) + (len(hosted_slice_snssais) * 70.0),
                    memory=256.0 + (len(attached_ues) * 48.0),
                    mec=700.0 + (len(hosted_slice_snssais) * 140.0) + (low_latency_slice_count * 80.0),
                    prb=1800.0 + (len(hosted_slice_snssais) * 260.0) + (len(attached_ues) * 90.0),
                ),
                hosted_slice_snssais=hosted_slice_snssais,
            )
        )

    for upf_index, upf_row in enumerate(upf_rows):
        upf_name = str(upf_row["name"]).strip()
        role = str(upf_row.get("role") or "").strip()
        if role == "branching-upf":
            hosted_slice_snssais = sorted(
                slice_specs[slice_ref]["snssai"]
                for slice_ref in gnb_slice_refs_by_upf[upf_name]
            )
        else:
            hosted_slice_snssais = list(all_slice_snssais)
        nodes.append(
            Node(
                id=100 + upf_index,
                name=upf_name,
                node_type="CN",
                capacity=NodeCapacity(
                    cpu=(2600.0 if role == "branching-upf" else 3400.0) + (len(hosted_slice_snssais) * 180.0),
                    memory=(1024.0 if role == "branching-upf" else 1536.0) + (len(hosted_slice_snssais) * 96.0),
                    mec=(900.0 if role == "branching-upf" else 1400.0) + (len(hosted_slice_snssais) * 110.0),
                    prb=0.0,
                ),
                hosted_slice_snssais=hosted_slice_snssais,
            )
        )

    return apps, slices, nodes


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


def sync_latest_flow_five_tuples_to_ue_context() -> Dict[str, int]:
    """
    Rebuild UE flow catalogs from the latest snapshot and refresh PCC flowInfos
    so flowDescription matches the authoritative five_tuple when available.
    """
    return sync_latest_snapshot_flow_catalog_to_ue_context()


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


def _load_latest_graph_scenario_strict() -> Tuple[List[App], List[Slice], List[Node], str]:
    from agents.tools.network_graph import get_latest_graph

    graph = get_latest_graph()
    if graph is None:
        raise RuntimeError("latest graph snapshot not found")

    snapshot_data = graph.to_compatibility_snapshot()
    if not _snapshot_uses_new_schema(snapshot_data):
        raise RuntimeError("latest graph snapshot does not contain a valid scenario payload")

    apps, slices, nodes = _deserialize_scenario(snapshot_data)
    snapshot_id = str(snapshot_data.get("snapshot_id") or "").strip()
    return apps, slices, nodes, snapshot_id


def rebuild_ue_related_tables_from_latest_graph() -> Dict[str, Any]:
    apps, slices, nodes, snapshot_id = _load_latest_graph_scenario_strict()

    with session_scope() as session:
        session.query(UeMobilityEventRecord).delete()
        session.query(UeServingNfBindingRecord).delete()
        session.query(UeAmPolicyAssociationRecord).delete()
        session.query(UeContextRecord).delete()

    seeded = _seed_ue_contexts_from_apps(apps)
    sync_summary = sync_latest_flow_five_tuples_to_ue_context()

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

    if all(key in scenario for key in ["slices", "upfs", "gnbs", "ues", "apps", "flows"]):
        try:
            return _deserialize_experiment_scenario(scenario)
        except Exception:
            return None

    if not all(k in scenario for k in ["apps", "slices", "nodes"]):
        return None

    return _deserialize_scenario(scenario)


def _snapshot_uses_new_schema(snapshot_data: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(snapshot_data, dict):
        return False
    apps = snapshot_data.get("apps")
    slices = snapshot_data.get("slices")
    nodes = snapshot_data.get("nodes")
    if not isinstance(apps, list) or not isinstance(slices, list) or not isinstance(nodes, list):
        return False
    if not apps or not slices or not nodes:
        return True

    first_app = apps[0] if isinstance(apps[0], dict) else {}
    first_flow = (first_app.get("flows") or [{}])[0] if isinstance(first_app, dict) else {}
    first_slice = slices[0] if isinstance(slices[0], dict) else {}
    first_node = nodes[0] if isinstance(nodes[0], dict) else {}

    return all(
        (
            "id" in first_app,
            isinstance(first_flow, dict) and "id" in first_flow and "sla" in first_flow,
            "capacity" in first_slice and "load" in first_slice and "qos" in first_slice,
            "capacity" in first_node and "node_type" in first_node,
        )
    )


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
        Slice("S1_Gold", sst=2, sd="000001", capacity=SliceCapacity(total_bandwidth_ul=120, total_bandwidth_dl=120), load=SliceLoad(), qos=SliceQos(latency=3, processing_delay=1, loss_rate=0.001, jitter=1.5)),
        Slice("S2_Silver", sst=1, sd="000001", capacity=SliceCapacity(total_bandwidth_ul=220, total_bandwidth_dl=220), load=SliceLoad(), qos=SliceQos(latency=10, processing_delay=2, loss_rate=0.01, jitter=8)),
        Slice("S3_Public", sst=1, sd="000002", capacity=SliceCapacity(total_bandwidth_ul=180, total_bandwidth_dl=180), load=SliceLoad(), qos=SliceQos(latency=40, processing_delay=5, loss_rate=0.03, jitter=25)),
        Slice("S4_Platinum", sst=2, sd="000002", capacity=SliceCapacity(total_bandwidth_ul=100, total_bandwidth_dl=80), load=SliceLoad(), qos=SliceQos(latency=2, processing_delay=0.8, loss_rate=0.0005, jitter=1.0)),
        Slice("S5_Massive", sst=3, sd="000001", capacity=SliceCapacity(total_bandwidth_ul=160, total_bandwidth_dl=160), load=SliceLoad(), qos=SliceQos(latency=100, processing_delay=10, loss_rate=0.05, jitter=60)),
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
                id=i,
                name=f"AN_gNB_{i}",
                node_type="AN",
                capacity=NodeCapacity(
                    cpu=1200 + i * 10,
                    memory=250 + i * 32,
                    mec=1200 + an_pref_count * 45 + i * 8,
                    prb=2500 + i * 50,
                ),
                hosted_slice_snssais=hosted,
            )
        )

    for i in range(4):
        hosted = _build_hosted_slices(node_idx=i, node_count=4, target_k=3)
        # eMBB/mMTC(sst=1/3) 在 CN 侧更偏 MEC，下式体现“不同 CN 节点的 MEC 分布差异”
        cn_pref_count = sum(1 for n in hosted if slice_by_snssai[n].sst in (1, 3))
        nodes_data.append(
            Node(
                id=100 + i,
                name=f"CN_UPF_{i}",
                node_type="CN",
                capacity=NodeCapacity(
                    cpu=4200 + i * 60,
                    memory=2024 + i * 128,
                    mec=1800 + cn_pref_count * 35 + i * 15,
                    prb=0,
                ),
                hosted_slice_snssais=hosted,
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
                id=_normalize_or_generate_id(
                    app_dict.get("id", app_dict.get("app_id")),
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
        if _snapshot_uses_new_schema(snapshot_data):
            apps, slices, nodes = _deserialize_scenario(snapshot_data)
            mobility_payload = snapshot_data.get("mobility") if isinstance(snapshot_data, dict) else None
            policy_payload = snapshot_data.get("policy_state") if isinstance(snapshot_data, dict) else None
            seeded = _seed_ue_contexts_from_apps(apps)
            sync_summary = sync_latest_flow_five_tuples_to_ue_context()
            print(f"[UEContext] seeded {seeded} UE records (from snapshot)")
            print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from snapshot)")
            cache_scenario(
                apps,
                slices,
                nodes,
                mobility_payload or _build_mobility_snapshot_payload(apps),
                policy_payload or _build_policy_state_payload(apps),
            )
            return apps, slices, nodes
        apps, slices, nodes = _create_default_scenario()
        mobility_payload = _build_mobility_snapshot_payload(apps)
        policy_payload = _build_policy_state_payload(apps)
        ok = update_scenario_in_db(
            apps,
            slices,
            nodes,
            mobility_data=mobility_payload,
            policy_data=policy_payload,
            trigger="System-Init-ResetLegacySnapshot",
        )
        if not ok:
            raise RuntimeError("Failed to persist reset scenario after detecting legacy snapshot schema")
        seeded = _seed_ue_contexts_from_apps(apps)
        sync_summary = sync_latest_flow_five_tuples_to_ue_context()
        print(f"[UEContext] seeded {seeded} UE records (after legacy reset)")
        print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (after legacy reset)")
        cache_scenario(apps, slices, nodes, mobility_payload, policy_payload)
        return apps, slices, nodes

    apps, slices, nodes = _create_default_scenario()
    mobility_payload = _build_mobility_snapshot_payload(apps)
    policy_payload = _build_policy_state_payload(apps)
    update_scenario_in_db(
        apps,
        slices,
        nodes,
        mobility_data=mobility_payload,
        policy_data=policy_payload,
        trigger="System-Init",
    )
    seeded = _seed_ue_contexts_from_apps(apps)
    sync_summary = sync_latest_flow_five_tuples_to_ue_context()
    print(f"[UEContext] seeded {seeded} UE records (from default)")
    print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from default)")
    cache_scenario(apps, slices, nodes, mobility_payload, policy_payload)
    return apps, slices, nodes


def get_current_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    apps, slices, nodes = get_cached_scenario()
    if apps is not None and slices is not None and nodes is not None:
        return apps, slices, nodes
    return get_initial_scenario()


def get_current_optimizer_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Always prefer the latest graph snapshot for optimizer inputs."""
    try:
        apps, slices, nodes, _snapshot_id = _load_latest_graph_scenario_strict()
        cache_scenario(
            apps,
            slices,
            nodes,
            _build_mobility_snapshot_payload(apps),
            _build_policy_state_payload(apps),
        )
        return apps, slices, nodes
    except Exception as exc:
        logger.warning(f"Failed to load optimizer scenario from latest graph snapshot: {exc}")

    return get_current_scenario()


def initialize_scenario(reset: bool = False, scenario_file: Optional[Union[str, Path]] = None) -> dict:
    """Initialize scenario in cache/DB and return summary.

    Args:
        reset: If True, force-generate default scenario or explicit scenario file and persist as a new snapshot.
               If False, load latest snapshot or create defaults when missing.
        scenario_file: Optional experiment scenario YAML path. Only valid with reset=True.
    """
    resolved_scenario_file = Path(scenario_file).expanduser().resolve() if scenario_file else None
    if resolved_scenario_file is not None and not reset:
        raise ValueError("scenario_file can only be used together with reset=True")

    if reset:
        if resolved_scenario_file is not None:
            scenario_payload = _load_experiment_scenario_payload(resolved_scenario_file)
            apps, slices, nodes = _deserialize_experiment_scenario(scenario_payload)
            trigger = f"Manual-Reset-Scenario:{scenario_payload.get('scenario_id')}"
        else:
            apps, slices, nodes = _create_default_scenario()
            trigger = "Manual-Reset"
        mobility_payload = _build_mobility_snapshot_payload(apps)
        policy_payload = _build_policy_state_payload(apps)
        cache_scenario(apps, slices, nodes, mobility_payload, policy_payload)
        ok = update_scenario_in_db(
            apps,
            slices,
            nodes,
            mobility_data=mobility_payload,
            policy_data=policy_payload,
            trigger=trigger,
        )
        if not ok:
            raise RuntimeError("Failed to persist reset scenario into DB")
        seeded = _seed_ue_contexts_from_apps(apps)
        sync_summary = sync_latest_flow_five_tuples_to_ue_context()
        print(f"[UEContext] seeded {seeded} UE records (from reset)")
        print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from reset)")
    else:
        apps, slices, nodes = get_initial_scenario()

    an_count = sum(1 for n in nodes if getattr(n, "node_type", "") == "AN")
    cn_count = sum(1 for n in nodes if getattr(n, "node_type", "") == "CN")
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
        "scenario_file": str(resolved_scenario_file) if resolved_scenario_file else "",
    }


def init_main() -> None:
    parser = argparse.ArgumentParser(description="Initialize scenario cache and DB snapshot")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force reset to generated default scenario and persist a new snapshot",
    )
    parser.add_argument(
        "--scenario-file",
        default="",
        help="Load an experiment scenario YAML and persist it as the initial state. Requires --reset.",
    )
    args = parser.parse_args()

    summary = initialize_scenario(
        reset=args.reset,
        scenario_file=str(args.scenario_file or "").strip() or None,
    )
    print("Scenario initialized:")
    print(summary)


if __name__ == "__main__":
    init_main()
