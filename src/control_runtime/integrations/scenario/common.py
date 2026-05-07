from __future__ import annotations

import re
import secrets
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..optimizer.models import (
    App,
    Flow,
    FlowAllocation,
    FlowService,
    FlowSLA,
    FlowTelemetry,
    FlowTraffic,
    Node,
    NodeCapacity,
    NodeTelemetry,
    Slice,
    SliceCapacity,
    SliceLoad,
    SliceQos,
    SliceTelemetry,
)


def snapshot_uses_new_schema(snapshot_data: Optional[Dict[str, Any]]) -> bool:
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


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_slice_current_bandwidth(
    *,
    load_payload: Dict[str, Any],
    telemetry_payload: Dict[str, Any],
    load_key: str,
    utilization_key: str,
    total_bandwidth: float,
    legacy_load_value: Any,
    legacy_utilization_value: Any,
) -> float:
    explicit_load = _optional_float(load_payload.get(load_key))
    if explicit_load is not None:
        return explicit_load

    telemetry_load = _optional_float(telemetry_payload.get(load_key))
    if telemetry_load is not None:
        return telemetry_load

    utilization = _optional_float(telemetry_payload.get(utilization_key))
    if utilization is None:
        utilization = _optional_float(legacy_utilization_value)
    if utilization is not None:
        return utilization * total_bandwidth

    legacy_load = _optional_float(legacy_load_value)
    if legacy_load is not None:
        return legacy_load
    return 0.0


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
    total_bandwidth_ul = float(capacity_payload.get("total_bandwidth_ul", s_dict.get("total_bw_ul", 0.0)) or 0.0)
    total_bandwidth_dl = float(capacity_payload.get("total_bandwidth_dl", s_dict.get("total_bw_dl", 0.0)) or 0.0)
    guaranteed_bandwidth_ul = float(capacity_payload.get("guaranteed_bandwidth_ul", 0.0) or 0.0)
    guaranteed_bandwidth_dl = float(capacity_payload.get("guaranteed_bandwidth_dl", 0.0) or 0.0)
    current_bandwidth_ul = _resolve_slice_current_bandwidth(
        load_payload=load_payload,
        telemetry_payload=telemetry_payload,
        load_key="current_bandwidth_ul",
        utilization_key="utilization_ul",
        total_bandwidth=total_bandwidth_ul,
        legacy_load_value=s_dict.get("current_load_bw_ul"),
        legacy_utilization_value=s_dict.get("sim_utilization_ul"),
    )
    current_bandwidth_dl = _resolve_slice_current_bandwidth(
        load_payload=load_payload,
        telemetry_payload=telemetry_payload,
        load_key="current_bandwidth_dl",
        utilization_key="utilization_dl",
        total_bandwidth=total_bandwidth_dl,
        legacy_load_value=s_dict.get("current_load_bw_dl"),
        legacy_utilization_value=s_dict.get("sim_utilization_dl"),
    )

    return Slice(
        name=str(s_dict.get("name", generated_name)),
        sst=int(sst or 0),
        sd=str(sd or "000000"),
        capacity=SliceCapacity(
            total_bandwidth_ul=total_bandwidth_ul,
            total_bandwidth_dl=total_bandwidth_dl,
            guaranteed_bandwidth_ul=guaranteed_bandwidth_ul,
            guaranteed_bandwidth_dl=guaranteed_bandwidth_dl,
        ),
        load=SliceLoad(
            current_bandwidth_ul=current_bandwidth_ul,
            current_bandwidth_dl=current_bandwidth_dl,
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
    "snapshot_id": None,
}


def cache_scenario(
    apps: List[App],
    slices: List[Slice],
    nodes: List[Node],
    mobility: Optional[List[Dict[str, Any]]] = None,
    policy_state: Optional[Dict[str, Any]] = None,
    snapshot_id: str = "",
) -> None:
    _SCENARIO_CACHE["apps"] = apps
    _SCENARIO_CACHE["slices"] = slices
    _SCENARIO_CACHE["nodes"] = nodes
    _SCENARIO_CACHE["mobility"] = mobility or []
    _SCENARIO_CACHE["policy_state"] = policy_state or {}
    _SCENARIO_CACHE["snapshot_id"] = str(snapshot_id or "").strip() or None


def get_cached_scenario() -> Tuple[Optional[List[App]], Optional[List[Slice]], Optional[List[Node]]]:
    return _SCENARIO_CACHE.get("apps"), _SCENARIO_CACHE.get("slices"), _SCENARIO_CACHE.get("nodes")


def get_cached_control_scenario() -> Dict[str, Any]:
    return {
        "apps": _SCENARIO_CACHE.get("apps"),
        "slices": _SCENARIO_CACHE.get("slices"),
        "nodes": _SCENARIO_CACHE.get("nodes"),
        "mobility": _SCENARIO_CACHE.get("mobility") or [],
        "policy_state": _SCENARIO_CACHE.get("policy_state") or {},
        "snapshot_id": _SCENARIO_CACHE.get("snapshot_id") or "",
    }


def clear_cached_scenario() -> None:
    _SCENARIO_CACHE["apps"] = None
    _SCENARIO_CACHE["slices"] = None
    _SCENARIO_CACHE["nodes"] = None
    _SCENARIO_CACHE["mobility"] = None
    _SCENARIO_CACHE["policy_state"] = None
    _SCENARIO_CACHE["snapshot_id"] = None


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


