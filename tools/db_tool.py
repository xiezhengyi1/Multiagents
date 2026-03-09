from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import logging
import json

from database.connection import SessionLocal
from database.models import SemanticKnowledge, NetworkStatusSnapshot
import sys
import os

try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback if running relative
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Flow:
    """定义应用内的单个数据流需求"""
    name: str # 保留作为描述性名称
    flow_id: str # 流唯一标识
    bw_ul: float       # 上行带宽 (Mbps)
    bw_dl: float       # 下行带宽 (Mbps)
    gbr_ul: float      # 上行保证比特率 (Mbps)
    gbr_dl: float      # 下行保证比特率 (Mbps)
    lat: float      # 时延 (ms)
    loss_req: float # 丢包率上限 (0~1)
    jitter_req: float # 抖动上限 (ms)
    priority: int   # 优先级 (数值越小越高)
    old_slice: Optional[str] = None # 流的原切片名称 (实为 S-NSSAI)
    old_allocated_bw_ul: Optional[float] = None # 上一次分配的实际上行带宽
    old_allocated_bw_dl: Optional[float] = None # 上一次分配的实际下行带宽
    supi: Optional[str] = None # 用户标识 (如 imsi-...)

@dataclass
class App:
    """定义应用及其聚合需求"""
    name: str # 保留作为描述性名称
    app_id: str # 应用唯一标识
    flows: List[Flow]
    total_bw_ul: float = field(init=False)
    total_bw_dl: float = field(init=False)
    min_lat: float = field(init=False)
    max_prio: int = field(init=False)

    def __post_init__(self):
        if not self.flows:
            self.total_bw_ul = 0.0
            self.total_bw_dl = 0.0
            self.min_lat = float('inf')
            self.max_prio = 0
        else:
            self.total_bw_ul = sum(f.bw_ul for f in self.flows)
            self.total_bw_dl = sum(f.bw_dl for f in self.flows)
            self.min_lat = min(f.lat for f in self.flows)
            self.max_prio = max(f.priority for f in self.flows)

@dataclass
class Slice:
    """定义网络切片资源与状态"""
    name: str # 描述性名称
    sst: int        # 切片服务类型
    sd: str         # 切片微分器
    snssai: str = field(init=False) # 唯一标识 (SST-SD), 自动生成
    total_bw_ul: float # 总带宽容量
    total_bw_dl: float # 总带宽容量
    current_load_bw_ul: float # 当前基础负载
    current_load_bw_dl: float # 当前基础负载
    latency: float  # 链路传输时延
    proc_delay: float # 处理时延
    loss: float # 切片丢包率 (0~1)
    jitter: float # 切片抖动 (ms)
    reserved_bw: float # 不可抢占的保留带宽

    def __post_init__(self):
        # 自动生成 snssai 标识
        self.snssai = f"{self.sst:02X}{self.sd}"

@dataclass
class Node:
    """定义物理节点资源"""
    name: str
    cpu_capacity: float
    memory_capacity: float # 内存容量
    slices_hosted: List[str] # 节点托管的切片列表


# 简单缓存：仅作为优化-下发-提交之间的短期桥接
_SCENARIO_CACHE: Dict[str, Optional[List[Any]]] = {
    "apps": None,
    "slices": None,
    "nodes": None
}

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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


def _serialize_scenario_for_db(apps: List[App], slices: List[Slice], nodes: List[Node]) -> Dict[str, Any]:
    """Serialize scenario data for DB storage (slice name removed)."""
    apps_data = [asdict(app) for app in apps]

    slices_data = []
    for s in slices:
        s_dict = asdict(s)
        if "name" in s_dict:
            del s_dict["name"]
        slices_data.append(s_dict)

    nodes_data = [asdict(n) for n in nodes]

    return {
        "apps": apps_data,
        "slices": slices_data,
        "nodes": nodes_data
    }


def serialize_scenario_for_api(apps: List[App], slices: List[Slice], nodes: List[Node]) -> Dict[str, Any]:
    """Serialize scenario data for API/tool output (keep slice name)."""
    return {
        "apps": [asdict(app) for app in apps],
        "slices": [asdict(s) for s in slices],
        "nodes": [asdict(n) for n in nodes]
    }


def _deserialize_scenario(data: Dict[str, Any]) -> Tuple[List[App], List[Slice], List[Node]]:
    """Reconstruct objects from DB data."""
    apps: List[App] = []
    for app_dict in data.get("apps", []):
        flows = []
        for f_dict in app_dict.get("flows", []):
            valid_keys = Flow.__annotations__.keys()
            flow_kwargs = {k: v for k, v in f_dict.items() if k in valid_keys}
            flows.append(Flow(**flow_kwargs))

        valid_app_keys = ["name", "app_id"]
        app_kwargs = {k: v for k, v in app_dict.items() if k in valid_app_keys}
        apps.append(App(flows=flows, **app_kwargs))

    slices: List[Slice] = []
    for s_dict in data.get("slices", []):
        sst = s_dict.get("sst", 0)
        sd = s_dict.get("sd", "")
        generated_name = f"Slice_{sst}_{sd}"

        valid_slice_keys = [k for k in Slice.__annotations__ if k != "snssai"]
        slice_kwargs = {k: v for k, v in s_dict.items() if k in valid_slice_keys}

        if "name" not in slice_kwargs:
            slice_kwargs["name"] = generated_name

        slices.append(Slice(**slice_kwargs))

    nodes: List[Node] = []
    for n_dict in data.get("nodes", []):
        nodes.append(Node(**n_dict))

    return apps, slices, nodes


def deserialize_scenario_payload(payload: Union[str, Dict[str, Any]]) -> Optional[Tuple[List[App], List[Slice], List[Node]]]:
    """Deserialize scenario data from a tool payload (dict/JSON)."""
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
    """Generate hardcoded default data with supi."""

    def create_app(name, app_id, flows, base_supi_seed):
        for i, f in enumerate(flows):
            if f.flow_id == "f_default":
                f.flow_id = f"{app_id}_f{i+1}_{f.name}"
            else:
                f.flow_id = f"{app_id}_{f.flow_id}"

            f.old_allocated_bw_ul = f.bw_ul
            f.old_allocated_bw_dl = f.bw_dl
            f.supi = f"imsi-{base_supi_seed}-{i:04d}"

        return App(name=name, app_id=app_id, flows=flows)

    apps_data = [
        create_app("Remote_Drive", "app_remote_drive", [
            Flow("Control", "f_control", 2, 2, 0.5, 0.5, 5, 0.001, 1, 20, old_slice="02000001"),
            Flow("Video_Feed", "f_video_feed", 8, 8, 4, 4, 20, 0.01, 5, 15, old_slice="02000001")
        ], "20893001"),

        create_app("4K_Video", "app_4k_video", [
            Flow("Main_Stream", "f_main_stream", 35, 30, 10, 10, 50, 0.02, 10, 10, old_slice="01000001"),
            Flow("Audio", "f_audio", 5, 5, 1, 1, 100, 0.01, 20, 5, old_slice="01000001")
        ], "20893002"),

        create_app("IoT_Sensor", "app_iot_sensor", [
            Flow("Telemetry", "f_telemetry", 2, 2, 0.1, 0.1, 20, 0.005, 5, 10, old_slice="02000001")
        ], "20893003"),

        create_app("Web_Browse", "app_web_browse", [
            Flow("HTTP", "f_http", 15, 20, 1, 1, 100, 0.03, 30, 1, old_slice="01000002")
        ], "20893004"),

        create_app("AR_Gaming", "app_ar_gaming", [
            Flow("Render", "f_render", 20, 15, 5, 5, 20, 0.01, 5, 15, old_slice="01000001"),
            Flow("Sync", "f_sync", 5, 6, 2, 2, 15, 0.005, 3, 15, old_slice="01000001")
        ], "20893005"),

        create_app("Factory_Robot", "app_factory_robot", [
            Flow("Motion_Cmd", "f_motion_cmd", 5, 5, 2.5, 2.5, 5, 0.0001, 1, 100, old_slice="02000001")
        ], "20893006"),

        create_app("Smart_Meter", "app_smart_meter", [
            Flow("Data_Report", "f_data_report", 0.5, 0.5, 0.05, 0.05, 200, 0.05, 50, 1, old_slice="01000002")
        ], "20893007")
    ]

    slices_data = [
        Slice("S1_Gold", sst=2, sd="000001", total_bw_ul=100, total_bw_dl=100, current_load_bw_ul=0, current_load_bw_dl=0, latency=3, proc_delay=1, loss=0.001, jitter=1.5, reserved_bw=20),
        Slice("S2_Silver", sst=1, sd="000001", total_bw_ul=200, total_bw_dl=200, current_load_bw_ul=0, current_load_bw_dl=0, latency=10, proc_delay=2, loss=0.01, jitter=8, reserved_bw=50),
        Slice("S3_Public", sst=1, sd="000002", total_bw_ul=150, total_bw_dl=150, current_load_bw_ul=0, current_load_bw_dl=0, latency=40, proc_delay=5, loss=0.03, jitter=25, reserved_bw=10),
        Slice("S4_Platinum", sst=2, sd="000002", total_bw_ul=50, total_bw_dl=50, current_load_bw_ul=0, current_load_bw_dl=0, latency=1, proc_delay=0.5, loss=0.0005, jitter=0.8, reserved_bw=5),
        Slice("S5_Massive", sst=3, sd="000001", total_bw_ul=30, total_bw_dl=30, current_load_bw_ul=0, current_load_bw_dl=0, latency=100, proc_delay=10, loss=0.05, jitter=60, reserved_bw=2)
    ]

    nodes_data = [
        Node("Node_Edge", cpu_capacity=100, memory_capacity=200, slices_hosted=["S1_Gold", "S2_Silver", "S4_Platinum"]),
        Node("Node_Core", cpu_capacity=300, memory_capacity=1000, slices_hosted=["S3_Public", "S5_Massive"])
    ]

    return apps_data, slices_data, nodes_data


def get_initial_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Get scenario data. Priority: DB (NetworkStatusSnapshot) -> Default (and save to DB)."""
    
    # 1. Try to load latest snapshot from NetworkStatusSnapshot
    try:
        with session_scope() as session:
            # Get the latest snapshot by timestamp
            latest_snapshot = session.query(NetworkStatusSnapshot).order_by(NetworkStatusSnapshot.timestamp.desc()).first()
            
            if latest_snapshot and latest_snapshot.snapshot_data:
                logger.info(f"Loaded scenario from NetworkStatusSnapshot (Timestamp: {latest_snapshot.timestamp}).")
                snapshot_data = latest_snapshot.snapshot_data
                apps, slices, nodes = _deserialize_scenario(snapshot_data)
                cache_scenario(apps, slices, nodes)
                return apps, slices, nodes
                
    except Exception as e:
        logger.warning(f"Failed to load form NetworkStatusSnapshot: {e}")

    # 2. Fallback to initialize defaults and save them
    logger.info("Initializing NetworkStatusSnapshot with default scenario...")
    apps, slices, nodes = _create_default_scenario()
    serialized = _serialize_scenario_for_db(apps, slices, nodes)

    try:
        with session_scope() as session:
            # Save the initial state as a snapshot
            snapshot = NetworkStatusSnapshot(
                snapshot_data=serialized,
                trigger_event="System-Init"
            )
            session.add(snapshot)
    except Exception as e:
        logger.error(f"Failed to save default snapshot: {e}")

    cache_scenario(apps, slices, nodes)
    return apps, slices, nodes


def get_current_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Prefer cached scenario, fall back to DB/default initialization."""
    apps, slices, nodes = get_cached_scenario()
    if apps and slices and nodes:
        return apps, slices, nodes
    return get_initial_scenario()


def update_scenario_in_db(apps: List[App], slices: List[Slice], nodes: List[Node], trigger: str = "Optimization-Result") -> bool:
    """
    Persist scenario state as a new NetworkStatusSnapshot.
    Now we treat network state as time-series snapshots instead of just updating a single config row.
    """
    serialized = _serialize_scenario_for_db(apps, slices, nodes)

    try:
        with session_scope() as session:
            snapshot = NetworkStatusSnapshot(
                snapshot_data=serialized,
                trigger_event=trigger
            )
            session.add(snapshot)
            # We can also keep updating SemanticKnowledge if other components rely on 'current_config'
            # But primarily we use snapshots now.
        return True
    except Exception as e:
        logger.error(f"Failed to save scenario snapshot: {e}")
        return False
