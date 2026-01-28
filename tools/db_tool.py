from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import asdict
from contextlib import contextmanager
import logging
import json

from database.connection import SessionLocal
from database.models import SemanticKnowledge
from .models import App, Flow, Slice, Node

logger = logging.getLogger(__name__)

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
    """Get scenario data. Priority: DB -> Default (and save to DB)."""
    db_data: Dict[str, Any] = {}
    keys = ["config:apps", "config:slices", "config:nodes"]

    try:
        with session_scope() as session:
            for k in keys:
                rec = session.query(SemanticKnowledge).filter_by(key=k).first()
                if rec and rec.value:
                    db_data[k.replace("config:", "")] = rec.value
    except Exception as e:
        logger.warning(f"DB access failed, using defaults. Error: {e}")
        apps, slices, nodes = _create_default_scenario()
        cache_scenario(apps, slices, nodes)
        return apps, slices, nodes

    if all(k in db_data for k in ["apps", "slices", "nodes"]):
        logger.info("Loaded scenario from SemanticKnowledge.")
        apps, slices, nodes = _deserialize_scenario(db_data)
        cache_scenario(apps, slices, nodes)
        return apps, slices, nodes

    logger.info("Initializing SemanticKnowledge with default scenario...")
    apps, slices, nodes = _create_default_scenario()
    serialized = _serialize_scenario_for_db(apps, slices, nodes)

    try:
        with session_scope() as session:
            if not session.query(SemanticKnowledge).filter_by(key="config:apps").first():
                session.add(SemanticKnowledge(key="config:apps", category="optimizer_config", value=serialized["apps"], description="Initial Apps Def"))

            if not session.query(SemanticKnowledge).filter_by(key="config:slices").first():
                session.add(SemanticKnowledge(key="config:slices", category="optimizer_config", value=serialized["slices"], description="Initial Slices Def"))

            if not session.query(SemanticKnowledge).filter_by(key="config:nodes").first():
                session.add(SemanticKnowledge(key="config:nodes", category="optimizer_config", value=serialized["nodes"], description="Initial Nodes Def"))
    except Exception as e:
        logger.error(f"Failed to save defaults: {e}")

    cache_scenario(apps, slices, nodes)
    return apps, slices, nodes


def get_current_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Prefer cached scenario, fall back to DB/default initialization."""
    apps, slices, nodes = get_cached_scenario()
    if apps and slices and nodes:
        return apps, slices, nodes
    return get_initial_scenario()


def update_scenario_in_db(apps: List[App], slices: List[Slice], nodes: List[Node]) -> bool:
    """Persist scenario to DB (SemanticKnowledge)."""
    serialized = _serialize_scenario_for_db(apps, slices, nodes)

    try:
        with session_scope() as session:
            for key, value in [
                ("config:apps", serialized["apps"]),
                ("config:slices", serialized["slices"]),
                ("config:nodes", serialized["nodes"])
            ]:
                rec = session.query(SemanticKnowledge).filter_by(key=key).first()
                if rec:
                    rec.value = value
                    rec.category = rec.category or "optimizer_config"
                else:
                    session.add(SemanticKnowledge(key=key, category="optimizer_config", value=value, description="Updated by optimizer"))
        return True
    except Exception as e:
        logger.error(f"Failed to update scenario in DB: {e}")
        return False
