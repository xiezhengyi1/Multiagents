import json
from typing import Optional

from langchain.tools import ToolRuntime, tool

from shared.tools.wrapper_think import tool_with_reason

from shared.runtime import AgentRuntimeContext
from dataclasses import asdict
from .init_scenario import get_current_scenario
from .network_graph import (
    NetworkGraph,
    get_graph_snapshot_payload,
)

@tool_with_reason
def save_network_status_snapshot(
    trigger_event: str = "Manual",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Take a snapshot of the current network status (Apps, Slices, Nodes) and save it to the history database.
    Useful for tracking system changes before/after optimization or periodically.
    
    Args:
        trigger_event: The reason for taking this snapshot (e.g., 'Before-Optimization', 'Periodic-Monitor').
        
    Returns:
        Status message indicating success or failure.
    """
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}Disabled: agent tools must read an existing network graph by snapshot_id and must not create graph snapshots."

def get_network_status_full():
    """
    [底层工具专用] 获取当前网络全量对象。
    直接返回(apps, slices, nodes)元组，供下游工具内部拼装。
    """
    return get_current_scenario()

def _load_graph_snapshot(snapshot_id: Optional[str]) -> Optional[NetworkGraph]:
    target_snapshot_id = str(snapshot_id or "").strip()
    if not target_snapshot_id:
        raise ValueError("network status tools require snapshot_id")
    payload = get_graph_snapshot_payload(target_snapshot_id)
    if not isinstance(payload, dict):
        return None
    return NetworkGraph.from_payload(payload)


def get_network_status_summary(flow_type_id: int = None, snapshot_id: Optional[str] = None) -> str:
    """
    [智能体专用] 获取当前网络切片和应用流的状态摘要。
    只返回切片的资源使用率和业务数量，以及简化的 App 列表，大幅减少 Token 消耗。
    """
    requested_snapshot_id = str(snapshot_id or "").strip()
    if not requested_snapshot_id:
        raise ValueError("get_network_status_summary requires snapshot_id")
    graph = _load_graph_snapshot(requested_snapshot_id)
    if requested_snapshot_id and graph is None:
        raise RuntimeError(f"Network graph snapshot not found: snapshot_id={requested_snapshot_id}")
    if graph is not None:
        snapshot = graph.to_compatibility_snapshot()
        apps = snapshot.get("apps", [])
        slices = snapshot.get("slices", [])
        nodes = snapshot.get("nodes", [])
    else:
        apps, slices, nodes = get_current_scenario()
        apps = [asdict(item) for item in apps]
        slices = [asdict(item) for item in slices]
        nodes = [asdict(item) for item in nodes]
    
    slice_status_list = []
    
    for s in slices:
        # 如果指定了 flow_type_id，则只处理匹配的切片
        sst = s.get("sst") if isinstance(s, dict) else getattr(s, "sst", None)
        if flow_type_id is not None and sst != flow_type_id:
            continue
            
        # 简化的占用统计
        slice_snssai = s.get("snssai") if isinstance(s, dict) else getattr(s, "snssai", None)
        dynamic_used_ul = sum(
            float(((flow.get("allocation") or {}).get("allocated_bandwidth_ul")) or 0.0)
            for a in apps
            for flow in (a.get("flows", []) if isinstance(a, dict) else [])
            if str(((flow.get("allocation") or {}).get("current_slice_snssai")) or "").strip() == str(slice_snssai or "").strip()
        )
        dynamic_used_dl = sum(
            float(((flow.get("allocation") or {}).get("allocated_bandwidth_dl")) or 0.0)
            for a in apps
            for flow in (a.get("flows", []) if isinstance(a, dict) else [])
            if str(((flow.get("allocation") or {}).get("current_slice_snssai")) or "").strip() == str(slice_snssai or "").strip()
        )
        active_flows_count = sum(
            1
            for a in apps
            for flow in (a.get("flows", []) if isinstance(a, dict) else [])
            if str(((flow.get("allocation") or {}).get("current_slice_snssai")) or "").strip() == str(slice_snssai or "").strip()
        )
        
        capacity = s.get("capacity", {}) if isinstance(s, dict) else getattr(s, "capacity", None)
        load = s.get("load", {}) if isinstance(s, dict) else getattr(s, "load", None)
        guaranteed_ul = float((capacity.get("guaranteed_bandwidth_ul") if isinstance(capacity, dict) else getattr(capacity, "guaranteed_bandwidth_ul", 0.0)) or 0.0)
        guaranteed_dl = float((capacity.get("guaranteed_bandwidth_dl") if isinstance(capacity, dict) else getattr(capacity, "guaranteed_bandwidth_dl", 0.0)) or 0.0)
        background_ul = float((load.get("current_bandwidth_ul") if isinstance(load, dict) else getattr(load, "current_bandwidth_ul", 0.0)) or 0.0)
        background_dl = float((load.get("current_bandwidth_dl") if isinstance(load, dict) else getattr(load, "current_bandwidth_dl", 0.0)) or 0.0)

        # 使用率口径只统计当前切片上的实际业务分配。
        # `guaranteed_bandwidth_*` 表示切片自身拿到的保底能力，不是已占用负载。
        # `load.current_bandwidth_*` 单独作为背景负载透出，避免与 flow allocation 重复计数。
        total_used_ul = dynamic_used_ul
        total_used_dl = dynamic_used_dl
        
        total_bw_ul = float((capacity.get("total_bandwidth_ul") if isinstance(capacity, dict) else getattr(capacity, "total_bandwidth_ul", 0.0)) or 0.0)
        total_bw_dl = float((capacity.get("total_bandwidth_dl") if isinstance(capacity, dict) else getattr(capacity, "total_bandwidth_dl", 0.0)) or 0.0)
        utilization_ul = (total_used_ul / total_bw_ul * 100) if total_bw_ul > 0 else 0.0
        utilization_dl = (total_used_dl / total_bw_dl * 100) if total_bw_dl > 0 else 0.0
        
        slice_status_list.append({
            "name": s.get("name") if isinstance(s, dict) else getattr(s, "name", None),
            "snssai": slice_snssai,
            "sst": sst,
            "usage_ul_pct": round(utilization_ul, 1),
            "usage_dl_pct": round(utilization_dl, 1),
            "allocated_ul_mbps": round(dynamic_used_ul, 3),
            "allocated_dl_mbps": round(dynamic_used_dl, 3),
            "guaranteed_ul_mbps": round(guaranteed_ul, 3),
            "guaranteed_dl_mbps": round(guaranteed_dl, 3),
            "background_ul_mbps": round(background_ul, 3),
            "background_dl_mbps": round(background_dl, 3),
            "active_flows": active_flows_count,
            "latency_sla": (s.get("qos", {}) if isinstance(s, dict) else getattr(s, "qos", None)).get("latency") if isinstance((s.get("qos", {}) if isinstance(s, dict) else getattr(s, "qos", None)), dict) else getattr(getattr(s, "qos", None), "latency", None)
        })

    # 简化的应用列表
    app_summary_list = []
    for a in apps:
        flows = a.get("flows", []) if isinstance(a, dict) else []
        if flow_type_id is not None and any(int(((flow.get("service") or {}).get("service_type_id")) or 0) != flow_type_id for flow in flows):
            continue
        app_summary_list.append({
            "app_id": a.get("id") if isinstance(a, dict) else None,
            "app_name": a.get("name") if isinstance(a, dict) else None,
            "flow_count": len(flows),
            "total_bw_mbps": round(sum(float(((flow.get("sla") or {}).get("bandwidth_ul")) or 0.0) + float(((flow.get("sla") or {}).get("bandwidth_dl")) or 0.0) for flow in flows), 2)
        })
        
    return json.dumps({
        "slices": slice_status_list,
        "apps": app_summary_list
    }, ensure_ascii=False, indent=2)

# 兼容旧代码调用
get_network_status = get_network_status_full

if __name__ == "__main__":
    save_network_status_snapshot(trigger_event="Manual-Test")


