import json
from uuid import uuid4
from typing import Optional

from langchain.tools import ToolRuntime, tool

from agent_runtime import AgentRuntimeContext
from tools.init_scenario import get_current_scenario
from dataclasses import asdict
from tools.network_graph import build_and_persist_graph_from_scenario, get_latest_graph

@tool
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
    try:
        apps, slices, nodes = get_current_scenario()
        snapshot_id = f"graph-{uuid4()}"
        build_and_persist_graph_from_scenario(
            apps=apps,
            slices=slices,
            nodes=nodes,
            snapshot_id=snapshot_id,
            trigger_event=str(trigger_event),
        )
            
        prefix = ""
        if runtime is not None:
            ctx = runtime.context
            prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
        return f"{prefix}Success: Network graph snapshot saved. snapshot_id={snapshot_id}"
    except Exception as e:
        return f"Error saving snapshot: {str(e)}"

def get_network_status_full():
    """
    [底层工具专用] 获取当前网络全量对象。
    直接返回(apps, slices, nodes)元组，供下游工具内部拼装。
    """
    return get_current_scenario()

def get_network_status_summary(flow_type_id: int = None) -> str:
    """
    [智能体专用] 获取当前网络切片和应用流的状态摘要。
    只返回切片的资源使用率和业务数量，以及简化的 App 列表，大幅减少 Token 消耗。
    """
    graph = get_latest_graph()
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
            float(flow.get("bw_ul") or 0.0)
            for a in apps
            for flow in (a.get("flows", []) if isinstance(a, dict) else [])
            if str(flow.get("old_slice") or "").strip() == str(slice_snssai or "").strip()
        )
        dynamic_used_dl = sum(
            float(flow.get("bw_dl") or 0.0)
            for a in apps
            for flow in (a.get("flows", []) if isinstance(a, dict) else [])
            if str(flow.get("old_slice") or "").strip() == str(slice_snssai or "").strip()
        )
        active_flows_count = sum(
            1
            for a in apps
            for flow in (a.get("flows", []) if isinstance(a, dict) else [])
            if str(flow.get("old_slice") or "").strip() == str(slice_snssai or "").strip()
        )
        
        total_used_ul = float((s.get("current_load_bw_ul") if isinstance(s, dict) else getattr(s, "current_load_bw_ul", 0.0)) or 0.0) + float((s.get("reserved_bw") if isinstance(s, dict) else getattr(s, "reserved_bw", 0.0)) or 0.0) + dynamic_used_ul
        total_used_dl = float((s.get("current_load_bw_dl") if isinstance(s, dict) else getattr(s, "current_load_bw_dl", 0.0)) or 0.0) + float((s.get("reserved_bw") if isinstance(s, dict) else getattr(s, "reserved_bw", 0.0)) or 0.0) + dynamic_used_dl
        
        total_bw_ul = float((s.get("total_bw_ul") if isinstance(s, dict) else getattr(s, "total_bw_ul", 0.0)) or 0.0)
        total_bw_dl = float((s.get("total_bw_dl") if isinstance(s, dict) else getattr(s, "total_bw_dl", 0.0)) or 0.0)
        utilization_ul = (total_used_ul / total_bw_ul * 100) if total_bw_ul > 0 else 0.0
        utilization_dl = (total_used_dl / total_bw_dl * 100) if total_bw_dl > 0 else 0.0
        
        slice_status_list.append({
            "name": s.get("name") if isinstance(s, dict) else getattr(s, "name", None),
            "snssai": slice_snssai,
            "sst": sst,
            "usage_ul_pct": round(utilization_ul, 1),
            "usage_dl_pct": round(utilization_dl, 1),
            "active_flows": active_flows_count,
            "latency_sla": s.get("latency") if isinstance(s, dict) else getattr(s, "latency", None)
        })

    # 简化的应用列表
    app_summary_list = []
    for a in apps:
        flows = a.get("flows", []) if isinstance(a, dict) else []
        if flow_type_id is not None and any(int(flow.get("service_type_id") or 0) != flow_type_id for flow in flows):
            continue
        app_summary_list.append({
            "app_id": a.get("app_id") if isinstance(a, dict) else None,
            "app_name": a.get("name") if isinstance(a, dict) else None,
            "flow_count": len(flows),
            "total_bw_mbps": round(sum(float(flow.get("bw_ul") or 0.0) + float(flow.get("bw_dl") or 0.0) for flow in flows), 2)
        })
        
    return json.dumps({
        "slices": slice_status_list,
        "apps": app_summary_list
    }, ensure_ascii=False, indent=2)

# 兼容旧代码调用
get_network_status = get_network_status_full

if __name__ == "__main__":
    save_network_status_snapshot(trigger_event="Manual-Test")
