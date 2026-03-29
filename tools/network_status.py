import pandas as pd
import json
from typing import Optional

from langchain.tools import ToolRuntime, tool

from agent_runtime import AgentRuntimeContext
from database.connection import SessionLocal
from database.models import NetworkStatusSnapshot
from tools.db_tool import _serialize_scenario_for_db
from tools.init_scenario import get_current_scenario
from dataclasses import asdict

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
        
        # Serialize the full state
        # Note: Ideally this should include richer metrics calculated in get_network_status
        # For now we save the raw configuration state which contains load info
        data = _serialize_scenario_for_db(apps, slices, nodes)
        
        # Save to DB
        session = SessionLocal()
        try:
            snapshot = NetworkStatusSnapshot(
                app_data=data.get("apps", []),
                slice_data=data.get("slices", []),
                node_data=data.get("nodes", []),
                trigger_event=str(trigger_event)
            )
            session.add(snapshot)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
        prefix = ""
        if runtime is not None:
            ctx = runtime.context
            prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
        return f"{prefix}Success: Network status snapshot saved."
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
    apps, slices, nodes = get_current_scenario()
    
    slice_status_list = []
    
    for s in slices:
        # 如果指定了 flow_type_id，则只处理匹配的切片
        if flow_type_id is not None and s.sst != flow_type_id:
            continue
            
        # 简化的占用统计
        dynamic_used_ul = sum(f.bw_ul for a in apps for f in a.flows if f.old_slice == s.snssai)
        dynamic_used_dl = sum(f.bw_dl for a in apps for f in a.flows if f.old_slice == s.snssai)
        active_flows_count = sum(1 for a in apps for f in a.flows if f.old_slice == s.snssai)
        
        total_used_ul = s.current_load_bw_ul + s.reserved_bw + dynamic_used_ul
        total_used_dl = s.current_load_bw_dl + s.reserved_bw + dynamic_used_dl
        
        utilization_ul = (total_used_ul / s.total_bw_ul * 100) if s.total_bw_ul > 0 else 0.0
        utilization_dl = (total_used_dl / s.total_bw_dl * 100) if s.total_bw_dl > 0 else 0.0
        
        slice_status_list.append({
            "name": s.name,
            "snssai": s.snssai,
            "sst": s.sst,
            "usage_ul_pct": round(utilization_ul, 1),
            "usage_dl_pct": round(utilization_dl, 1),
            "active_flows": active_flows_count,
            "latency_sla": s.latency
        })

    # 简化的应用列表
    app_summary_list = []
    for a in apps:
        if flow_type_id is not None and any(f.service_type_id != flow_type_id for f in a.flows):
            continue
        app_summary_list.append({
            "app_id": a.app_id,
            "app_name": a.name,
            "flow_count": len(a.flows),
            "total_bw_mbps": round(a.total_bw_ul + a.total_bw_dl, 2)
        })
        
    return json.dumps({
        "slices": slice_status_list,
        "apps": app_summary_list
    }, ensure_ascii=False, indent=2)

# 兼容旧代码调用
get_network_status = get_network_status_full
