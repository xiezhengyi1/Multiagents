import pandas as pd
import json
from langchain_core.tools import tool
from tools.db_tool import get_current_scenario, _serialize_scenario_for_db
from database.models import NetworkStatusSnapshot
from database.connection import SessionLocal

@tool
def save_network_status_snapshot(trigger_event: str = "Manual") -> str:
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
                snapshot_data=data,
                trigger_event=str(trigger_event)
            )
            session.add(snapshot)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
        return "Success: Network status snapshot saved."
    except Exception as e:
        return f"Error saving snapshot: {str(e)}"

def get_network_status():
    """
    获取当前网络切片和节点的状态摘要。
    返回包含切片资源使用情况、节点状态的文本报告。
    """
    # 1. 获取场景数据 (优先缓存，否则DB/默认)
    apps, slices, nodes = get_current_scenario()

    # 2. 统计切片负载
    # 切片本身可能有 base load (current_load_bw) 和 reserved_bw
    # 还需要加上当前分配到该切片的所有 App Flow 的带宽
    
    slice_status_list = []
    
    for s in slices:
        # 基础占用 (上下行)
        base_used_ul = s.current_load_bw_ul + s.reserved_bw
        base_used_dl = s.current_load_bw_dl + s.reserved_bw
        
        # 动态业务占用
        # 遍历所有 App 的所有 Flow，检查其 old_slice 是否指向当前切片 s.snssai
        dynamic_used_ul = 0.0
        dynamic_used_dl = 0.0
        active_flows_count = 0
        
        for app in apps:
            for flow in app.flows:
                if flow.old_slice == s.snssai:
                    dynamic_used_ul += flow.bw_ul
                    dynamic_used_dl += flow.bw_dl
                    active_flows_count += 1
        
        total_used_ul = base_used_ul + dynamic_used_ul
        total_used_dl = base_used_dl + dynamic_used_dl
        
        utilization_ul = (total_used_ul / s.total_bw_ul * 100) if s.total_bw_ul > 0 else 0.0
        utilization_dl = (total_used_dl / s.total_bw_dl * 100) if s.total_bw_dl > 0 else 0.0
        
        remaining_ul = s.total_bw_ul - total_used_ul
        remaining_dl = s.total_bw_dl - total_used_dl
        
        slice_status_list.append({
            "Slice Name": s.name,
            "S-NSSAI": s.snssai,
            "UL Total (Mbps)": s.total_bw_ul,
            "UL Used (Mbps)": round(total_used_ul, 2),
            "UL Usage (%)": round(utilization_ul, 1),
            "DL Total (Mbps)": s.total_bw_dl,
            "DL Used (Mbps)": round(total_used_dl, 2),
            "DL Usage (%)": round(utilization_dl, 1),
            "Latency (ms)": s.latency,
            "Active Flows": active_flows_count
        })

    # 3. 统计节点状态 (简单展示)
    node_status_list = []
    for n in nodes:
        node_status_list.append({
            "Node Name": n.name,
            "Hosted Slices": ", ".join(n.slices_hosted),
            "CPU Cap": n.cpu_capacity,
            "Mem Cap": n.memory_capacity
        })
        
    # 4. 返回JSON格式
    return json.dumps({
        "slice_status": slice_status_list,
        "node_status": node_status_list,
        "total_apps": len(apps)
    }, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print(get_network_status())
