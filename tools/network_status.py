import pandas as pd
from tools.optimization import _GLOBAL_SCENARIO_CONTEXT, get_initial_scenario, set_global_scenario

def get_network_status():
    """
    获取当前网络切片和节点的状态摘要。
    返回包含切片资源使用情况、节点状态的文本报告。
    """
    # 1. 获取全局上下文
    if _GLOBAL_SCENARIO_CONTEXT["apps"] is None:
        apps, slices, nodes = get_initial_scenario()
        set_global_scenario(apps, slices, nodes)
    else:
        apps = _GLOBAL_SCENARIO_CONTEXT["apps"]
        slices = _GLOBAL_SCENARIO_CONTEXT["slices"]
        nodes = _GLOBAL_SCENARIO_CONTEXT["nodes"]

    # 2. 统计切片负载
    # 切片本身可能有 base load (current_load_bw) 和 reserved_bw
    # 还需要加上当前分配到该切片的所有 App Flow 的带宽
    
    slice_status_list = []
    
    for s in slices:
        # 基础占用
        base_used = s.current_load_bw + s.reserved_bw
        
        # 动态业务占用
        # 遍历所有 App 的所有 Flow，检查其 old_slice 是否指向当前切片 s.snssai
        # 注意: 在 optimization.py 中，flow.old_slice 被用来存储当前实际所在的切片
        dynamic_used = 0.0
        active_flows_count = 0
        
        for app in apps:
            for flow in app.flows:
                if flow.old_slice == s.snssai:
                    dynamic_used += flow.bw
                    active_flows_count += 1
        
        total_used = base_used + dynamic_used
        utilization = (total_used / s.total_bw * 100) if s.total_bw > 0 else 0.0
        remaining = s.total_bw - total_used
        
        slice_status_list.append({
            "Slice Name": s.name,
            "S-NSSAI": s.snssai,
            "Total BW (Mbps)": s.total_bw,
            "Used BW (Mbps)": round(total_used, 2),
            "Usage (%)": round(utilization, 1),
            "Remaining (Mbps)": round(remaining, 2),
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
        
    # 4. 格式化输出
    df_slice = pd.DataFrame(slice_status_list)
    df_node = pd.DataFrame(node_status_list)
    
    report = []
    report.append("### 当前网络切片状态")
    try:
        report.append(df_slice.to_markdown(index=False, numalign="left", stralign="left"))
    except ImportError:
         report.append(df_slice.to_string(index=False))
    
    report.append("\n### 物理基础设施状态")
    try:
        report.append(df_node.to_markdown(index=False, numalign="left", stralign="left"))
    except ImportError:
        report.append(df_node.to_string(index=False))
    
    report.append(f"\n当前接入应用总数: {len(apps)}")
    
    return "\n".join(report)

if __name__ == "__main__":
    print(get_network_status())
