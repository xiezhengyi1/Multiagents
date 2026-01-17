from typing import List, Tuple, Dict, Optional
from .models import App, Flow, Slice, Node

# Global context
_GLOBAL_SCENARIO_CONTEXT = {
    "apps": None,
    "slices": None,
    "nodes": None
}

def set_global_scenario(apps: List[App], slices: List[Slice], nodes: List[Node]):
    _GLOBAL_SCENARIO_CONTEXT["apps"] = apps
    _GLOBAL_SCENARIO_CONTEXT["slices"] = slices
    _GLOBAL_SCENARIO_CONTEXT["nodes"] = nodes
    
def get_initial_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """初始化模拟场景数据"""
    
    # 辅助函数：快速构造 App 并将 old_slice 传递给 flows (简单初始化逻辑)
    def create_app(name, app_id, flows):
        # 确保每个 Flow 都有 ID
        for i, f in enumerate(flows):
            if f.flow_id == "f_default":
                 f.flow_id = f"{app_id}_f{i+1}_{f.name}"
            else:
                 f.flow_id = f"{app_id}_{f.flow_id}"
            
            # 如果没有显式设置 GBR (在 Flow 初始化时未传入会导致错误，但这里假设已经修改了 Flow 的调用)
            # 不过实际上 Flow 是 dataclass，必须传入所有非默认参数。
            # 所以这里不需要额外处理，直接在下面修改 Flow 的实例化即可。
            
            f.old_allocated_bw_ul = f.bw_ul
            f.old_allocated_bw_dl = f.bw_dl
            
        return App(name=name, app_id=app_id, flows=flows)

    apps_data = [
        # S1_Gold (SST=2, SD=000001) -> SNSSAI="02000001"
        create_app("Remote_Drive", "app_remote_drive", [
            Flow("Control", "f_control", 2, 2, 0.5, 0.5, 5, 0.001, 1, 20, old_slice="02000001"),
            Flow("Video_Feed", "f_video_feed", 8, 8, 4, 4, 20, 0.01, 5, 15, old_slice="02000001")
        ]),
        
        # S2_Silver (SST=1, SD=000001) -> SNSSAI="01000001"
        create_app("4K_Video", "app_4k_video", [
            Flow("Main_Stream", "f_main_stream", 35, 30, 10, 10, 50, 0.02, 10, 10, old_slice="01000001"),
            Flow("Audio", "f_audio", 5, 5, 1, 1, 100, 0.01, 20, 5, old_slice="01000001")
        ]),
        
        # S1_Gold -> "02000001"
        create_app("IoT_Sensor", "app_iot_sensor", [
            Flow("Telemetry", "f_telemetry", 2, 2, 0.1, 0.1, 20, 0.005, 5, 10, old_slice="02000001")
        ]),
        
        # S3_Public (SST=1, SD=000002) -> SNSSAI="01000002"
        create_app("Web_Browse", "app_web_browse", [
            Flow("HTTP", "f_http", 15, 20, 1, 1, 100, 0.03, 30, 1, old_slice="01000002")
        ]),
        
        # S2_Silver -> "01000001"
        create_app("AR_Gaming", "app_ar_gaming", [
            Flow("Render", "f_render", 20, 15, 5, 5, 20, 0.01, 5, 15, old_slice="01000001"),
            Flow("Sync", "f_sync", 5, 6, 2, 2, 15, 0.005, 3, 15, old_slice="01000001")
        ]),
        
        # S1_Gold -> "02000001"
        create_app("Factory_Robot", "app_factory_robot", [
            Flow("Motion_Cmd", "f_motion_cmd", 5, 5, 2.5, 2.5, 5, 0.0001, 1, 100, old_slice="02000001")
        ]),
        
        # S3_Public -> "01000002"
        create_app("Smart_Meter", "app_smart_meter", [
            Flow("Data_Report", "f_data_report", 0.5, 0.5, 0.05, 0.05, 200, 0.05, 50, 1, old_slice="01000002")
        ])
    ]

    slices_data = [
        # SST: 1=eMBB, 2=URLLC, 3=MIoT
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
