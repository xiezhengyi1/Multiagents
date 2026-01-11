import pulp
import pandas as pd
import numpy as np
import sys
import os
import uuid
try:
    from utils.logger import setup_logger
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import setup_logger

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# --- 配置与日志 ---
logger = setup_logger(__name__)

# --- 数据模型 ---

@dataclass
class Flow:
    """定义应用内的单个数据流需求"""
    name: str # 保留作为描述性名称
    bw: float       # 带宽 (Mbps)
    lat: float      # 时延 (ms)
    priority: int   # 优先级 (数值越小越高)
    flow_id: str = "f_default" # 流唯一标识 (主要Key)
    old_slice: Optional[str] = None # 流的原切片名称 (实为 S-NSSAI)

@dataclass
class App:
    """定义应用及其聚合需求"""
    name: str # 保留作为描述性名称
    flows: List[Flow]
    weight: float   # 业务权重 (Vi)
    app_id: str = "app_default" # 应用唯一标识 (主要Key)
    old_slice: Optional[str] = None # App 维度的原切片
    
    # 聚合属性 (自动计算)
    total_bw: float = field(init=False)
    min_lat: float = field(init=False)
    max_prio: int = field(init=False)

    def __post_init__(self):
        if not self.flows:
            self.total_bw = 0.0
            self.min_lat = float('inf')
            self.max_prio = 0
        else:
            self.total_bw = sum(f.bw for f in self.flows)
            self.min_lat = min(f.lat for f in self.flows)
            self.max_prio = max(f.priority for f in self.flows)

@dataclass
class Slice:
    """定义网络切片资源与状态"""
    name: str # 描述性名称
    sst: int        # 切片服务类型
    sd: str         # 切片微分器
    snssai: str = field(init=False) # 唯一标识 (SST-SD), 自动生成
    total_bw: float # 总带宽容量
    current_load_bw: float # 当前基础负载
    latency: float  # 链路传输时延
    proc_delay: float # 处理时延
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

@dataclass
class OptimizationConfig:
    """优化算法参数配置"""
    rho: float = 0.8   # 目标负载率
    w1: float = 100.0  # 负载均衡权重
    w2: float = 50.0   # 信令开销权重
    w3: float = 1000.0 # 体验损失权重
    alpha: float = 0.1 # 带宽转CPU消耗系数
    beta: float = 0.05 # 带宽转内存消耗系数

# --- 核心逻辑 ---

class SliceOptimizationEngine:
    """切片资源分配优化引擎 - 支持单流粒度映射"""
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config

    def solve(self, apps: List[App], slices: List[Slice], nodes: List[Node]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建并求解优化问题
        返回: (应用分配结果DataFrame, 切片负载状态DataFrame)
        """
        logger.info(f"开始优化: Apps={len(apps)}, Slices={len(slices)}, Nodes={len(nodes)}")
        
        prob = pulp.LpProblem("5G_Slice_Resource_Allocation_FlowLevel", pulp.LpMinimize)

        # 1. 变量定义
        # x_af_s[app_id, flow_id, snssai]: App的Flow是否映射到Slice
        x = pulp.LpVariable.dicts(
            "x", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            cat='Binary'
        )
        
        # B_act[app_id, flow_id, snssai]: 实际分配的带宽
        B_act = pulp.LpVariable.dicts(
            "B_act", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )

        # 辅助变量
        dev = pulp.LpVariable.dicts("dev", (s.snssai for s in slices), lowBound=0)
        change = pulp.LpVariable.dicts(
            "change", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )

        # 2. 约束条件构建
        self._add_flow_constraints(prob, apps, slices, x, B_act, change)
        self._add_slice_constraints(prob, apps, slices, B_act, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act)

        # 3. 目标函数
        self._set_objective(prob, apps, slices, B_act, dev, change)

        # 4. 求解
        solver = pulp.PULP_CBC_CMD(msg=0)
        status = prob.solve(solver)
        
        status_str = pulp.LpStatus[status]
        logger.info(f"求解完成. 状态: {status_str}")

        if status != pulp.LpStatusOptimal:
            logger.warning("注意: 未找到最优解 (可能无解或仅找到可行解)")

        flow_results = self._format_results(apps, slices, x, B_act)
        slice_results = self._format_slice_stats(apps, slices, B_act)
        
        return flow_results, slice_results

    def _add_flow_constraints(self, prob, apps, slices, x, B_act, change):
        """添加流级别的约束"""
        for app in apps:
            for f in app.flows:
                # C1: 每个 Flow 必须选择且仅选择一个切片
                prob += pulp.lpSum(x[app.app_id, f.flow_id, s.snssai] for s in slices) == 1
                
                for s in slices:
                    # C2: [已放宽] URSP 类型兼容 -> 隐式控制

                    # C3: 时延约束
                    total_latency = s.latency + s.proc_delay
                    if total_latency > f.lat:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 0
                    
                    # C4: 带宽分配上限
                    prob += B_act[app.app_id, f.flow_id, s.snssai] <= x[app.app_id, f.flow_id, s.snssai] * f.bw
                    
                    # C5: 线性化信令开销 |x - x_old|
                    is_old = 1 if f.old_slice == s.snssai else 0 
                    prob += change[app.app_id, f.flow_id, s.snssai] >= x[app.app_id, f.flow_id, s.snssai] - is_old
                    prob += change[app.app_id, f.flow_id, s.snssai] >= is_old - x[app.app_id, f.flow_id, s.snssai]

    def _add_slice_constraints(self, prob, apps, slices, B_act, dev):
        for s in slices:
            # C6: 切片容量约束
            real_available_bw = s.total_bw - s.reserved_bw - s.current_load_bw
            if real_available_bw < 0:
                real_available_bw = 0 
            
            # 统计所有已调度流的带宽之和
            total_allocated_in_slice = pulp.lpSum(
                B_act[app.app_id, f.flow_id, s.snssai] 
                for app in apps for f in app.flows
            )
            prob += total_allocated_in_slice <= real_available_bw

            # C7: 负载均衡辅助约束
            final_load = total_allocated_in_slice + s.reserved_bw + s.current_load_bw
            load_ratio = final_load / s.total_bw if s.total_bw > 0 else 0
            
            prob += dev[s.snssai] >= load_ratio - self.config.rho
            prob += dev[s.snssai] >= self.config.rho - load_ratio

    def _add_node_constraints(self, prob, apps, slices, nodes, B_act):
        for node in nodes:
            # 计算托管切片上的流量总和
            # 注意: node.slices_hosted 中存储的可能是 slice.name，这里假设已统一为 SNSSAI 
            # 或者我们需要一个映射。为了简化，假设 User 知道 slices_hosted 里填的是 snssai。
            # 但实际上初始化数据填的是 name。这里做一个简单的 name->snssai 查找比较好，
            # 或者直接修改 nodes_data 的 slices_hosted 为 snssai。
            # 鉴于目前架构，通过 name 查找 snssai 最稳妥。
            
            hosted_snssais = [s.snssai for s in slices if s.name in node.slices_hosted]

            current_node_cpu = sum(s.current_load_bw * self.config.alpha for s in slices if s.snssai in hosted_snssais)
            current_node_mem = sum(s.current_load_bw * self.config.beta for s in slices if s.snssai in hosted_snssais)

            new_traffic_sum = pulp.lpSum(
                B_act[app.app_id, f.flow_id, s.snssai]
                for s in slices if s.snssai in hosted_snssais
                for app in apps for f in app.flows
            )
            
            # C8: 物理节点 CPU
            prob += current_node_cpu + new_traffic_sum * self.config.alpha <= node.cpu_capacity

            # C9: 物理节点 Memory
            prob += current_node_mem + new_traffic_sum * self.config.beta <= node.memory_capacity

    def _set_objective(self, prob, apps, slices, B_act, dev, change):
        term_load = pulp.lpSum(dev[s.snssai] for s in slices)
        
        # 信令开销 (Flow 级别的迁移动作)
        term_sig = pulp.lpSum(
            change[app.app_id, f.flow_id, s.snssai] 
            for app in apps for f in app.flows for s in slices
        )
        
        # 体验损失: Sum Weighted * ( (Req - Act) / Req )
        term_exp = pulp.lpSum(
            (1.0 / f.priority if f.priority > 0 else 1.0) * (f.bw - pulp.lpSum(B_act[app.app_id, f.flow_id, s.snssai] for s in slices)) / f.bw
            for app in apps for f in app.flows if f.bw > 0
        )

        prob += self.config.w1 * term_load + self.config.w2 * term_sig + self.config.w3 * term_exp

    def _format_results(self, apps, slices, x, B_act) -> pd.DataFrame:
        results = []
        for app in apps:
            for f in app.flows:
                mapped_slice = None
                allocated_bw = 0.0
                
                for s in slices:
                    if pulp.value(x[app.app_id, f.flow_id, s.snssai]) == 1:
                        mapped_slice = s.snssai # 记录 SNSSAI
                        allocated_bw = float(pulp.value(B_act[app.app_id, f.flow_id, s.snssai]))
                        break
                
                strategies = self._determine_strategy(f, mapped_slice, allocated_bw)

                results.append({
                    "App": app.name,
                    "App ID": app.app_id,
                    "Flow ID": f.flow_id,
                    "Flow Name": f.name,
                    "Old Slice": f.old_slice,
                    "New Slice": mapped_slice,
                    "Req BW": f.bw,
                    "Act BW": round(allocated_bw, 2),
                    "Strategies": ", ".join(strategies)
                })
        
        return pd.DataFrame(results)

    def _format_slice_stats(self, apps, slices, B_act) -> pd.DataFrame:
        slice_stats = []
        for s in slices:
            allocated_to_flows = sum(
                pulp.value(B_act[app.app_id, f.flow_id, s.snssai]) 
                for app in apps for f in app.flows
            )
            
            # 总负载 = 动态分配 + 保留 + 静态负载
            total_load = allocated_to_flows + s.reserved_bw + s.current_load_bw
            
            load_ratio = (total_load / s.total_bw) * 100 if s.total_bw > 0 else 0
            
            # 剩余也应当扣除所有占用
            remaining = s.total_bw - total_load

            slice_stats.append({
                "Slice": s.name,
                "SNSSAI": s.snssai,
                "Total Cap (M)": s.total_bw,
                "Reserved (M)": s.reserved_bw, # 仅展示
                "Static Load (M)": s.current_load_bw, # 新增展示
                "Allocated (M)": round(allocated_to_flows, 2),
                "Total Load (M)": round(total_load, 2),
                "Load Ratio (%)": round(load_ratio, 1),
                "Remaining (M)": round(remaining, 2)
            })
        return pd.DataFrame(slice_stats)

    def _determine_strategy(self, flow: Flow, mapped_slice: Optional[str], allocated_bw: float) -> List[str]:
        strategies = []
        if mapped_slice != flow.old_slice:
            strategies.append("策略B(重路由)")
        
        TOLERANCE = 0.01
        if allocated_bw < flow.bw - TOLERANCE:
            if allocated_bw < TOLERANCE:
                strategies.append("策略A(拒绝/被抢占)")
            else:
                strategies.append("策略C(降级)")
        
        if not strategies:
            strategies.append("保持")
        return strategies

# --- 场景管理 ---

def get_initial_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """初始化模拟场景数据"""
    
    # 辅助函数：快速构造 App 并将 old_slice 传递给 flows (简单初始化逻辑)
    def create_app(name, app_id, flows, weight, old_slice_snssai):
        # 确保每个 Flow 都有 ID (如果未指定)
        for i, f in enumerate(flows):
            if f.flow_id == "f_default":
                 f.flow_id = f"{app_id}_f{i+1}_{f.name}"
            f.old_slice = old_slice_snssai # 初始状态假定都在旧切片 (使用 SNSSAI)
            
        return App(name=name, app_id=app_id, flows=flows, weight=weight, old_slice=old_slice_snssai)

    apps_data = [
        # S1_Gold (SST=2, SD=000001) -> SNSSAI="02000001"
        create_app("Remote_Drive", "app_remote_drive", [
            Flow("Control", 2, 5, 20),
            Flow("Video_Feed", 8, 20, 15)
        ], weight=1000, old_slice_snssai="02000001"),
        
        # S2_Silver (SST=1, SD=000001) -> SNSSAI="01000001"
        create_app("4K_Video", "app_4k_video", [
            Flow("Main_Stream", 35, 50, 10),
            Flow("Audio", 5, 100, 5)
        ], weight=50, old_slice_snssai="01000001"),
        
        # S1_Gold -> "02000001"
        create_app("IoT_Sensor", "app_iot_sensor", [
            Flow("Telemetry", 2, 20, 10)
        ], weight=100, old_slice_snssai="02000001"),
        
        # S3_Public (SST=1, SD=000002) -> SNSSAI="01000002"
        create_app("Web_Browse", "app_web_browse", [
            Flow("HTTP", 15, 100, 1, 'f_http')
        ], weight=10, old_slice_snssai="01000002"),
        
        # S2_Silver -> "01000001"
        create_app("AR_Gaming", "app_ar_gaming", [
            Flow("Render", 20, 20, 15),
            Flow("Sync", 5, 15, 15)
        ], weight=200, old_slice_snssai="01000001"),
        
        # S1_Gold -> "02000001"
        create_app("Factory_Robot", "app_factory_robot", [
            Flow("Motion_Cmd", 5, 5, 100)
        ], weight=2000, old_slice_snssai="02000001"),
        
        # S3_Public -> "01000002"
        create_app("Smart_Meter", "app_smart_meter", [
            Flow("Data_Report", 0.5, 200, 1)
        ], weight=20, old_slice_snssai="01000002")
    ]

    slices_data = [
        # SST: 1=eMBB, 2=URLLC, 3=MIoT
        Slice("S1_Gold", sst=2, sd="000001", total_bw=100, current_load_bw=0, latency=3, proc_delay=1, reserved_bw=20),
        Slice("S2_Silver", sst=1, sd="000001", total_bw=200, current_load_bw=0, latency=10, proc_delay=2, reserved_bw=50),
        Slice("S3_Public", sst=1, sd="000002", total_bw=150, current_load_bw=0, latency=40, proc_delay=5, reserved_bw=10),
        Slice("S4_Platinum", sst=2, sd="000002", total_bw=50, current_load_bw=0, latency=1, proc_delay=0.5, reserved_bw=5),
        Slice("S5_Massive", sst=3, sd="000001", total_bw=30, current_load_bw=0, latency=100, proc_delay=10, reserved_bw=2)
    ]
    
    nodes_data = [
        Node("Node_Edge", cpu_capacity=100, memory_capacity=200, slices_hosted=["S1_Gold", "S2_Silver", "S4_Platinum"]),
        Node("Node_Core", cpu_capacity=300, memory_capacity=1000, slices_hosted=["S3_Public", "S5_Massive"])
    ]
    
    return apps_data, slices_data, nodes_data

def decide_strategy_for_new_flow(new_app: App, current_apps: List[App], slices: List[Slice], nodes: List[Node]):
    """处理新业务请求 (Debug Helper)"""
    print(f"\n>>> 收到新业务请求: {new_app.name}")
    updated_apps = current_apps + [new_app]
    engine = SliceOptimizationEngine()
    results_df, slice_stats_df = engine.solve(updated_apps, slices, nodes)
    
    if results_df.empty:
        logger.error("优化未返回结果")
        return

    # 这里我们打印该 App 下所有 Flow 的结果
    my_result = results_df[results_df['App'] == new_app.name]
    print(f"\n--- <{new_app.name}> 决策结果 ---")
    print(my_result.to_string(index=False))
    
    print("\n--- 切片负载状态 ---")
    print(slice_stats_df.to_string(index=False))
    return my_result

def main():
    # 1. 初始化
    apps, slices, nodes = get_initial_scenario()
    engine = SliceOptimizationEngine()
    
    # 2. 初始场景优化
    print("--- 初始场景优化 ---")
    df_initial, slice_stats_initial = engine.solve(apps, slices, nodes)
    print(df_initial.to_string())
    print("\n--- 初始切片负载 ---")
    print(slice_stats_initial.to_string(index=False))
    
    # 3. 模拟新业务接入
    new_flow_app = App(
        name="Emergency_Call", 
        flows=[Flow("Voice", 100, 5, 20), Flow("Video", 50, 20, 10)], 
        weight=5000, 
        old_slice=None
    )
    
    decide_strategy_for_new_flow(new_flow_app, apps, slices, nodes)


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

def optimize_network_slices(new_app_data: dict, w1: float, w2: float, w3: float) -> str:
    """
    执行网络切片资源优化求解 (粒度: Flow)
    """
    try:
        # A. 准备环境
        if _GLOBAL_SCENARIO_CONTEXT["apps"] is not None:
             apps = _GLOBAL_SCENARIO_CONTEXT["apps"]
             slices = _GLOBAL_SCENARIO_CONTEXT["slices"]
             nodes = _GLOBAL_SCENARIO_CONTEXT["nodes"]
        else:
             apps, slices, nodes = get_initial_scenario()
        
        # B. 构造新应用对象
        app_name = new_app_data.get('app_name', 'NewApp')
        # 生成唯一 ID 防止冲突
        app_uuid = str(uuid.uuid4())[:6]
        new_app_id = f"app_{app_name}_{app_uuid}"

        flows = []
        for i, f in enumerate(new_app_data.get('flows', [])):
            # 兼容处理带宽字段 (IntentAgent V1 输出 bandwidth_demand, V2 输出 bandwidth_demand_dl/ul)
            bw_val = f.get('bandwidth_demand', 0)
            if bw_val == 0:
                bw_val = f.get('bandwidth_demand_dl', 0) + f.get('bandwidth_demand_ul', 0)

            # 构造唯一 Flow ID
            original_fid = f.get('flow_id', '')
            if original_fid:
                f_id = f"{new_app_id}_{original_fid}"
            else:
                f_name = f.get('description', 'flow').replace(" ", "_")
                f_id = f"{new_app_id}_f{i}_{f_name}"

            flows.append(Flow(
                name=f.get('description', f.get('flow_id', 'DefaultFlow')), # 使用描述或ID作为内部名称
                bw=bw_val, 
                lat=f.get('latency_requirement', 100),
                priority=f.get('priority_level', 10),
                flow_id=f_id,
                old_slice=None
            ))
            
        new_app = App(
            name=app_name,
            app_id=new_app_id,
            flows=flows,
            weight=1000, 
            old_slice=None
        )
        
        # C. 更新配置
        config = OptimizationConfig(w1=w1, w2=w2, w3=w3)
        engine = SliceOptimizationEngine(config)
        
        # D. 求解
        updated_apps = apps + [new_app]
        results_df, slice_stats_df = engine.solve(updated_apps, slices, nodes)
        
        if results_df.empty:
            return "求解器未返回结果。"
            

        # --- Side Effect: 更新全局状态 ---
        if _GLOBAL_SCENARIO_CONTEXT["apps"] is not None:
            # 1. 移除旧的同名应用 (注意: 这里依然按Name移除旧的, 这是一个业务逻辑选择, 覆盖同名App)
            _GLOBAL_SCENARIO_CONTEXT["apps"][:] = [a for a in _GLOBAL_SCENARIO_CONTEXT["apps"] if a.name != new_app.name]
            # 2. 加入新应用
            _GLOBAL_SCENARIO_CONTEXT["apps"].append(new_app)
            
            # 3. 结果写回对象 (Flow级更新)
            # 遍历结果DF，找到对应App和Flow，更新其 old_slice
            for index, res_row in results_df.iterrows():
                r_app_id = res_row['App ID']
                r_flow_id = res_row['Flow ID']
                r_new_slice = res_row['New Slice']
                
                # 在全局 App 列表中找到该 App (使用ID匹配更准确)
                target_app = next((a for a in _GLOBAL_SCENARIO_CONTEXT["apps"] if a.app_id == r_app_id), None)
                if target_app:
                    # 更新 App 的 old_slice 为主 Slice (仅做展示用)
                    target_app.old_slice = r_new_slice 
                    
                    # 关键: 更新 Flow 对象的 old_slice
                    target_flow = next((f for f in target_app.flows if f.flow_id == r_flow_id), None)
                    if target_flow:
                        target_flow.old_slice = r_new_slice

        # E. 格式化输出
        output = []
        output.append(f"--- 优化求解报告 (Flow Level) ---")
        output.append(f"使用的权重: w1={w1}, w2={w2}, w3={w3}")
        
        # 1. 新业务结果 (用ID匹配)
        my_result = results_df[results_df['App ID'] == new_app.app_id]
        if not my_result.empty:
            output.append(f"新业务 '{new_app.name}' 分配结果:")
            for idx, row in my_result.iterrows():
                output.append(
                    f"  - 流 [{row['Flow Name']}] (ID:{row['Flow ID']}) -> 切片: {row['New Slice'] if row['New Slice'] else '无'} "
                    f"(BW: {row['Act BW']}/{row['Req BW']}M, {row['Strategies']})"
                )
        else:
            output.append("错误: 结果中未找到新业务。")

        # 2. 受影响的旧业务
        other_results = results_df[results_df['App ID'] != new_app.app_id]
        # 筛选出策略不是 "保持" 的流 (即发生了重路由、降级或抢占)
        impacted_results = other_results[other_results['Strategies'].str.contains("策略", na=False)] 
        # 注意: 之前的 Strategies 格式是 "保持", "策略B(重路由)", "策略B(重路由), 策略C(降级)"
        # 只要不全是 "保持" 就算受影响。
        # 简单的判断方法: Strategies != "保持"

        if not impacted_results.empty:
            output.append("\n受影响的现有业务流:")
            for idx, row in impacted_results.iterrows():
                if row['Strategies'] == "保持": continue
                output.append(
                    f"  - {row['App']} / [{row['Flow Name']}] (ID:{row['Flow ID']}) -> {row['New Slice'] if row['New Slice'] else '无'} "
                    f"(BW: {row['Act BW']}/{row['Req BW']}M, {row['Strategies']})"
                )

        output.append("\n切片负载摘要:")
        # 简化输出
        for _, row in slice_stats_df.iterrows():
            output.append(f"  - {row['Slice']} ({row['SNSSAI']}): Load {row['Load Ratio (%)']}% (Rem: {row['Remaining (M)']}M)")
            
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"优化过程发生异常: {e}")
        return f"优化求解失败: {str(e)}"

if __name__ == "__main__":
    main()
