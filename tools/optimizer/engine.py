import pulp
import pandas as pd
from typing import List, Tuple, Optional, Dict

from .models import App, Slice, Node, Flow, OptimizationConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)

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
        
        # B_act_ul[app_id, flow_id, snssai]: 实际分配的上行带宽
        B_act_ul = pulp.LpVariable.dicts(
            "B_act_ul", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )
        
        # B_act_dl[app_id, flow_id, snssai]: 实际分配的下行带宽
        B_act_dl = pulp.LpVariable.dicts(
            "B_act_dl", 
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
        self._add_flow_constraints(prob, apps, slices, x, B_act_ul, B_act_dl, change)
        self._add_slice_constraints(prob, apps, slices, B_act_ul, B_act_dl, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act_ul, B_act_dl)

        # 3. 目标函数
        self._set_objective(prob, apps, slices, B_act_ul, B_act_dl, dev, change)

        # 4. 求解
        solver = pulp.PULP_CBC_CMD(msg=0)
        status = prob.solve(solver)
        
        status_str = pulp.LpStatus[status]
        logger.info(f"求解完成. 状态: {status_str}")

        if status != pulp.LpStatusOptimal:
            logger.warning("注意: 未找到最优解 (可能无解或仅找到可行解)")

        flow_results = self._format_results(apps, slices, x, B_act_ul, B_act_dl)
        slice_results = self._format_slice_stats(apps, slices, B_act_ul, B_act_dl)
        
        return flow_results, slice_results

    def _add_flow_constraints(self, prob, apps, slices, x, B_act_ul, B_act_dl, change):
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
                    
                    # C4: 上行带宽分配上限
                    prob += B_act_ul[app.app_id, f.flow_id, s.snssai] <= x[app.app_id, f.flow_id, s.snssai] * f.bw_ul
                    
                    # C5: 下行带宽分配上限
                    prob += B_act_dl[app.app_id, f.flow_id, s.snssai] <= x[app.app_id, f.flow_id, s.snssai] * f.bw_dl
                    
                    # C6: 线性化信令开销 |x - x_old|
                    is_old = 1 if f.old_slice == s.snssai else 0 
                    prob += change[app.app_id, f.flow_id, s.snssai] >= x[app.app_id, f.flow_id, s.snssai] - is_old
                    prob += change[app.app_id, f.flow_id, s.snssai] >= is_old - x[app.app_id, f.flow_id, s.snssai]

    def _add_slice_constraints(self, prob, apps, slices, B_act_ul, B_act_dl, dev):
        for s in slices:
            # C6: 切片容量约束 (假设总带宽为 ul + dl)
            real_available_bw = s.total_bw - s.reserved_bw - s.current_load_bw
            if real_available_bw < 0:
                real_available_bw = 0 
            
            # 统计所有已调度流的带宽之和 (ul + dl)
            total_allocated_in_slice = pulp.lpSum(
                B_act_ul[app.app_id, f.flow_id, s.snssai] + B_act_dl[app.app_id, f.flow_id, s.snssai]
                for app in apps for f in app.flows
            )
            prob += total_allocated_in_slice <= real_available_bw

            # C7: 负载均衡辅助约束
            final_load = total_allocated_in_slice + s.reserved_bw + s.current_load_bw
            load_ratio = final_load / s.total_bw if s.total_bw > 0 else 0
            
            prob += dev[s.snssai] >= load_ratio - self.config.rho
            prob += dev[s.snssai] >= self.config.rho - load_ratio

    def _add_node_constraints(self, prob, apps, slices, nodes, B_act_ul, B_act_dl):
        for node in nodes:
            # 计算托管切片上的流量总和
            hosted_snssais = [s.snssai for s in slices if s.name in node.slices_hosted]

            current_node_cpu = sum(s.current_load_bw * self.config.alpha for s in slices if s.snssai in hosted_snssais)
            current_node_mem = sum(s.current_load_bw * self.config.beta for s in slices if s.snssai in hosted_snssais)

            new_traffic_sum = pulp.lpSum(
                B_act_ul[app.app_id, f.flow_id, s.snssai] + B_act_dl[app.app_id, f.flow_id, s.snssai]
                for s in slices if s.snssai in hosted_snssais
                for app in apps for f in app.flows
            )
            
            # C8: 物理节点 CPU
            prob += current_node_cpu + new_traffic_sum * self.config.alpha <= node.cpu_capacity

            # C9: 物理节点 Memory
            prob += current_node_mem + new_traffic_sum * self.config.beta <= node.memory_capacity

    def _set_objective(self, prob, apps, slices, B_act_ul, B_act_dl, dev, change):
        term_load = pulp.lpSum(dev[s.snssai] for s in slices)
        
        # 信令开销 (Flow 级别的迁移动作)
        term_sig = pulp.lpSum(
            change[app.app_id, f.flow_id, s.snssai] 
            for app in apps for f in app.flows for s in slices
        )
        
        # 体验损失: Sum Weighted * ( (Req - Act) / Req ) for UL and DL
        term_exp = pulp.lpSum(
            (1.0 / f.priority if f.priority > 0 else 1.0) * (
                (f.bw_ul - pulp.lpSum(B_act_ul[app.app_id, f.flow_id, s.snssai] for s in slices)) / f.bw_ul if f.bw_ul > 0 else 0 +
                (f.bw_dl - pulp.lpSum(B_act_dl[app.app_id, f.flow_id, s.snssai] for s in slices)) / f.bw_dl if f.bw_dl > 0 else 0
            )
            for app in apps for f in app.flows
        )

        prob += self.config.w1 * term_load + self.config.w2 * term_sig + self.config.w3 * term_exp

    def _format_results(self, apps, slices, x, B_act_ul, B_act_dl) -> pd.DataFrame:
        results = []
        for app in apps:
            for f in app.flows:
                mapped_slice = None
                allocated_bw_ul = 0.0
                allocated_bw_dl = 0.0
                
                for s in slices:
                    if pulp.value(x[app.app_id, f.flow_id, s.snssai]) == 1:
                        mapped_slice = s.snssai # 记录 SNSSAI
                        allocated_bw_ul = float(pulp.value(B_act_ul[app.app_id, f.flow_id, s.snssai]))
                        allocated_bw_dl = float(pulp.value(B_act_dl[app.app_id, f.flow_id, s.snssai]))
                        break
                
                strategies = self._determine_strategy(f, mapped_slice, allocated_bw_ul, allocated_bw_dl)

                results.append({
                    "App": app.name,
                    "App ID": app.app_id,
                    "Flow ID": f.flow_id,
                    "Flow Name": f.name,
                    "Old Slice": f.old_slice,
                    "New Slice": mapped_slice,
                    "Req BW UL": f.bw_ul,
                    "Req BW DL": f.bw_dl,
                    "Act BW UL": round(allocated_bw_ul, 2),
                    "Act BW DL": round(allocated_bw_dl, 2),
                    "Strategies": ", ".join(strategies)
                })
        
        return pd.DataFrame(results)

    def _format_slice_stats(self, apps, slices, B_act_ul, B_act_dl) -> pd.DataFrame:
        slice_stats = []
        for s in slices:
            allocated_to_flows_ul = sum(
                pulp.value(B_act_ul[app.app_id, f.flow_id, s.snssai]) 
                for app in apps for f in app.flows
            )
            allocated_to_flows_dl = sum(
                pulp.value(B_act_dl[app.app_id, f.flow_id, s.snssai]) 
                for app in apps for f in app.flows
            )
            allocated_to_flows = allocated_to_flows_ul + allocated_to_flows_dl
            
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
                "Allocated UL (M)": round(allocated_to_flows_ul, 2),
                "Allocated DL (M)": round(allocated_to_flows_dl, 2),
                "Allocated (M)": round(allocated_to_flows, 2),
                "Total Load (M)": round(total_load, 2),
                "Load Ratio (%)": round(load_ratio, 1),
                "Remaining (M)": round(remaining, 2)
            })
        return pd.DataFrame(slice_stats)

    def _determine_strategy(self, flow: Flow, mapped_slice: Optional[str], allocated_bw_ul: float, allocated_bw_dl: float) -> List[str]:
        strategies = []
        if mapped_slice != flow.old_slice:
            strategies.append("策略B(重路由)")
        
        TOLERANCE = 0.01
        
        # 策略A: 拒绝/被抢占 (带宽归零或接近零)
        if allocated_bw_ul < TOLERANCE and flow.bw_ul > TOLERANCE:
            strategies.append("策略A(拒绝/被抢占 UL)")
        if allocated_bw_dl < TOLERANCE and flow.bw_dl > TOLERANCE:
            strategies.append("策略A(拒绝/被抢占 DL)")

        # 策略C: 修改
        if flow.old_allocated_bw_ul is not None and flow.old_allocated_bw_dl is not None:
            if abs(allocated_bw_ul - flow.old_allocated_bw_ul) > TOLERANCE or abs(allocated_bw_dl - flow.old_allocated_bw_dl) > TOLERANCE:
                strategies.append("策略C(修改)")

        # 如果带宽未被满足 (降级)
        if allocated_bw_ul < flow.bw_ul - TOLERANCE and allocated_bw_ul > TOLERANCE:
            strategies.append("策略D(降级 UL)")
        if allocated_bw_dl < flow.bw_dl - TOLERANCE and allocated_bw_dl > TOLERANCE:
            strategies.append("策略D(降级 DL)")
        
        if not strategies:
            strategies.append("保持")
        return strategies
