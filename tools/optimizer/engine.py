import pulp
import pandas as pd
from typing import List, Tuple, Optional, Dict

from ..models import App, Slice, Node, Flow, OptimizationConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SliceOptimizationEngine:
    """切片资源分配优化引擎 - 支持单流粒度映射"""
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config

    def solve(self, apps: List[App], slices: List[Slice], nodes: List[Node]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[float], Dict[str, float]]:
        """
        构建并求解优化问题 (三权重单目标)
        目标函数:
            w1 * 负载均衡 + w2 * 信令开销 + w3 * 体验损失(含SLA违约惩罚)
        说明: 为保证解的唯一性，加入极小扰动项作为平局打破。
        """
        logger.info(f"开始优化(三权重): Apps={len(apps)}, Slices={len(slices)}, Nodes={len(nodes)}")

        prob = pulp.LpProblem("5G_Slice_R_A_Layered", pulp.LpMinimize)

        # 1. 变量定义
        x = pulp.LpVariable.dicts(
            "x", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            cat='Binary'
        )
        
        B_act_ul = pulp.LpVariable.dicts(
            "B_act_ul", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )
        
        B_act_dl = pulp.LpVariable.dicts(
            "B_act_dl", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )

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

        # 3. 构建目标函数项
        term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak = self._build_objectives(
            apps, slices, x, B_act_ul, B_act_dl, dev, change
        )

        # 4. 单目标求解 (三权重)
        objective = (self.config.w1 * term_load) + (self.config.w2 * term_sig) + (self.config.w3 * (term_exp + term_qos)) + term_tiebreak
        prob.setObjective(objective)
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        status_str = pulp.LpStatus[prob.status]
        logger.info(f"求解完成. 状态: {status_str}")

        flow_results, slice_results, _, objective_val = self._finalize_results(prob, apps, slices, x, B_act_ul, B_act_dl)
        breakdown = self._build_objective_breakdown(
            term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak, objective_val
        )
        return flow_results, slice_results, status_str, objective_val, breakdown

    def solve_incremental(self, apps: List[App], slices: List[Slice], nodes: List[Node]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[float], Dict[str, float]]:
        """
        增量优化：固定已有流的原切片映射，仅优化新增/无历史的流。
        若旧分配带宽可用，则保持；否则允许在同一切片内微调以保证可行。
        """
        logger.info(f"开始增量优化: Apps={len(apps)}, Slices={len(slices)}, Nodes={len(nodes)}")

        prob = pulp.LpProblem("5G_Slice_R_A_Incremental", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "x", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            cat='Binary'
        )
        B_act_ul = pulp.LpVariable.dicts(
            "B_act_ul", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )
        B_act_dl = pulp.LpVariable.dicts(
            "B_act_dl", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )
        dev = pulp.LpVariable.dicts("dev", (s.snssai for s in slices), lowBound=0)
        change = pulp.LpVariable.dicts(
            "change", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )

        self._add_flow_constraints(prob, apps, slices, x, B_act_ul, B_act_dl, change)
        self._add_incremental_constraints(prob, apps, slices, x, B_act_ul, B_act_dl)
        self._add_slice_constraints(prob, apps, slices, B_act_ul, B_act_dl, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act_ul, B_act_dl)

        term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak = self._build_objectives(
            apps, slices, x, B_act_ul, B_act_dl, dev, change
        )

        objective = (self.config.w1 * term_load) + (self.config.w2 * term_sig) + (self.config.w3 * (term_exp + term_qos)) + term_tiebreak
        prob.setObjective(objective)
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status_str = pulp.LpStatus[prob.status]
        logger.info(f"增量求解完成. 状态: {status_str}")

        flow_results, slice_results, _, objective_val = self._finalize_results(prob, apps, slices, x, B_act_ul, B_act_dl)
        breakdown = self._build_objective_breakdown(
            term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak, objective_val
        )
        return flow_results, slice_results, status_str, objective_val, breakdown

    def solve_hybrid(self, apps: List[App], slices: List[Slice], nodes: List[Node]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[float], Dict[str, float]]:
        """
        混合优化：基于挤占的策略。
        1. 固定已有流的切片映射 (保持稳定性，避免大规模重路由)。
        2. 允许调整已有流的带宽 (Squeezing/Downgrading)，以便为高优先级新业务腾出空间。
        3. 优化目标依然包含体验(Priority weighted)，因此低优先级业务会被优先挤占。
        """
        logger.info(f"开始混合/挤占优化: Apps={len(apps)}, Slices={len(slices)}, Nodes={len(nodes)}")

        prob = pulp.LpProblem("5G_Slice_R_A_Hybrid", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "x", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            cat='Binary'
        )
        B_act_ul = pulp.LpVariable.dicts(
            "B_act_ul", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )
        B_act_dl = pulp.LpVariable.dicts(
            "B_act_dl", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )
        dev = pulp.LpVariable.dicts("dev", (s.snssai for s in slices), lowBound=0)
        change = pulp.LpVariable.dicts(
            "change", 
            ((a.app_id, f.flow_id, s.snssai) for a in apps for f in a.flows for s in slices), 
            lowBound=0
        )

        # 核心差异：使用 Hybrid 约束（只固定 x，不固定 B_act）
        self._add_flow_constraints(prob, apps, slices, x, B_act_ul, B_act_dl, change)
        self._add_hybrid_constraints(prob, apps, slices, x)
        self._add_slice_constraints(prob, apps, slices, B_act_ul, B_act_dl, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act_ul, B_act_dl)

        term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak = self._build_objectives(
            apps, slices, x, B_act_ul, B_act_dl, dev, change
        )

        objective = (self.config.w1 * term_load) + (self.config.w2 * term_sig) + (self.config.w3 * (term_exp + term_qos)) + term_tiebreak
        prob.setObjective(objective)
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status_str = pulp.LpStatus[prob.status]
        logger.info(f"混合求解完成. 状态: {status_str}")

        flow_results, slice_results, _, objective_val = self._finalize_results(prob, apps, slices, x, B_act_ul, B_act_dl)
        breakdown = self._build_objective_breakdown(
            term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak, objective_val
        )
        return flow_results, slice_results, status_str, objective_val, breakdown

    def _finalize_results(self, prob, apps, slices, x, B_act_ul, B_act_dl):
        status_str = pulp.LpStatus[prob.status]
        flow_results = self._format_results(apps, slices, x, B_act_ul, B_act_dl)
        slice_results = self._format_slice_stats(apps, slices, B_act_ul, B_act_dl)
        objective_val = pulp.value(prob.objective) if prob.objective is not None else 0.0
        return flow_results, slice_results, status_str, objective_val
    
    def _build_objectives(self, apps, slices, x, B_act_ul, B_act_dl, dev, change):
        num_slices = max(1, len(slices))
        num_flows = max(1, sum(len(a.flows) for a in apps))

        # 用软约束替换硬约束，计算违约数量
        term_qos = pulp.lpSum(
            (
                (1 if (s.latency + s.proc_delay > f.lat) else 0) + 
                (1 if (f.loss_req > 0 and s.loss > f.loss_req) else 0) +
                (1 if (f.jitter_req > 0 and s.jitter > f.jitter_req) else 0)
            ) * x[app.app_id, f.flow_id, s.snssai]
            for app in apps for f in app.flows for s in slices
        )

        term_exp = pulp.lpSum(
            (1.0 / f.priority if f.priority > 0 else 1.0) * (
                (f.bw_ul - pulp.lpSum(B_act_ul[app.app_id, f.flow_id, s.snssai] for s in slices)) / f.bw_ul if f.bw_ul > 0 else 0 +
                (f.bw_dl - pulp.lpSum(B_act_dl[app.app_id, f.flow_id, s.snssai] for s in slices)) / f.bw_dl if f.bw_dl > 0 else 0
            ) 
            for app in apps for f in app.flows
        )

        term_load_raw = pulp.lpSum(dev[s.snssai] for s in slices)
        term_sig_raw = pulp.lpSum(
            change[app.app_id, f.flow_id, s.snssai] 
            for app in apps for f in app.flows for s in slices
        )
        
        # 归一化
        term_load = term_load_raw / num_slices
        term_sig = term_sig_raw / num_flows
        term_qos = term_qos / num_flows

        # 唯一性: 极小扰动作为平局打破项 (不影响主目标)
        epsilon = 1e-6
        slice_index = {s.snssai: idx + 1 for idx, s in enumerate(slices)}
        term_tiebreak = epsilon * pulp.lpSum(
            slice_index[s.snssai] * x[app.app_id, f.flow_id, s.snssai]
            for app in apps for f in app.flows for s in slices
        )

        return term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak

    def _build_objective_breakdown(
        self,
        term_load_raw,
        term_sig_raw,
        term_load,
        term_sig,
        term_exp,
        term_qos,
        term_tiebreak,
        objective_val,
    ) -> Dict[str, float]:
        return {
            "load_raw": float(pulp.value(term_load_raw) or 0.0),
            "signal_raw": float(pulp.value(term_sig_raw) or 0.0),
            "load_norm": float(pulp.value(term_load) or 0.0),
            "signal_norm": float(pulp.value(term_sig) or 0.0),
            "exp": float(pulp.value(term_exp) or 0.0),
            "qos_norm": float(pulp.value(term_qos) or 0.0),
            "tiebreak": float(pulp.value(term_tiebreak) or 0.0),
            "objective_total": float(objective_val or 0.0),
        }

    def _add_hybrid_constraints(self, prob, apps, slices, x):
        """混合模式约束：仅固定已有流的切片映射，允许调整带宽。"""
        for app in apps:
            for f in app.flows:
                if not f.old_slice:
                    continue
                # 容错：如果原切片已不在切片列表中，则当作新流处理（允许迁移）
                if f.old_slice not in {s.snssai for s in slices}:
                    continue

                for s in slices:
                    if s.snssai == f.old_slice:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 1
                    else:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 0

    def _add_incremental_constraints(self, prob, apps, slices, x, B_act_ul, B_act_dl):
        """增量优化约束：固定已有流的切片映射，并在可行范围内保持历史带宽。"""
        for app in apps:
            for f in app.flows:
                if not f.old_slice:
                    continue

                if f.old_slice not in {s.snssai for s in slices}:
                    continue

                for s in slices:
                    if s.snssai == f.old_slice:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 1
                    else:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 0

                # 若历史带宽存在且落在 [gbr, bw] 内则固定；否则保持可调
                if f.old_allocated_bw_ul is not None:
                    fixed_ul = min(max(f.old_allocated_bw_ul, f.gbr_ul), f.bw_ul)
                    prob += B_act_ul[app.app_id, f.flow_id, f.old_slice] == fixed_ul
                if f.old_allocated_bw_dl is not None:
                    fixed_dl = min(max(f.old_allocated_bw_dl, f.gbr_dl), f.bw_dl)
                    prob += B_act_dl[app.app_id, f.flow_id, f.old_slice] == fixed_dl

    def _add_flow_constraints(self, prob, apps, slices, x, B_act_ul, B_act_dl, change):
        """添加流级别的约束"""
        for app in apps:
            for f in app.flows:
                # C1: 每个 Flow 必须选择且仅选择一个切片
                prob += pulp.lpSum(x[app.app_id, f.flow_id, s.snssai] for s in slices) == 1
                
                for s in slices:
                    # C2: [已放宽] URSP 类型兼容 -> 隐式控制

                    # C3: 时延/丢包/抖动约束 -> 移至 Stage 1 目标函数 (软约束)
                    # 避免因单个指标不满足导致无解
                    # 仅保留硬性逻辑约束

                    # C4: 上行带宽分配上限 & 下限保障
                    # 强制要求: 如果分配了切片, 至少分配 GBR 的请求带宽
                    prob += B_act_ul[app.app_id, f.flow_id, s.snssai] <= x[app.app_id, f.flow_id, s.snssai] * f.bw_ul
                    prob += B_act_ul[app.app_id, f.flow_id, s.snssai] >= x[app.app_id, f.flow_id, s.snssai] * f.gbr_ul
                    
                    # C5: 下行带宽分配上限 & 下限保障
                    prob += B_act_dl[app.app_id, f.flow_id, s.snssai] <= x[app.app_id, f.flow_id, s.snssai] * f.bw_dl
                    prob += B_act_dl[app.app_id, f.flow_id, s.snssai] >= x[app.app_id, f.flow_id, s.snssai] * f.gbr_dl
                    
                    # C6: 线性化信令开销 |x - x_old|
                    is_old = 1 if f.old_slice == s.snssai else 0 
                    prob += change[app.app_id, f.flow_id, s.snssai] >= x[app.app_id, f.flow_id, s.snssai] - is_old
                    prob += change[app.app_id, f.flow_id, s.snssai] >= is_old - x[app.app_id, f.flow_id, s.snssai]

    def _add_slice_constraints(self, prob, apps, slices, B_act_ul, B_act_dl, dev):
        for s in slices:
            # C6.1: 上行切片容量约束
            real_avail_ul = s.total_bw_ul - s.reserved_bw - s.current_load_bw_ul
            if real_avail_ul < 0: real_avail_ul = 0
            
            total_allocated_ul = pulp.lpSum(
                B_act_ul[app.app_id, f.flow_id, s.snssai] 
                for app in apps for f in app.flows
            )
            prob += total_allocated_ul <= real_avail_ul

            # C6.2: 下行切片容量约束
            real_avail_dl = s.total_bw_dl - s.reserved_bw - s.current_load_bw_dl
            if real_avail_dl < 0: real_avail_dl = 0
            
            total_allocated_dl = pulp.lpSum(
                B_act_dl[app.app_id, f.flow_id, s.snssai] 
                for app in apps for f in app.flows
            )
            prob += total_allocated_dl <= real_avail_dl

            # C7: 负载均衡辅助约束 (分别衡量上下行偏离目标负载率的情况)
            load_ratio_ul = (total_allocated_ul + s.reserved_bw + s.current_load_bw_ul) / s.total_bw_ul if s.total_bw_ul > 0 else 0
            load_ratio_dl = (total_allocated_dl + s.reserved_bw + s.current_load_bw_dl) / s.total_bw_dl if s.total_bw_dl > 0 else 0
            
            prob += dev[s.snssai] >= load_ratio_ul - self.config.rho
            prob += dev[s.snssai] >= self.config.rho - load_ratio_ul
            prob += dev[s.snssai] >= load_ratio_dl - self.config.rho
            prob += dev[s.snssai] >= self.config.rho - load_ratio_dl

    def _add_node_constraints(self, prob, apps, slices, nodes, B_act_ul, B_act_dl):
        for node in nodes:
            # 计算托管切片上的流量总和
            hosted_snssais = [s.snssai for s in slices if s.name in node.slices_hosted]

            current_node_cpu = sum((s.current_load_bw_ul + s.current_load_bw_dl) * self.config.alpha for s in slices if s.snssai in hosted_snssais)
            current_node_mem = sum((s.current_load_bw_ul + s.current_load_bw_dl) * self.config.beta for s in slices if s.snssai in hosted_snssais)

            new_traffic_sum = pulp.lpSum(
                B_act_ul[app.app_id, f.flow_id, s.snssai] + B_act_dl[app.app_id, f.flow_id, s.snssai]
                for s in slices if s.snssai in hosted_snssais
                for app in apps for f in app.flows
            )
            
            # C8: 物理节点 CPU
            prob += current_node_cpu + new_traffic_sum * self.config.alpha <= node.cpu_capacity

            # C9: 物理节点 Memory
            prob += current_node_mem + new_traffic_sum * self.config.beta <= node.memory_capacity

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
            
            # 总负载 = 动态分配 + 保留 + 静态负载
            total_load_ul = allocated_to_flows_ul + s.reserved_bw + s.current_load_bw_ul
            total_load_dl = allocated_to_flows_dl + s.reserved_bw + s.current_load_bw_dl
            
            load_ratio_ul = (total_load_ul / s.total_bw_ul) * 100 if s.total_bw_ul > 0 else 0
            load_ratio_dl = (total_load_dl / s.total_bw_dl) * 100 if s.total_bw_dl > 0 else 0
            
            remaining_ul = s.total_bw_ul - total_load_ul
            remaining_dl = s.total_bw_dl - total_load_dl

            slice_stats.append({
                "Slice": s.name,
                "SNSSAI": s.snssai,
                "Cap UL/DL (M)": f"{s.total_bw_ul}/{s.total_bw_dl}",
                "Alloc UL (M)": round(allocated_to_flows_ul, 2),
                "Alloc DL (M)": round(allocated_to_flows_dl, 2),
                "Load UL (%)": round(load_ratio_ul, 1),
                "Load DL (%)": round(load_ratio_dl, 1),
                "Rem UL (M)": round(remaining_ul, 2),
                "Rem DL (M)": round(remaining_dl, 2)
            })
        return pd.DataFrame(slice_stats)

    def _determine_strategy(self, flow: Flow, mapped_slice: Optional[str], allocated_bw_ul: float, allocated_bw_dl: float) -> List[str]:
        strategies = []
        if mapped_slice != flow.old_slice:
            strategies.append("策略B(重路由)")
        
        TOLERANCE = 0.01
        
        # 策略A: 拒绝/被抢占 (带宽归零或接近零)
        if allocated_bw_ul < TOLERANCE and allocated_bw_dl < TOLERANCE:
            strategies.append("策略A(拒绝/被抢占 UL)")

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
