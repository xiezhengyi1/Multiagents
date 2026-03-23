import pulp
import pandas as pd
from typing import List, Tuple, Optional, Dict

from .models import App, Slice, Node, Flow, OptimizationConfig, ServiceType, SLAProfile, Link, Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SliceOptimizationEngine:
    """切片资源分配优化引擎 - 支持单流粒度映射"""
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config

    def _safe_pulp_value(self, expr, default: float = 0.0) -> float:
        """关键步骤：统一处理求解失败/无解时的 None 值，避免结果格式化崩溃。"""
        value = pulp.value(expr)
        return float(value) if value is not None else float(default)

    def _mec_overhead_by_sst(self, sst: int) -> float:
        """关键步骤：根据切片 SST 返回 MEC 开销系数。"""
        overhead = getattr(self.config, "mec_overhead", [1.0]) or [1.0]
        idx = max(0, min(int(sst) - 1, len(overhead) - 1))
        return float(overhead[idx])

    def _create_common_variables(self, apps: List[App], slices: List[Slice]):
        """关键步骤：统一创建三种求解模式共用的决策变量。"""
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
        return x, B_act_ul, B_act_dl, dev, change

    def _solve_problem_and_collect(
        self,
        prob,
        apps,
        slices,
        x,
        B_act_ul,
        B_act_dl,
        term_load_raw,
        term_sig_raw,
        term_load,
        term_sig,
        term_exp,
        term_qos,
        term_tiebreak,
    ):
        """关键步骤：统一执行求解并返回结果与目标分解。"""
        objective = (self.config.w1 * term_load) + (self.config.w2 * term_sig) + (self.config.w3 * (term_exp + term_qos)) + term_tiebreak
        prob.setObjective(objective)
        solver_time_limit = int(getattr(self.config, "solver_time_limit", 30) or 30)
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit)
        prob.solve(solver)
        status_str = pulp.LpStatus[prob.status]

        flow_results, slice_results, _, objective_val = self._finalize_results(prob, apps, slices, x, B_act_ul, B_act_dl)
        breakdown = self._build_objective_breakdown(
            term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak, objective_val
        )
        return flow_results, slice_results, status_str, objective_val, breakdown

    def solve(self, apps: List[App], slices: List[Slice], nodes: List[Node]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[float], Dict[str, float]]:
        """
        构建并求解优化问题 (三权重单目标)
        目标函数:
            w1 * 负载均衡 + w2 * 信令开销 + w3 * 体验损失(含SLA违约惩罚)
        说明: 为保证解的唯一性，加入极小扰动项作为平局打破。
        """
        logger.info(f"开始优化(三权重): Apps={len(apps)}, Slices={len(slices)}, Nodes={len(nodes)}")

        prob = pulp.LpProblem("5G_Slice_R_A_Layered", pulp.LpMinimize)

        # 关键步骤：创建共用决策变量
        x, B_act_ul, B_act_dl, dev, change = self._create_common_variables(apps, slices)

        # 2. 约束条件构建
        self._add_flow_constraints(prob, apps, slices, x, B_act_ul, B_act_dl, change)
        self._add_slice_constraints(prob, apps, slices, B_act_ul, B_act_dl, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act_ul, B_act_dl)

        # 3. 构建目标函数项
        term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak = self._build_objectives(
            apps, slices, x, B_act_ul, B_act_dl, dev, change
        )

        flow_results, slice_results, status_str, objective_val, breakdown = self._solve_problem_and_collect(
            prob,
            apps,
            slices,
            x,
            B_act_ul,
            B_act_dl,
            term_load_raw,
            term_sig_raw,
            term_load,
            term_sig,
            term_exp,
            term_qos,
            term_tiebreak,
        )
        logger.info(f"求解完成. 状态: {status_str}")
        return flow_results, slice_results, status_str, objective_val, breakdown

    def solve_incremental(self, apps: List[App], slices: List[Slice], nodes: List[Node]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[float], Dict[str, float]]:
        """
        增量优化：固定已有流的原切片映射，仅优化新增/无历史的流。
        若旧分配带宽可用，则保持；否则允许在同一切片内微调以保证可行。
        """
        logger.info(f"开始增量优化: Apps={len(apps)}, Slices={len(slices)}, Nodes={len(nodes)}")

        prob = pulp.LpProblem("5G_Slice_R_A_Incremental", pulp.LpMinimize)

        # 关键步骤：创建共用决策变量
        x, B_act_ul, B_act_dl, dev, change = self._create_common_variables(apps, slices)

        self._add_flow_constraints(prob, apps, slices, x, B_act_ul, B_act_dl, change)
        self._add_incremental_constraints(prob, apps, slices, x, B_act_ul, B_act_dl)
        self._add_slice_constraints(prob, apps, slices, B_act_ul, B_act_dl, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act_ul, B_act_dl)

        term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak = self._build_objectives(
            apps, slices, x, B_act_ul, B_act_dl, dev, change
        )

        flow_results, slice_results, status_str, objective_val, breakdown = self._solve_problem_and_collect(
            prob,
            apps,
            slices,
            x,
            B_act_ul,
            B_act_dl,
            term_load_raw,
            term_sig_raw,
            term_load,
            term_sig,
            term_exp,
            term_qos,
            term_tiebreak,
        )
        logger.info(f"增量求解完成. 状态: {status_str}")
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

        # 关键步骤：创建共用决策变量
        x, B_act_ul, B_act_dl, dev, change = self._create_common_variables(apps, slices)

        # 核心差异：使用 Hybrid 约束（只固定 x，不固定 B_act）
        self._add_flow_constraints(prob, apps, slices, x, B_act_ul, B_act_dl, change)
        self._add_hybrid_constraints(prob, apps, slices, x)
        self._add_slice_constraints(prob, apps, slices, B_act_ul, B_act_dl, dev)
        self._add_node_constraints(prob, apps, slices, nodes, B_act_ul, B_act_dl)

        term_load_raw, term_sig_raw, term_load, term_sig, term_exp, term_qos, term_tiebreak = self._build_objectives(
            apps, slices, x, B_act_ul, B_act_dl, dev, change
        )

        flow_results, slice_results, status_str, objective_val, breakdown = self._solve_problem_and_collect(
            prob,
            apps,
            slices,
            x,
            B_act_ul,
            B_act_dl,
            term_load_raw,
            term_sig_raw,
            term_load,
            term_sig,
            term_exp,
            term_qos,
            term_tiebreak,
        )
        logger.info(f"混合求解完成. 状态: {status_str}")
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
            "load_raw": self._safe_pulp_value(term_load_raw),
            "signal_raw": self._safe_pulp_value(term_sig_raw),
            "load_norm": self._safe_pulp_value(term_load),
            "signal_norm": self._safe_pulp_value(term_sig),
            "exp": self._safe_pulp_value(term_exp),
            "qos_norm": self._safe_pulp_value(term_qos),
            "tiebreak": self._safe_pulp_value(term_tiebreak),
            "objective_total": float(objective_val or 0.0),
        }

    def _add_hybrid_constraints(self, prob, apps, slices, x):
        """混合模式约束：仅固定已有流的切片映射，允许调整带宽。"""
        valid_snssais = {s.snssai for s in slices}
        for app in apps:
            for f in app.flows:
                if not f.old_slice:
                    continue
                # 容错：如果原切片已不在切片列表中，则当作新流处理（允许迁移）
                if f.old_slice not in valid_snssais:
                    continue

                for s in slices:
                    if s.snssai == f.old_slice:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 1
                    else:
                        prob += x[app.app_id, f.flow_id, s.snssai] == 0

    def _add_incremental_constraints(self, prob, apps, slices, x, B_act_ul, B_act_dl):
        """增量优化约束：固定已有流的切片映射，并在可行范围内保持历史带宽。"""
        valid_snssais = {s.snssai for s in slices}
        for app in apps:
            for f in app.flows:
                if not f.old_slice:
                    continue

                if f.old_slice not in valid_snssais:
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
            new_traffic_sum = pulp.lpSum(
                B_act_ul[app.app_id, f.flow_id, s.snssai] + B_act_dl[app.app_id, f.flow_id, s.snssai]
                for s in slices if s.snssai in hosted_snssais
                for app in apps for f in app.flows
            )

            if node.type == "CN":
                # C8: 物理节点 CPU
                prob += node.sim_cpu_utilization * node.cpu_capacity + new_traffic_sum * self.config.alpha_cn <= node.cpu_capacity

            if node.type == "AN":
                # C8: 物理节点 PRB
                prob += new_traffic_sum * self.config.alpha_an + node.sim_cpu_utilization * node.cpu_capacity <= node.cpu_capacity
                prob += new_traffic_sum * self.config.prb  + node.sim_prb_utilization * node.prb_capacity <= node.prb_capacity

            if node.mec_capacity > 0 and hosted_snssais:
                # C9: MEC 资源约束（根据节点持有切片的 SST 决定系数）
                mec_load = pulp.lpSum(
                    self._mec_overhead_by_sst(s.sst)
                    * pulp.lpSum(
                        B_act_ul[app.app_id, f.flow_id, s.snssai] + B_act_dl[app.app_id, f.flow_id, s.snssai]
                        for app in apps for f in app.flows
                    )
                    for s in slices if s.snssai in hosted_snssais
                )
                prob += (1-node.sim_mec_utilization) * node.mec_capacity >= mec_load

            

    def _format_results(self, apps, slices, x, B_act_ul, B_act_dl) -> pd.DataFrame:
        results = []
        TOL = 1e-6
        for app in apps:
            for f in app.flows:
                mapped_slice = None
                allocated_bw_ul = 0.0
                allocated_bw_dl = 0.0
                
                for s in slices:
                    if self._safe_pulp_value(x[app.app_id, f.flow_id, s.snssai]) >= 1.0 - TOL:
                        mapped_slice = s.snssai # 记录 SNSSAI
                        allocated_bw_ul = self._safe_pulp_value(B_act_ul[app.app_id, f.flow_id, s.snssai])
                        allocated_bw_dl = self._safe_pulp_value(B_act_dl[app.app_id, f.flow_id, s.snssai])
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
                self._safe_pulp_value(B_act_ul[app.app_id, f.flow_id, s.snssai])
                for app in apps for f in app.flows
            )
            allocated_to_flows_dl = sum(
                self._safe_pulp_value(B_act_dl[app.app_id, f.flow_id, s.snssai])
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

class IBNSOptimizationEngine:
    """
    Intent-Based Network Slicing (IBNS) Optimization Engine.
    Implements P1 (Fulfillment) and P2 (Assurance) problems from the reference paper.
    """
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config
        # Default IBNS parameters
        self.v_a = 0.1  # VNF cost at AN
        self.v_n0 = 0.1 # VNF cost at CN
        self.c_u = 0.05  # MEC cost
        self.bandwidth_prb = 180000.0 # 180 kHz
        self.distance_km = 1.0

    def _safe_pulp_value(self, expr, default: float = 0.0) -> float:
        """关键步骤：统一处理求解失败/无解时的 None 值，避免结果格式化崩溃。"""
        value = pulp.value(expr)
        return float(value) if value is not None else float(default)

    def _effective_flow_throughput_bps(self, flow: Flow) -> float:
        """
        关键步骤：统一吞吐量需求建模。
        学术上双向业务常受上下行最弱方向约束，采用调和平均近似双向等效吞吐：
            R_eff = 2 * R_ul * R_dl / (R_ul + R_dl)
        若提供了 packet_size/arrival_rate，则优先使用该业务生成率。
        """
        if flow.packet_size > 0 and flow.arrival_rate > 0:
            return float(flow.packet_size * flow.arrival_rate)

        r_ul = max(0.0, float(flow.bw_ul) * 1e6)
        r_dl = max(0.0, float(flow.bw_dl) * 1e6)

        if r_ul > 0 and r_dl > 0:
            return (2.0 * r_ul * r_dl) / (r_ul + r_dl)
        return max(r_ul, r_dl)

    def calculate_snr(self, flow: Flow, an: Node) -> float:
        # Placeholder for SNR calculation (Delta_{u,a})
        # In a real scenario, this depends on location of user vs AN.
        return 100.0

    def calculate_transmission_rate(self, snr: float) -> float:
        """Eq (4): eta = upsilon * log2(1 + delta)"""
        import math
        return self.bandwidth_prb * math.log2(1 + snr)

    def solve_p1_fulfillment(self, 
                             flows: List[Flow], 
                             nodes: List[Node], 
                             links: List[Link], 
                             paths: List[Path],
                             service_types: Dict[int, ServiceType]) -> pd.DataFrame:
        """
        Solves the P1 problem: Resource Fulfillment (Maximizing Throughput)
        without SLA constraints.
        """
        return self._solve_optimization(flows, nodes, links, paths, service_types, p2_mode=False)

    def solve_p2_assurance(self, 
                           flows: List[Flow], 
                           nodes: List[Node], 
                           links: List[Link], 
                           paths: List[Path],
                           service_types: Dict[int, ServiceType],
                           sla_profiles: Dict[int, SLAProfile]) -> pd.DataFrame:
        """
        Solves the P2 problem: Service Assurance (SLA Compliance)
        adds SLA constraints to P1.
        """
        return self._solve_optimization(flows, nodes, links, paths, service_types, 
                                      p2_mode=True, sla_profiles=sla_profiles)

    def _solve_optimization(self, 
                            flows: List[Flow], 
                            nodes: List[Node], 
                            links: List[Link], 
                            paths: List[Path],
                            service_types: Dict[int, ServiceType],
                            p2_mode: bool = False,
                            sla_profiles: Dict[int, SLAProfile] = None) -> pd.DataFrame:
        
        # Helper for unit conversion (bps -> Mbps) for compute constraints
        def _dr_mbps(flow):
            return self._effective_flow_throughput_bps(flow) / 1e6

        prob = pulp.LpProblem("IBNS_Optimization", pulp.LpMaximize)

        # Helpers
        an_nodes = {n.id: n for n in nodes if n.is_an}
        cn_node = next((n for n in nodes if n.is_cn), None)
        
        # Paths map: AN_ID -> List[Path]
        paths_map = {}
        for p in paths:
            if p.an_id not in paths_map:
                paths_map[p.an_id] = []
            paths_map[p.an_id].append(p)
            
        link_map = {(l.u, l.v): l for l in links}

        # Sets
        U_set = range(len(flows))
        A_set = list(an_nodes.keys())
        flow_thr_bps = {u: self._effective_flow_throughput_bps(flows[u]) for u in U_set}
        
        # Variables
        # I[(u, a)]
        I = pulp.LpVariable.dicts("I", 
                                  ((u, a) for u in U_set for a in A_set), 
                                  cat='Binary')
        
        # T variables for path routing
        T_vars = {}
        for a in A_set:
            if a in paths_map:
                for p in paths_map[a]:
                    T_vars[(a, p.id)] = pulp.LpVariable(f"t_a{a}_p{p.id}", lowBound=0)

        # Objective: Maximize Throughput
        prob += pulp.lpSum([
            flow_thr_bps[u] * I[(u, a)] 
            for u in U_set for a in A_set
        ]), "Objective_Throughput"

        # --- Constraints ---

        # (3) UE Association: Each UE connected to at most 1 AN
        for u in U_set:
            prob += pulp.lpSum([I[(u, a)] for a in A_set]) <= 1, f"Assoc_UE_{u}"

        # (5) AN PRB Capacity
        # We need to allocate enough PRBs to satisfy BOTH throughput demand AND latency requirements.
        # Effective Rate = max(DataRate, PacketSize / (Latency - Propagation - Routing))
        req_rates = {} # Store for latency check
        
        for a in A_set:
            an_node = an_nodes[a]
            usage_exprs = []
            for u in U_set:
                flow = flows[u]
                snr = self.calculate_snr(flow, an_node)
                eta = self.calculate_transmission_rate(snr)
                if eta <= 0: continue
                
                # Fix 3: packet_size=0 fallback
                pkt_size = flow.packet_size if flow.packet_size > 0 else max(flow_thr_bps[u] * 0.001, 12000)

                # Estimate delays to determine required rate for latency
                d_prp = 0.0001 # 0.1ms
                d_rot = 0.0001
                if a in paths_map and paths_map[a]:
                    path = paths_map[a][0]
                    # d_rot logic
                    path_delay = sum([pkt_size / link_map[l].capacity for l in path.links if l in link_map])
                    if path_delay > 0: d_rot = path_delay
                
                max_trn_delay = flow.lat - d_prp - d_rot
                if max_trn_delay <= 0:
                    # Impossible to meet latency
                    req_rate_lat = float('inf')
                else:
                    req_rate_lat = pkt_size / max_trn_delay
                
                # The effective rate we must allocate
                eff_rate = max(flow_thr_bps[u], req_rate_lat)
                req_rates[(u, a)] = (eff_rate, d_prp, d_rot) # Store for (15)

                rho_needed = eff_rate / eta
                usage_exprs.append( I[(u, a)] * rho_needed )
            
            prob += pulp.lpSum(usage_exprs) <= an_node.prb_capacity, f"PRB_Capacity_AN_{a}"

        # (6) VNF Computation at AN
        for a in A_set:
            an_node = an_nodes[a]
            prob += pulp.lpSum([
                I[(u, a)] * _dr_mbps(flows[u]) * self.v_a 
                for u in U_set
            ]) <= an_node.cpu_capacity, f"VNF_Comp_AN_{a}"

        # (7) VNF Computation at CN
        if cn_node:
             prob += pulp.lpSum([
                I[(u, a)] * _dr_mbps(flows[u]) * self.v_n0
                for u in U_set for a in A_set
            ]) <= cn_node.cpu_capacity, f"VNF_Comp_CN"

        # (8) MEC Computation at AN (Local)
        for a in A_set:
            an_node = an_nodes[a]
            mec_usage = []
            for u in U_set:
                flow = flows[u]
                if flow.service_type_id in service_types:
                    srv = service_types[flow.service_type_id]
                    # If Critical Latency (1 in critical_kpis), use AN MEC
                    if 1 in srv.critical_kpis:
                        mec_usage.append( I[(u, a)] * _dr_mbps(flow) * self.c_u )
            
            prob += pulp.lpSum(mec_usage) <= an_node.mec_capacity, f"MEC_Comp_AN_{a}"

        # (9) MEC Computation at CN (Central)
        if cn_node:
            cn_mec_usage = []
            for u in U_set:
                flow = flows[u]
                if flow.service_type_id in service_types:
                    srv = service_types[flow.service_type_id]
                    if 1 not in srv.critical_kpis:
                         for a in A_set:
                            cn_mec_usage.append( I[(u, a)] * _dr_mbps(flow) * self.c_u )
            prob += pulp.lpSum(cn_mec_usage) <= cn_node.mec_capacity, f"MEC_Comp_CN"

        # (10) TN Routing
        for a in A_set:
            if a in paths_map:
                traffic_demand = pulp.lpSum([
                    flow_thr_bps[u] * I[(u, a)] 
                    for u in U_set
                ])
                path_supply = pulp.lpSum([
                    T_vars[(a, p.id)] for p in paths_map[a]
                ])
                prob += path_supply == traffic_demand, f"TN_Routing_Demand_AN_{a}"

        # (11) Link Capacity
        for link_key, link in link_map.items():
            link_load = []
            for a in A_set:
                if a in paths_map:
                    for p in paths_map[a]:
                        # Check if link is in path p.links (list of tuples)
                        # We need to handle directionality or assume undirected match
                        if link_key in p.links or (link_key[1], link_key[0]) in p.links: 
                             # Assuming straightforward match for now
                             link_load.append(T_vars[(a, p.id)])
            if link_load:
                prob += pulp.lpSum(link_load) <= link.capacity, f"Link_Cap_{link_key}"

        # (15) Global Latency Constraint
        lhs_terms = []
        rhs_terms = []
        
        for u in U_set:
            flow = flows[u]
            # Fix 3 (continued): Need pkt_size here too
            pkt_size = flow.packet_size if flow.packet_size > 0 else max(flow_thr_bps[u] * 0.001, 12000)
            
            for a in A_set:
                snr = self.calculate_snr(flow, an_nodes[a])
                eta = self.calculate_transmission_rate(snr)
                if eta <= 0: continue
                
                # Retrieve parameters calculated in (5)
                eff_rate, d_prp, d_rot = req_rates.get((u, a), (flow_thr_bps[u], 0.0001, 0.0001))
                
                # Check fundamental limits again (using the precomputed values)
                if eff_rate > 1e12 or eff_rate <= 0: # Indicates infinity
                    d_tot = 999.0 # Impossible
                else:
                    d_trn = pkt_size / (eff_rate + 1e-9)
                    d_tot = d_trn + d_prp + d_rot
                
                # If calculated d_tot exceeds limit due to numerical issues
                if d_tot > flow.lat + 1e-9:
                    d_tot = 999.0
                
                # For constraint (15), we sum latency requirements vs achieved
                lhs_terms.append(flow.lat * I[(u, a)])
                rhs_terms.append(d_tot * I[(u, a)])
        
        prob += pulp.lpSum(lhs_terms) >= pulp.lpSum(rhs_terms), "Latency_Compliance"

        # --- P2 Constraints (SLA) ---
        if p2_mode and sla_profiles:
            for s_id, srv in service_types.items():
                 ues_in_slice = [u for u in U_set if flows[u].service_type_id == s_id]
                 if not ues_in_slice: continue
                 
                 # Check if profile exists
                 profile = next((p for p in sla_profiles.values() if p.service_type_id == s_id), None)
                 if not profile: continue

                 thresholds = profile.kpi_thresholds

                 total_ues_count = len(ues_in_slice)
                 
                 # (23) Delay KPI - Fix 4
                 if 1 in thresholds:
                     tau = thresholds[1]
                     for u in ues_in_slice:
                        for a in A_set:
                            if (u, a) in req_rates:
                                eff_rate, d_prp, d_rot = req_rates[(u, a)]
                                flow = flows[u]
                                pkt_size = flow.packet_size if flow.packet_size > 0 else max(flow_thr_bps[u] * 0.001, 12000)
                                d_trn = pkt_size / (eff_rate + 1e-9)
                                d_tot = d_trn + d_prp + d_rot
                                if d_tot > tau:
                                    prob += I[(u, a)] == 0, f"SLA_Delay_Limit_{s_id}_{u}_{a}"

                 # (24) Throughput KPI - Fix 5
                 if 2 in thresholds:
                     tau = thresholds[2] # Mbps
                     prob += pulp.lpSum([
                         flow_thr_bps[u] * I[(u, a)] 
                         for u in ues_in_slice for a in A_set
                     ]) >= tau * 1e6, f"SLA_Throughput_Slice_{s_id}"
                 
                 # (25) Reliability KPI - Fix 6
                 if 3 in thresholds:
                     tau = thresholds[3]
                     prob += pulp.lpSum([
                         I[(u,a)] for u in ues_in_slice for a in A_set
                     ]) >= tau * total_ues_count, f"SLA_Reliability_Slice_{s_id}"

                 # (26) Connection Density - Fix 7
                 if 4 in thresholds:
                     # tau = thresholds[4]
                     prob += pulp.lpSum([
                         I[(u, a)] for u in ues_in_slice for a in A_set
                     ]) >= total_ues_count, f"SLA_Connection_Slice_{s_id}"
        
        # Solve
        solver_time_limit = int(getattr(self.config, "solver_time_limit", 30) or 30)
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))
        
        # Collect results
        results = []
        if status == pulp.LpStatusOptimal or status == 1: # 1 is Optimal in constants
            for u in U_set:
                flow = flows[u]
                allocated = False
                for a in A_set:
                    if self._safe_pulp_value(I[(u, a)]) >= 1.0 - 1e-6:
                        allocated = True
                        # Find path used?
                        # This is tricky as traffic is split on T vars, but we want to know which AN used.
                        # Since I[(u,a)] is binary, we know the AN.
                        # The path is internal to TN routing optimization.
                        results.append({
                            "Flow ID": flow.flow_id,
                            "Service Type": service_types[flow.service_type_id].name if flow.service_type_id in service_types else "Unknown",
                            "Allocated AN": an_nodes[a].name,
                            "Status": "Served",
                            "Throughput": flow_thr_bps[u],
                            "Req BW UL": flow.bw_ul,
                            "Req BW DL": flow.bw_dl
                        })
                        break
                if not allocated:
                    results.append({
                        "Flow ID": flow.flow_id,
                        "Service Type": service_types[flow.service_type_id].name if flow.service_type_id in service_types else "Unknown",
                        "Allocated AN": "None",
                        "Status": "Unserved",
                        "Throughput": 0.0,
                        "Req BW UL": flow.bw_ul,
                        "Req BW DL": flow.bw_dl
                    })
        else:
             logger.warning("Optimization failed to find optimal solution")
        
        return pd.DataFrame(results)
