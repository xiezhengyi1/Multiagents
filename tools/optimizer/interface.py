import uuid
from typing import List
from .models import App, Flow, OptimizationConfig
from .engine import SliceOptimizationEngine
from .data import get_initial_scenario, _GLOBAL_SCENARIO_CONTEXT
from utils.logger import setup_logger

logger = setup_logger(__name__)

def optimize_network_slices(new_app_data: dict, w1: float, w2: float, w3: float) -> str:
    """
    执行网络切片资源优化求解 (粒度: Flow)
    优化逻辑：
    1. 识别应用：基于 app_id 或 app_name 查找是否为已有应用
    2. 识别/构建流：尝试将新请求的流与已有流进行匹配（基于ID或特征），以为了继承 old_slice 状态
    3. 全局更新：求解后，用新的应用状态覆盖旧状态
    """
    try:
        if _GLOBAL_SCENARIO_CONTEXT["apps"] is not None:
             apps = _GLOBAL_SCENARIO_CONTEXT["apps"]
             slices = _GLOBAL_SCENARIO_CONTEXT["slices"]
             nodes = _GLOBAL_SCENARIO_CONTEXT["nodes"]
        else:
             apps, slices, nodes = get_initial_scenario()
        
        # B. 应用身份识别
        target_app_id = new_app_data.get('app_id')
        target_app_name = new_app_data.get('name', 'NewApp')
        
        # 查找现有应用 (ID优先，Name其次)
        existing_app = next((a for a in apps if (target_app_id and a.app_id == target_app_id) or a.name == target_app_name), None)
        if existing_app:
            final_app_id = existing_app.app_id
        else:
            final_app_id = target_app_id if target_app_id else f"{target_app_name}_{uuid.uuid4().hex[:4]}"

        # C. 流构建与匹配 (Flow Construction & Mapping)
        existing_flow_map = {f.flow_id: f for f in existing_app.flows} if existing_app else {}
        new_flows_obj = []
        
        for i, f_data in enumerate(new_app_data.get('flows', [])):
            # 1. 基础属性解析
            bw_ul_val = f_data.get('bw_ul', 0)
            bw_dl_val = f_data.get('bw_dl', 0)
            f_desc = f_data.get('name', f'flow_{i}')
            f_prio = f_data.get('priority', 10)
            f_lat = f_data.get('lat', 100)
            
            # 2. 确定 Flow ID (标准化格式: {AppID}_{Suffix})
            raw_fid = f_data.get('flow_id')
            if raw_fid:
                # 如果输入自带App前缀则保留，否则拼接
                current_f_id = raw_fid if raw_fid.startswith(f"{final_app_id}_") else f"{final_app_id}_{raw_fid}"
            else:
                current_f_id = f"{final_app_id}_f{i}_{f_desc.replace(' ', '_')}"
            
            # 3. 继承旧状态
            matched_old_flow = existing_flow_map.get(current_f_id)
            print(matched_old_flow)
 
            new_flows_obj.append(Flow(
                name=f_desc,
                bw_ul=bw_ul_val,
                bw_dl=bw_dl_val,
                lat=f_lat,
                priority=f_prio,
                flow_id=current_f_id,
                old_slice=matched_old_flow.old_slice if matched_old_flow else None,
                old_allocated_bw_ul=matched_old_flow.old_allocated_bw_ul if matched_old_flow else None,
                old_allocated_bw_dl=matched_old_flow.old_allocated_bw_dl if matched_old_flow else None
            ))

        # D. 构建新 App 对象
        new_app = App(
            name=target_app_name,
            app_id=final_app_id,
            flows=new_flows_obj,
        )
        
        # E. 调用求解引擎
        config = OptimizationConfig(w1=w1, w2=w2, w3=w3)
        engine = SliceOptimizationEngine(config)
        
        active_apps = [a for a in apps if a.app_id != final_app_id and a.name != target_app_name]
        updated_apps_list = active_apps + [new_app]
        
        results_df, slice_stats_df = engine.solve(updated_apps_list, slices, nodes)
        
        if results_df.empty:
            return "求解器未返回结果 (Empty Result)。"

        # F. 更新全局状态 (Side Effect)
        if _GLOBAL_SCENARIO_CONTEXT["apps"] is not None:
             _GLOBAL_SCENARIO_CONTEXT["apps"][:] = updated_apps_list
             
             # 回写状态
             for _, row in results_df.iterrows():
                 r_app_id = row['App ID']
                 r_flow_id = row['Flow ID']
                 r_new_slice = row['New Slice']
                 r_act_bw_ul = row['Act BW UL']
                 r_act_bw_dl = row['Act BW DL']
                 
                 target_app_obj = next((a for a in _GLOBAL_SCENARIO_CONTEXT["apps"] if a.app_id == r_app_id), None)
                 if target_app_obj:
                     target_flow_obj = next((f for f in target_app_obj.flows if f.flow_id == r_flow_id), None)
                     if target_flow_obj:
                         target_flow_obj.old_slice = r_new_slice
                         target_flow_obj.old_allocated_bw_ul = r_act_bw_ul
                         target_flow_obj.old_allocated_bw_dl = r_act_bw_dl

        # G. 生成报告
        output = []
        output.append(f"--- 优化求解报告 (Flow Level) ---")
        output.append(f"参数: w1={w1}, w2={w2}, w3={w3}")
        
        # 1. 本次业务结果
        my_results = results_df[results_df['App ID'] == final_app_id]
        if not my_results.empty:
            output.append(f"业务 '{target_app_name}' (ID: {final_app_id}) 分配详情:")
            for _, row in my_results.iterrows():
                strategy_note = f", 策略: {row['Strategies']}" if row['Strategies'] != "保持" else ""
                output.append(
                    f"  - 流 [{row['Flow Name']}] -> 切片: {row['New Slice'] or '未分配'} "
                    f"(分配带宽 UL: {row['Act BW UL']}/{row['Req BW UL']}M, DL: {row['Act BW DL']}/{row['Req BW DL']}M{strategy_note})"
                )
        else:
            output.append(f"警告: 结果中未找到目标业务 {target_app_name}")

        # 2. 其他受影响业务
        other_results = results_df[results_df['App ID'] != final_app_id]
        impacted_mask = other_results['Strategies'] != "保持"
        impacted_df = other_results[impacted_mask]
        
        if not impacted_df.empty:
            output.append("\n其他受影响的业务流:")
            for _, row in impacted_df.iterrows():
                output.append(
                    f"  - {row['App']} / [{row['Flow Name']}] -> {row['New Slice']} "
                    f"(策略: {row['Strategies']})"
                )

        output.append("\n各切片资源状态:")
        for _, row in slice_stats_df.iterrows():
            output.append(f"  - {row['Slice']} ({row['SNSSAI']}): 负载 {row['Load Ratio (%)']}% (剩余 {row['Remaining (M)']}M)")

        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"优化过程发生异常: {e}", exc_info=True)
        return f"系统错误: 优化求解失败 - {str(e)}"
