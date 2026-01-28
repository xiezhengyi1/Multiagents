from typing import Dict, Any, Union
import json
import logging
from .db_tool import update_scenario_in_db, get_cached_scenario, deserialize_scenario_payload
from utils.logger import setup_logger

logger = setup_logger(__name__)

def commit_optimization_result_to_db(optimization_result: Union[Dict, str]) -> str:
    """
    将优化结果（内存中的最新状态）提交并保存到数据库。
    此工具供 PolicyDispatchAgent 在确认策略下发并验证监控指标符合预期后调用。
    
    Args:
        optimization_result: 优化步骤返回的结果字典或 JSON 字符串。
                             (实际上此工具更依赖于 global context 中的内存状态，但接收此参数以确保流程依赖性)
                             
    Returns:
        执行结果状态消息
    """
    logger.info("正在将优化后的网络状态保存至数据库 (SemanticKnowledge)...")
    
    # 1. 验证输入 (其实主要作为触发信号)
    if isinstance(optimization_result, str):
        try:
            # 尝试解析只要确定有效性
            json.loads(optimization_result)
        except:
            pass # 也可以容忍字符串描述
            
    # 2. 优先从工具输入解析场景
    parsed = deserialize_scenario_payload(optimization_result)
    if parsed:
        apps, slices, nodes = parsed
    else:
        # 3. 从缓存获取最新场景
        apps, slices, nodes = get_cached_scenario()

    if not apps or not slices or not nodes:
        return "Error: 未找到可提交的场景数据。请先执行优化求解并确保结果被缓存或传入。"

    # 4. 执行 DB 更新
    success = update_scenario_in_db(apps, slices, nodes)
    
    if success:
        return "Success: 网络切片状态已成功同步到数据库。"
    else:
        return "Failed: 数据库更新过程发生错误，请查看日志。"
