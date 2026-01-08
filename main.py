import os
import sys
from multi_agents.IntentEncodingAgent import IntentEncodingAgent
from multi_agents.OptimizationStrategyAgent import OptimizationStrategyAgent
from utils.logger import setup_logger

logger = setup_logger(name="MainSystem")

def main():
    logger.info("=== 5G Network Slicing Multi-Agent System Demo ===")
    
    # 简单的环境检查
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("警告: 未检测到 OPENAI_API_KEY 环境变量。请在 .env 文件或系统环境变量中配置。")
        # 这里的Demo可能会失败，除非用户设置了Forwarding或本地大模型
    
    # 1. 实例化智能体
    intent_agent = IntentEncodingAgent()
    strategy_agent = OptimizationStrategyAgent()

    # 2. 模拟用户输入 (模拟App_4接入场景)
    user_input = """
    我是应急抢险指挥车(App_4)，现在需要接入网络。
    我有两个主要业务流：
    1. 远程机械臂控制，这是关键控制流，带宽大概20Mbps，但这涉及到生命安全，时延不能超过10ms。
    2. 现场多路4K高清直播，带宽需要50Mbps，时延60ms左右就行。
    """
    
    logger.info(f"\n[Step 1] 用户输入:\n{user_input}")

    # 3. Intent Encoding Agent 工作
    logger.info("\n[Step 2] Intent Encoding Agent 正在分析意图...")
    user_intent = intent_agent.analyze_intent(user_input)
    
    if user_intent:
        logger.info("\n>>> 分析结果:")
        logger.info(f"应用名称: {user_intent.app_name}")
        logger.info(f"整体紧急度: {user_intent.urgency}")
        logger.info("业务流详情:")
        for flow in user_intent.flows:
            logger.info(f"  - ID: {flow.flow_id} | Type: {flow.business_type} | BW: {flow.bandwidth_demand}Mbps | Delay: {flow.latency_requirement}ms | Priority: {flow.priority_level}")
    else:
        logger.error("意图分析失败，终止流程。")
        return

    # 4. 模拟网络状态
    # S1 (URLLC): 剩余 15Mbps (不足以支撑流1的20Mbps)
    # S2 (eMBB): 剩余 200Mbps
    network_status = """
    当前切片资源状态:
    - Slice S1 (URLLC): 总容量 100Mbps, 剩余可用 15Mbps, 链路时延 8ms.
    - Slice S2 (eMBB): 总容量 1000Mbps, 剩余可用 200Mbps, 链路时延 45ms.
    - Slice S3 (mMTC): 总容量 50Mbps, 剩余可用 40Mbps.
    """
    logger.info(f"\n[Step 3] 获取当前网络状态:\n{network_status}")

    # 5. Optimization Strategy Agent 工作
    logger.info("\n[Step 4] Optimization Strategy Agent 正在制定策略并调用求解器...")
    # 将Pydantic对象转为dict传给下一个Agent
    intent_dict = user_intent.dict()
    final_report = strategy_agent.generate_strategy(intent_dict, network_status)
    
    logger.info("\n>>> 最终执行报告:")
    logger.info(final_report)

if __name__ == "__main__":
    main()
