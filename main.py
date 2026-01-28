import os
import sys
from multi_agents.IntentEncodingAgent import IntentEncodingAgent
from multi_agents.OptimizationStrategyAgent import OptimizationStrategyAgent
from multi_agents.PolicyDispatchAgent import PolicyDispatchAgent
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
    dispatch_agent = PolicyDispatchAgent()

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
            # 注意：字段名需与 IntentEncodingAgent 定义一致
            logger.info(f"  - ID: {flow.flow_id} | Type: {flow.business_type} | BW(DL): {flow.bw_dl}Mbps | Delay: {flow.lat}ms | Priority: {flow.priority}")
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
    strategy_output = strategy_agent.generate_strategy(intent_dict)
    
    logger.info("\n>>> 最终执行报告:")
    if hasattr(strategy_output, "model_dump_json"):
        strategy_json = strategy_output.model_dump_json(indent=2, ensure_ascii=False)
        logger.info(strategy_json)
    else:
        strategy_json = str(strategy_output)
        logger.info(strategy_output)

    # 6. Policy Dispatch Agent 工作 (闭环反馈)
    logger.info("\n[Step 5] Policy Dispatch Agent 正在下发策略并验证反馈...")
    feedback_report = dispatch_agent.execute_and_evaluate(strategy_output)

    logger.info("\n>>> 策略执行反馈报告:")
    logger.info(f"执行状态: {feedback_report.execution_status}")
    logger.info(f"网元性能指标: {feedback_report.performance_metrics}")
    logger.info(f"违规详情: {feedback_report.violation_details}")
    logger.info(f"修正建议: {feedback_report.correction_suggestion}")

    # 简单展示闭环逻辑
    if feedback_report.correction_suggestion and feedback_report.correction_suggestion != "None":
        logger.warning(f"检测到策略执行问题，建议触发【意图修正】或【策略重算】流程。建议内容: {feedback_report.correction_suggestion}")
    else:
        logger.success("策略执行成功，闭环结束。") if hasattr(logger, 'success') else logger.info("策略执行成功，闭环结束。")

if __name__ == "__main__":
    main()
