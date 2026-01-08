from typing import List, Dict, Any
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from .basemodel import BaseAgent
from utils.logger import setup_logger
from tools.optimization import optimize_network_slices

logger = setup_logger(name="OptimizationStrategyAgent")

# --- 定义工具 ---
@tool
def run_optimization_solver(w1: float, w2: float, w3: float, app_details: str) -> str:
    """
    调用底层优化求解器(Solver)，传入权重参数和应用详情，计算网络切片分配结果。
    这是执行网络资源分配的唯一方式。
    
    Args:
        w1: 负载均衡权重
        w2: 信令开销权重
        w3: 业务体验损失权重 (如果业务非常重要，请大幅提高此权重)
        app_details: 新应用的详细信息的 JSON 字符串 (包含 app_name, type, flows 等)
    
    Returns:
        优化结果的文本报告，包含分配的切片、策略动作和当前网络负载。
    """
    try:
        # 尝试解析 app_details JSON，如果已经是dict了则不用解析(有些模型会传dict)
        if isinstance(app_details, str):
            app_data = json.loads(app_details)
        else:
            app_data = app_details
            
        result = optimize_network_slices(app_data, w1, w2, w3)
        logger.info(f"Optimization Solver Result:\n{result}")
        return result
    except Exception as e:
        return f"工具执行错误: {str(e)}"

# --- Agent 定义 ---

class OptimizationStrategyAgent(BaseAgent):
    def __init__(self, model_name="qwen-plus"):
        super().__init__(model_name=model_name)
        
        # 1. 定义工具集
        self.tools = [run_optimization_solver]
        
        # 2. 绑定工具到 LLM
        # 直接使用 bind_tools 而不是 AgentExecutor，已避免复杂的依赖问题
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 3. 工具映射（用于手动执行）
        self.tool_map = {tool.name: tool for tool in self.tools}

    def generate_strategy(self, user_intent: dict, network_status: str) -> str:
        """
        执行策略生成与求解流程 (手动实现 ReAct/ToolCall 循环)
        """
        # 为了方便 Tool 调用，我们将 user_intent 转为JSON字符串传给 prompt
        intent_json = json.dumps(user_intent, ensure_ascii=False)
        
        # 构造初始 Prompt
        system_prompt = """
        你是一个5G网络切片系统的优化决策与执行智能体 (DSA - Decision & Solver Agent)。
        你的职责是：
        1. 分析用户的接入意图和网络状态。
        2. 制定优化目标函数的权重 (w1, w2, w3)。
        3. **必须调用** `run_optimization_solver` 工具来执行具体的资源分配计算。
        4. 原封不动地返回工具返回的结果，并向用户汇报最终的策略执行情况。
        
        权重参数说明：
        - w1 (负载均衡): 防止单点拥塞。
        - w2 (信令开销): 减少配置变动。
        - w3 (体验损失): **关键!** 当必须保障高优先级业务（如URLLC、生命安全）接入时，必须将 w3 设置得非常高 (例如 >1000)，以允许挤占低优先级业务。
        
        策略参考：
        - 如果资源紧张且新业务极其重要 -> 极大提高 w3 (1000~5000)，适当降低 w1/w2，触发挤占(Preemption)策略。
        - 如果资源充足 -> 均衡设置 w1, w2, w3 (例如 100, 50, 100)。
        """
        
        user_prompt = f"""
        用户意图:
        {intent_json}
        
        当前网络状态:
        {network_status}
        
        请分析并执行。
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Round 1: LLM 思考并决定调用工具
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # 检查是否有工具调用
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    logger.info(f"LLM 决定调用工具: {tool_name} 参数: {tool_args}")
                    
                    # 执行工具
                    if tool_name in self.tool_map:
                        tool_instance = self.tool_map[tool_name]
                        # 运行 tool
                        tool_result = tool_instance.invoke(tool_args)
                    else:
                        tool_result = f"Error: Tool {tool_name} not found."
                    
                    # 将工具结果添加回消息历史
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                
                # Round 2: LLM 根据工具结果生成最终回答
                final_response = self.llm_with_tools.invoke(messages)
                return final_response.content
                
            else:
                # 如果 LLM 没有调用工具，直接返回其回复（可能是拒绝执行或询问更多信息）
                logger.warning("Agent 未调用优化工具，直接返回结果。")
                return response.content

        except Exception as e:
            logger.error(f"Agent执行出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"策略执行失败: {str(e)}"

