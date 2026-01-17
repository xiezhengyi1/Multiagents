from typing import List, Dict, Any, Union
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from .basemodel import BaseAgent
from tools.optimizer import optimize_network_slices
from tools.network_status import get_network_status

logger = logging.getLogger("OptimizationStrategyAgent")

# --- 定义工具 ---
@tool
def fetch_network_status() -> str:
    """
    获取当前网络切片和节点的状态摘要。
    
    Returns:
        包含切片资源使用情况、节点状态的文本报告。
    """
    try:
        status = get_network_status()
        logger.info("Fetched network status.")
        logger.info(f"Network Status:\n{status}")
        return status
    except Exception as e:
        return f"获取网络状态失败: {str(e)}"

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
    
# --- Output 定义 ---

from model.UrspRuleRequest import UrspRuleRequest
from model.SmPolicyDecision import SmPolicyDecision

class Strategy(BaseModel):
    """优化策略输出结构"""
    recommended_actions: List[str] = Field(..., description="推荐的具体策略动作列表")
    policy_type: str = Field(..., description="输出的策略类型，如 'UrspRuleRequest' 或 'SmPolicyDecision'")
    policy_details: Union[UrspRuleRequest, SmPolicyDecision] = Field(..., description="具体的策略内容，根据 policy_type 决定")

class OutputStrategy(BaseModel):
    """优化策略输出封装"""
    all_policies: List[Strategy] = Field(..., description="所有推荐的策略列表")

# --- Agent 定义 ---

class OptimizationStrategyAgent(BaseAgent):
    def __init__(self, model_name="qwen-plus"):
        super().__init__(model_name=model_name)
        
        # 1. 定义工具集
        self.tools = [run_optimization_solver, fetch_network_status]
        
        # 2. 绑定工具到 LLM
        # 直接使用 bind_tools 而不是 AgentExecutor，已避免复杂的依赖问题
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 3. 工具映射（用于手动执行）
        self.tool_map = {tool.name: tool for tool in self.tools}

        # 4. 定义输出解析器 (可选)
        self.output_parser = PydanticOutputParser(pydantic_object=OutputStrategy)

    def generate_strategy(self, user_intent: dict) -> OutputStrategy:
        """
        执行策略生成与求解流程 (手动实现 ReAct/ToolCall 循环)
        """
        # 为了方便 Tool 调用，我们将 user_intent 转为JSON字符串传给 prompt
        intent_json = json.dumps(user_intent, ensure_ascii=False)
        
        # 构造初始 Prompt
        system_prompt = """
        你是一个5G网络切片系统的优化决策与执行智能体 (DSA - Decision & Solver Agent)。
        你的职责是：
        1. **首先必须调用** `fetch_network_status` 工具获取当前网络切片状态。
        2. 分析用户的接入意图和获取到的网络状态。
        3. 制定优化目标函数的权重 (w1, w2, w3)。
        4. **必须调用** `run_optimization_solver` 工具来执行具体的资源分配计算。
        5. **最后**，根据工具执行结果，生成结构化的策略输出 (OutputStrategy)。
        
        权重参数说明：
        - w1 (负载均衡): 防止单点拥塞。
        - w2 (信令开销): 减少配置变动。
        - w3 (体验损失): **关键!** 当必须保障高优先级业务（如URLLC、生命安全）接入时，一般 >1000。
        
        【输出格式决策逻辑】
        根据优化求解器的结果中显示的"策略"，决定输出的 `policy_type` 和 `policy_details`：
        
        A. 如果结果包含 "策略B(重路由)" -> 意味着需要终端发起新连接到新切片
           - policy_type: "UrspRuleRequest"
           - policy_details: 必须包含 `routeSelParamSets`。
             * 从优化结果中提取新切片的 S-NSSAI (例如 "01000001" -> sst=1, sd="000001")。
             * 构造结构: {{ "routeSelParamSets": [ {{ "dnn": "default", "snssai": {{ "sst": 1, "sd": "000001" }}, "precedence": 1 }} ], "relatPrecedence": 1 }}
           - 再生成一项发起新连接请求后，应该发出的 "SmPolicyDecision"
        
        B. 如果结果是 "策略A", "策略C", "策略D" 或 "保持" -> 意味着网络侧控制资源
           - policy_type: "SmPolicyDecision"
           - policy_details:必须包含 `pccRules` 和 `qosDecs`。
             * 根据优化结果的 "Act BW" 设置 `maxbrDl` / `maxbrUl`。
             * 构造结构: {{ "pccRules": {{ ... }}, "qosDecs": {{ ... }} }} (请生成合理的默认值)

        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "用户意图:\n{intent_json}\n\n请获取网络状态并据此执行优化策略。")
        ])
        
        # 注入 format_instructions
        formatted_prompt = prompt.format_messages(
            intent_json=intent_json,
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        messages = formatted_prompt
        
        try:
            # ReAct 循环
            max_iterations = 5
            iteration = 0
            
            last_tool_result = ""

            while iteration < max_iterations:
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
                
                # 检查是否有工具调用
                if not response.tool_calls:
                    # 如果 LLM 没有调用工具，说明它已经准备好生成最终回答了
                    # 此时 response.content 应该是符合 OutputStrategy 格式的 JSON
                    self.logger.info("Agent 循环结束，解析最终输出...")
                    try:
                        # 尝试解析 JSON 为 Pydantic 对象
                        final_output = self.output_parser.parse(response.content)
                        return final_output
                    except Exception as parse_err:
                        self.logger.error(f"解析最终输出失败: {parse_err}. Content: {response.content}")
                        # Fallback: 返回一个包含错误信息的简单对象或重试 (这里简化处理)
                        return OutputStrategy(
                            recommended_actions=["Error parsing agent output"], 
                            policy_type="SmPolicyDecision",
                            policy_details={}
                        )

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    self.logger.info(f"LLM 决定调用工具: {tool_name} 参数: {tool_args}")
                    
                    # 执行工具
                    if tool_name in self.tool_map:
                        tool_instance = self.tool_map[tool_name]
                        try:
                            # 运行 tool
                            tool_result = tool_instance.invoke(tool_args)
                            last_tool_result = str(tool_result) # 保存最近一次结果
                        except Exception as e:
                            tool_result = f"工具执行异常: {e}"
                    else:
                        tool_result = f"Error: Tool {tool_name} not found."
                    
                    # 将工具结果添加回消息历史
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                
                iteration += 1
            
            self.logger.warning("达到最大迭代次数。")
            return OutputStrategy(
                recommended_actions=["Max iterations reached"], 
                policy_type="SmPolicyDecision",
                policy_details={}
            )

        except Exception as e:
            self.logger.error(f"策略生成出错: {e}")
            return OutputStrategy(
                recommended_actions=[f"Agent execution error: {e}"], 
                policy_type="SmPolicyDecision",
                policy_details={}
            )
