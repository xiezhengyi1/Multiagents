from typing import List, Optional, Dict, Any
import re
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from .basemodel import BaseAgent
from utils.logger import setup_logger
from .Prompt import IEA_SYSTEM_PROMPT
from tools.pcf_tools import get_ue_context
from tools.knowledge_tool import search_semantic_knowledge, get_knowledge_by_key



# 定义输出结构
class FlowIntent(BaseModel):
    name: str = Field(description="业务流名称")
    flow_id: str = Field(description="业务流ID")
    business_type: str = Field(description="业务类型，例如 URLLC, eMBB, mMTC")
    bw_ul: float = Field(description="上行带宽需求(Mbps)")
    bw_dl: float = Field(description="下行带宽需求(Mbps)")
    gbr_ul: float = Field(description="上行保证比特率需求(Mbps)")
    gbr_dl: float = Field(description="下行保证比特率需求(Mbps)")
    lat: float = Field(description="时延要求(ms)")
    loss_req: float = Field(description="丢包率要求(0~1)")
    jitter_req: float = Field(description="抖动要求(ms)")
    priority: int = Field(description="优先级，数字越小优先级越高")
    description: str = Field(description="业务流的简要描述")

class UserIntent(BaseModel):
    supi: Optional[str] = Field(description="用户唯一标识 SUPI，如 imsi-...")
    app_name: str = Field(description="应用名称")
    app_id: str = Field(description="应用ID")
    operation_type: str = Field(description="操作类型，可选值: add, modify, delete")
    flows: List[FlowIntent] = Field(description="该应用包含的业务流列表")
    urgency: str = Field(description="整体紧急程度，如：Normal, High, Critical")
    raw_intent_summary: str = Field(description="对用户原始意图的总结")

class IntentEncodingAgent(BaseAgent):
    def __init__(self, model_name="qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=UserIntent)
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[95m")  # 黄色日志
        self.tools = [get_ue_context, search_semantic_knowledge, get_knowledge_by_key]
        
        # 1. 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 2. 工具映射
        self.tool_map = {tool.name: tool for tool in self.tools}

    def analyze_intent(self, user_input: str, context: str = "") -> UserIntent:
        """
        分析用户输入并转化为结构化意图
        """
        
        # 1. 初始化变量 (UE Context 由 Agent 自行获取)
        ue_context = ""

        # 2. 构建 Prompt
        # 直接利用 ChatPromptTemplate 填充 IEA_SYSTEM_PROMPT 中的变量
        prompt = ChatPromptTemplate.from_messages([
            ("system", IEA_SYSTEM_PROMPT)
        ])
        
        formatted_messages = prompt.format_messages(
            user_input=user_input, 
            context=context, 
            ue_context=ue_context,
            format_instructions=self.parser.get_format_instructions()
        )
        
        messages = formatted_messages
        
        try:
            # 3. 手动执行 ReAct / ToolCall 循环
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
                
                # 如果没有工具调用，说明已经生成了最终回复(希望是JSON)
                if not response.tool_calls:
                    output_str = response.content.strip()
                    self.logger.info("LLM 回复内容，尝试解析为 JSON...")
                    
                    try:
                        # 简单的 Markdown JSON 清理
                        if "```json" in output_str:
                            output_str = output_str.split("```json")[1].split("```")[0]
                        elif "```" in output_str:
                            output_str = output_str.split("```")[1].split("```")[0]
                            
                        result = self.parser.parse(output_str)
                        self.logger.info(f"LLM 意图分析成功: {result.app_name}")
                        return result
                    except Exception as parse_err:
                        self.logger.error(f"结果解析失败: {parse_err}. 原始内容: {output_str}")
                        return None

                # 处理工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    self.logger.info(f"LLM 决定调用工具: {tool_name} 参数: {tool_args}")
                    
                    if tool_name in self.tool_map:
                        tool_instance = self.tool_map[tool_name]
                        try:
                            tool_result = tool_instance.invoke(tool_args)
                            self.logger.info(f"工具 {tool_name} 执行结果: {tool_result}")
                        except Exception as e:
                            tool_result = f"工具执行异常: {e}"
                    else:
                        tool_result = f"Error: Tool {tool_name} not found."
                    
                    # 将工具结果添加回消息历史
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                
                iteration += 1
            
            self.logger.warning("达到最大迭代次数，未生成有效 JSON。")
            return None

        except Exception as e:
            self.logger.error(f"意图解析整体流程出错: {e}")
            return None

    @staticmethod
    def _extract_supi(text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"(?i)\bimsi-\d{5,}\b", text)
        if match:
            return match.group(0)

        match = re.search(r"(?i)\bsupi\s*[:=]?\s*([\w-]+)", text)
        if match:
            return match.group(1)

        return None
