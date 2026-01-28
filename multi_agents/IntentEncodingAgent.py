from typing import List, Optional, Dict, Any
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .basemodel import BaseAgent
from utils.logger import setup_logger
from .Prompt import IEA_SYSTEM_PROMPT
from tools.pcf_tools import get_ue_context


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
        self.tools = [get_ue_context]

    def analyze_intent(self, user_input: str, context: str = "") -> UserIntent:
        """
        分析用户输入并转化为结构化意图
        """
        
        prompt_template = IEA_SYSTEM_PROMPT

        supi = self._extract_supi(user_input)
        ue_context = ""
        if supi:
            try:
                ue_context = get_ue_context(supi)
            except Exception as e:
                self.logger.warning(f"UE Context 查询失败: {e}")

        merged_context = context
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_input", "context", "ue_context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({"user_input": user_input, "context": merged_context, "ue_context": ue_context})
            self.logger.info(f"LLM 意图分析结果: {result}")
            return result
        except Exception as e:
            self.logger.error(f"意图解析出错: {e}")
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
