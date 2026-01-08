from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .basemodel import BaseAgent
from utils.logger import setup_logger

logger = setup_logger(name="IntentEncodingAgent")

# 定义输出结构
class FlowIntent(BaseModel):
    flow_id: str = Field(description="业务流ID，例如 f_41")
    business_type: str = Field(description="业务类型，例如 URLLC, eMBB, mMTC")
    bandwidth_demand: float = Field(description="带宽需求(Mbps)")
    latency_requirement: float = Field(description="时延要求(ms)")
    priority_level: int = Field(description="优先级，数字越小优先级越高，或者数字越大优先级越高(需根据系统定义，此处假设数字越小优先级越高)")
    description: str = Field(description="业务流的简要描述")

class UserIntent(BaseModel):
    app_name: str = Field(description="应用名称")
    flows: List[FlowIntent] = Field(description="该应用包含的业务流列表")
    urgency: str = Field(description="整体紧急程度，如：Normal, High, Critical")
    raw_intent_summary: str = Field(description="对用户原始意图的总结")

class IntentEncodingAgent(BaseAgent):
    def __init__(self, model_name="qwen-plus"):
        super().__init__(model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=UserIntent)

    def analyze_intent(self, user_input: str, context: str = "") -> UserIntent:
        """
        分析用户输入并转化为结构化意图
        """
        
        prompt_template = """
        你是一个5G网络切片系统的意图识别Agent (Intent Encoding Agent)。
        你的任务是根据用户的自然语言描述，分析出具体的网络切片需求。
        
        请仔细分析用户的输入，提取出应用名称、包含的业务流及其具体的QoS需求（带宽、时延、业务类型、优先级）。
        
        参考背景信息：
        - URLLC: 超高可靠低时延通信，适用于远程控制、自动驾驶等。
        - eMBB: 增强型移动宽带，适用于高清视频、大数据传输等。
        - mMTC: 海量机器类通信，适用于传感器网络等。
        - 优先级：通常1-5为高优先级（关键业务），6-10为中等，11+为低优先级。
        
        用户输入:
        {user_input}
        
        额外上下文(如有):
        {context}
        
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_input", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({"user_input": user_input, "context": context})
            logger.info(f"LLM 意图分析结果: {result}")
            return result
        except Exception as e:
            logger.error(f"意图解析出错: {e}")
            return None
