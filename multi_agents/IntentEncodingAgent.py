from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .basemodel import BaseAgent


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
    app_name: str = Field(description="应用名称")
    app_id: str = Field(description="应用ID")
    operation_type: str = Field(description="操作类型，可选值: add, modify, delete")
    flows: List[FlowIntent] = Field(description="该应用包含的业务流列表")
    urgency: str = Field(description="整体紧急程度，如：Normal, High, Critical")
    raw_intent_summary: str = Field(description="对用户原始意图的总结")

# 定义输入结构
from model.UeContext import UeSmPolicyData

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

        请仔细分析用户的输入和上下文，提取出应用名称、操作类型（新增/更改/删除）、包含的业务流及其具体的QoS需求（带宽、时延、业务类型、优先级）。

        参考背景信息：
        - URLLC: 超高可靠低时延通信，适用于远程控制、自动驾驶等。
        - eMBB: 增强型移动宽带，适用于高清视频、大数据传输等。
        - mMTC: 海量机器类通信，适用于传感器网络等。
        - 优先级：通常1-5为高优先级（关键业务），6-10为中等，11+为低优先级。

        上下文字段含义：
        subsDefQos (基准权益)：用户归属地的默认 QoS 配置（优先级 arp、类别 5qi），意图未指定时继承此值。
        vplmnQos (漫游上限)：用户漫游时的 QoS 硬性限制（带宽 maxFbr、总量 sessionAmbr），决策时不可突破此上限。
        5qi：5G QoS 指标，数值越低优先级越高，常见映射如下：
            - 1-4: 语音、视频通话等实时业务
            - 5-9: 视频流、在线游戏等高带宽业务
            - 10-15: 普通数据业务、背景下载等低优先级业务
        请根据以上信息，生成符合 UserIntent 结构的输出。

        用户输入:
        {user_input}
        
        用户实际需求和订阅数据:
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
            self.logger.info(f"LLM 意图分析结果: {result}")
            return result
        except Exception as e:
            self.logger.error(f"意图解析出错: {e}")
            return None
