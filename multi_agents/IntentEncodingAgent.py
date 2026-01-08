from typing import List, Optional, Dict, Any
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
    priority_level: int = Field(description="优先级，数字越小优先级越高")
    
    # 关联应用会话
    app_session_id: Optional[str] = Field(None, description="关联的应用会话ID，对应 active_app_session_ids")
    
    description: str = Field(description="业务流的简要描述")

# 定义输入结构
class UserIntent(BaseModel):
    app_name: str = Field(description="应用名称")
    flows: List[FlowIntent] = Field(description="该应用包含的业务流列表")
    urgency: str = Field(description="整体紧急程度，如：Normal, High, Critical")
    raw_intent_summary: str = Field(description="对用户原始意图的总结")


class SmPolicyContextDecisionData(BaseModel):
    # --- 1. 标识与业务 ---
    supi: str
    pduSessionId: int
    sliceInfo: Dict[str, Any]  # SNSSAI: { sst, sd }
    dnn: str
    pduSessionType: str        # IPV4, IPV6...

    # --- 2. 接入与位置 ---
    accessType: Optional[str] = None
    ratType: Optional[str] = None      # NR, EUTRA...
    servingNetwork: Optional[Dict[str, str]] = None # PLMN ID
    userLocationInfo: Optional[Dict[str, Any]] = None
    
    # --- 3. 资源限制 ---
    subsSessAmbr: Optional[Dict[str, str]] = None # { uplink, downlink }
    subsDefQos: Optional[Dict[str, Any]] = None   # { 5qi, arp }
    
    # --- 4. 计费与状态 ---
    online: bool = False
    offline: bool = False
    is_ps_data_off: bool = False # 对应 3gppPsDataOffStatus
    
    # --- 5. 动态感知 ---
    nwdafDatas: Optional[List[Dict[str, Any]]] = None

class OpenApiSubscriptionData(BaseModel):
    """占位符：对应 models.SmPolicyData (来自 UDR 的签约信息)"""
    # 包含字段如: var3gppDnnSnssaiMbs, smPolicySnssaiData 等
    content: Dict[str, Any] = Field(default_factory=dict)

class AmContextInfo(BaseModel):
    """提取自 UeAMPolicyData 的移动性与接入信息"""
    user_location: Optional[Dict[str, Any]] = Field(None, description="对应 models.UserLocation")
    serving_plmn: Optional[Dict[str, Any]] = Field(None, description="对应 models.PlmnIdNid")
    access_type: Optional[str] = Field(None, description="对应 models.AccessType")
    rat_type: Optional[str] = None

class SmSessionState(BaseModel):
    """提取自 UeSmPolicyData: 单个 PDU 会话的决策依据"""
    
    # 1. 基础上下文 (The 'Check' & 'Fetch' phase)
    policy_context: SmPolicyContextDecisionData = Field(..., description="SMF 提供的会话上下文，包含当前切片、DNN、IP等")
    subscription_data: OpenApiSubscriptionData = Field(..., description="UDR 提供的用户策略签约数据")

    # 2. 动态资源状态 (The 'Constraint' phase)
    # 对应 RemainGbrUL/DL
    remain_gbr_ul: Optional[float] = Field(None, description="剩余上行 GBR 带宽")
    remain_gbr_dl: Optional[float] = Field(None, description="剩余下行 GBR 带宽")

    # 3. 应用层需求 (The 'Dynamic Rule' phase)
    # 对应 AppSessions map[string]bool
    active_app_session_ids: List[str] = Field(default_factory=list, description="关联的活动应用会话ID列表 (AF请求)")
    
    # 4. 流量路由影响
    # 对应 InfluenceDataToPccRule
    traffic_influence_data: Dict[str, str] = Field(default_factory=dict, description="流量影响数据到 PCC 规则的映射")

class UeSmDecisionContext(BaseModel):
    """虽然 PCF 是做 SM 决策，但通常也是基于整个 UE 的上下文"""
    supi: str
    
    # 全局/移动性信息 (来自 UeAMPolicyData)
    # 某些 QoS 策略可能基于位置 (Location-Based QoS)
    am_info: Optional[AmContextInfo] = None
    
    # 目标 PDU 会话信息
    # PCF 每次决策通常针对特定的一个 PDU Session
    target_session: SmSessionState


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
        
        用户订阅数据:
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
