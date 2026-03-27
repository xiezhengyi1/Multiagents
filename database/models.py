import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from .connection import Base

class SessionContext(Base):
    """
    表 A: 任务会话表 (Short-term memory)
    用于存储“进行中”的闭环任务。充当 Agent 间的“接力棒”。
    """
    __tablename__ = "session_context"

    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    current_step = Column(String)  # 枚举: intent, generation, execution
    intent_data = Column(JSONB, nullable=True)  # 用户原始意图数据
    policy_data = Column(JSONB, nullable=True)  # 生成的策略数据
    status = Column(String, default="active")  # active, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EpisodicExperience(Base):
    """
    表 B: 经验回溯表 (Long-term memory)
    存储已完成的闭环数据，供 Agent 2 在生成策略时参考。
    """
    __tablename__ = "episodic_experience"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 意图向量: 您的 Embedding 模型输出为 1024 维 (如 text-embedding-v4)
    intent_vector = Column(Vector(1024))

    raw_intent = Column(Text)
    applied_policy = Column(JSONB)
    environment_state = Column(JSONB)  # 执行时的资源情况
    feedback_metrics = Column(JSONB)   # 成功率/延迟变化等
    reward_score = Column(Float)       # 评价该策略的好坏
    created_at = Column(DateTime, default=datetime.utcnow)

class SemanticKnowledge(Base):
    """
    表 C: 基础配置表 (Static Knowledge)
    存储静态的领域知识，如 5G 切片定义、映射表等。
    """
    __tablename__ = "semantic_knowledge"

    key = Column(String, primary_key=True)
    category = Column(String, nullable=True, index=True)  # e.g., "SmPolicyDecision", "UEProfile"
    value = Column(JSONB, nullable=False)
    description = Column(Text, nullable=True)

    # 向量字段
    embedding = Column(Vector(1024))

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NetworkStatusSnapshot(Base):
    """
    表 D: 网络状态快照表 (Monitor History)
    用于存储网络切片和节点在特定时间点的性能指标历史记录。
    """
    __tablename__ = "network_status_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # 关键步骤: 拆分存储为切片、APP、节点三列，便于独立查询与演进
    slice_data = Column(JSONB, nullable=False, default=list)
    app_data = Column(JSONB, nullable=False, default=list)
    node_data = Column(JSONB, nullable=False, default=list)

    # 触发快照的原因，例如 "PeriodicMonitor", "Pre-Optimization", "Post-Optimization"
    trigger_event = Column(String, nullable=True)

class UeContextRecord(Base):
    """
    表 E: UE 上下文表
    对齐 UeContext/SmPolicyDecision，仅保留关键策略字段。
    """
    __tablename__ = "ue_context"

    supi = Column(String, primary_key=True)

    # 关键步骤: 按 smPolicyId 维度保存会话策略（与 UeContext.smPolicyData 对齐）
    sm_policy_data = Column(JSONB, nullable=True)

    # 关键步骤: 提取常用决策字段，便于检索/下游消费
    pcc_rules = Column(JSONB, nullable=True)
    qos_decs = Column(JSONB, nullable=True)
    sess_rules = Column(JSONB, nullable=True)
    traff_cont_decs = Column(JSONB, nullable=True)
    chg_decs = Column(JSONB, nullable=True)
    app_catalog = Column(JSONB, nullable=True)
    flow_catalog = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
