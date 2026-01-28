import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Handle the case where pgvector is not installed yet to avoid import errors during setup
    Vector = None

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
    
    # 意图向量: 您的 Embedding 模型输出为 1024 维 (如 bge-large-zh 或 阿里某些模型)
    if Vector:
        intent_vector = Column(Vector(1024))
    else:
        # Fallback if library missing
        intent_vector = Column(String) 

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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
