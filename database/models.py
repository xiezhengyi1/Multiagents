import uuid
from datetime import UTC, datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Integer, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from .connection import Base


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)

class SessionContext(Base):
    """
    表 A: 任务会话表 (Short-term memory)
    用于存储“进行中”的闭环任务。充当 Agent 间的“接力棒”。
    """
    __tablename__ = "session_context"

    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    current_step = Column(String)  # 兼容旧字段
    current_stage = Column(String, nullable=True)
    current_snapshot_id = Column(String, nullable=True)
    current_artifact_id = Column(String, nullable=True)
    round_index = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    intent_data = Column(JSONB, nullable=True)  # 用户原始意图数据
    policy_data = Column(JSONB, nullable=True)  # 生成的策略数据
    status = Column(String, default="active")  # active, completed, failed
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class AgentTask(Base):
    __tablename__ = "agent_task"

    task_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    artifact_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True, index=True)
    snapshot_id = Column(String, nullable=True, index=True)
    correlation_id = Column(String, nullable=True, index=True)
    source_agent = Column(String, nullable=False)
    target_agent = Column(String, nullable=False, index=True)
    artifact_type = Column(String, nullable=False)
    status = Column(String, default="queued", index=True)
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint("artifact_id", "target_agent", name="uq_agent_task_artifact_target"),
    )


class AgentArtifact(Base):
    __tablename__ = "agent_artifact"

    artifact_id = Column(String, primary_key=True)
    correlation_id = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    snapshot_id = Column(String, nullable=True, index=True)
    source_agent = Column(String, nullable=False)
    target_agent = Column(String, nullable=False, index=True)
    artifact_type = Column(String, nullable=False)
    kind = Column(String, nullable=False)  # request or response
    path = Column(Text, nullable=False)
    payload_summary = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class AgentHandoffRecord(Base):
    __tablename__ = "agent_handoff"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=True, index=True)
    snapshot_id = Column(String, nullable=True, index=True)
    round_index = Column(Integer, default=0)
    source_agent = Column(String, nullable=False)
    target_agent = Column(String, nullable=False)
    artifact_id = Column(String, nullable=True, index=True)
    artifact_type = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    handoff_payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class SessionStageResult(Base):
    __tablename__ = "session_stage_result"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    snapshot_id = Column(String, nullable=True, index=True)
    round_index = Column(Integer, default=0)
    stage_name = Column(String, nullable=False, index=True)
    artifact_id = Column(String, nullable=True, index=True)
    status = Column(String, nullable=False)
    payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

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
    created_at = Column(DateTime, default=_utcnow)

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

    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

class NetworkStatusSnapshot(Base):
    """
    表 D: 网络状态快照表 (Monitor History)
    用于存储网络切片和节点在特定时间点的性能指标历史记录。
    """
    __tablename__ = "network_status_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utcnow, index=True)

    # 关键步骤: 拆分存储为切片、APP、节点三列，便于独立查询与演进
    slice_data = Column(JSONB, nullable=False, default=list)
    app_data = Column(JSONB, nullable=False, default=list)
    node_data = Column(JSONB, nullable=False, default=list)
    mobility_data = Column(JSONB, nullable=False, default=list)
    policy_data = Column(JSONB, nullable=False, default=dict)

    # 触发快照的原因，例如 "PeriodicMonitor", "Pre-Optimization", "Post-Optimization"
    trigger_event = Column(String, nullable=True)


class NetworkGraphSnapshot(Base):
    __tablename__ = "network_graph_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String, unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    base_network_snapshot_id = Column(String, nullable=True, index=True)
    trigger_event = Column(String, nullable=True)
    graph_summary = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=_utcnow, index=True)


class GraphNode(Base):
    __tablename__ = "graph_node"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String, ForeignKey("network_graph_snapshot.snapshot_id"), nullable=False, index=True)
    node_key = Column(String, nullable=False)
    node_type = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True)
    properties = Column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "node_key", name="uq_graph_node_snapshot_key"),
    )


class GraphEdge(Base):
    __tablename__ = "graph_edge"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String, ForeignKey("network_graph_snapshot.snapshot_id"), nullable=False, index=True)
    edge_key = Column(String, nullable=False)
    edge_type = Column(String, nullable=False, index=True)
    source_key = Column(String, nullable=False, index=True)
    target_key = Column(String, nullable=False, index=True)
    properties = Column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "edge_key", name="uq_graph_edge_snapshot_key"),
    )


class GraphMetric(Base):
    __tablename__ = "graph_metric"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String, ForeignKey("network_graph_snapshot.snapshot_id"), nullable=False, index=True)
    owner_type = Column(String, nullable=False)  # node or edge
    owner_key = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(JSONB, nullable=True)
    observed_at = Column(DateTime, default=_utcnow, index=True)

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
    ursp_rules = Column(JSONB, nullable=True)
    app_catalog = Column(JSONB, nullable=True)
    flow_catalog = Column(JSONB, nullable=True)
    access_mobility_context = Column(JSONB, nullable=True)
    am_policy_context = Column(JSONB, nullable=True)
    serving_nf_context = Column(JSONB, nullable=True)
    mobility_summary = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class UeAmPolicyAssociationRecord(Base):
    __tablename__ = "ue_am_policy_assoc"

    id = Column(Integer, primary_key=True, autoincrement=True)
    supi = Column(String, ForeignKey("ue_context.supi", ondelete="CASCADE"), nullable=False, index=True)
    pol_asso_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True, index=True)
    snapshot_id = Column(String, nullable=True, index=True)
    round_index = Column(Integer, default=0)
    association_request = Column(JSONB, nullable=False)
    association_policy = Column(JSONB, nullable=False)
    status = Column(String, nullable=False, default="draft", index=True)
    trigger_event = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=_utcnow, index=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint("supi", "pol_asso_id", name="uq_ue_am_policy_assoc_supi_pol_asso_id"),
    )


class UeMobilityEventRecord(Base):
    __tablename__ = "ue_mobility_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    supi = Column(String, ForeignKey("ue_context.supi", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String, nullable=True, index=True)
    snapshot_id = Column(String, nullable=True, index=True)
    event_type = Column(String, nullable=False, index=True)
    event_summary = Column(Text, nullable=True)
    event_payload = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime, default=_utcnow, index=True)


class UeServingNfBindingRecord(Base):
    __tablename__ = "ue_serving_nf_binding"

    id = Column(Integer, primary_key=True, autoincrement=True)
    supi = Column(String, ForeignKey("ue_context.supi", ondelete="CASCADE"), nullable=False, index=True)
    nf_type = Column(String, nullable=False, index=True)
    nf_instance_id = Column(String, nullable=True, index=True)
    nf_uri = Column(String, nullable=True)
    binding_info = Column(JSONB, nullable=False, default=dict)
    status = Column(String, nullable=False, default="active", index=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("supi", "nf_type", name="uq_ue_serving_nf_binding_supi_nf_type"),
    )
