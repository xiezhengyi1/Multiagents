from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetMessage(BaseModel):
    role: str
    content: str


class TrajectoryMessage(BaseModel):
    role: str = Field(description="ChatML role. ChatML traces must start with system and end with assistant.")
    content: str = Field(description="Rendered message content. Tool turns store only the raw tool result body inside <tool_result>.")
    step_index: int
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


class ChatmlSftRecord(BaseModel):
    sample_id: str
    task: str
    agent: str
    messages: List[DatasetMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectedTraceRecord(BaseModel):
    trace_id: str
    timestamp: str
    agent_name: str
    session_id: str
    snapshot_id: str
    thread_id: str
    model_name: str
    input_messages: List[Dict[str, Any]]
    message_trajectory: List[TrajectoryMessage] = Field(default_factory=list, description="Rendered ChatML-style trajectory for SFT export.")
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    structured_response: Any
    status: str
    error: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario_tags: List[str] = Field(default_factory=list)
    path_label: Optional[str] = None
    advisor_decision: Any = None
    compiler_output: Any = None


class FineTuningTraceRecord(BaseModel):
    """The minimal per-agent trajectory used as a fine-tuning example.

    The runtime run tree is intentionally richer and remains in ``raw_traces``
    for debugging.  This record is the stable, compact artifact emitted by an
    experiment for model training.
    """

    trace_id: str
    timestamp: str
    agent_name: str
    session_id: str
    snapshot_id: str
    input: List[Dict[str, Any]] = Field(default_factory=list)
    output: Any = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    status: str
    error: Optional[str] = None


class WorkflowTrajectoryRecord(BaseModel):
    workflow_id: str
    record_index: Optional[int] = None
    scenario_id: Optional[str] = None
    scenario_tags: List[str] = Field(default_factory=list)
    session_id: str
    snapshot_id: str
    status: str
    completed: bool
    user_input: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: str = ""
    global_intent: Any = None
    unified_plan: Any = None
    qos_feedback: Any = None
    mobility_feedback: Any = None
    diagnosis: Any = None
    round_count: int = 0
    retry_count: int = 0
    round_traces: List[Dict[str, Any]] = Field(default_factory=list)
    agent_trajectories: Dict[str, List[ProjectedTraceRecord]] = Field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
