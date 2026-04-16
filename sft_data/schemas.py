from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

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


class SupervisedSftRecord(BaseModel):
    sample_id: str
    task: str
    agent: str
    messages: List[DatasetMessage]
    target: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCallWarmupRecord(BaseModel):
    sample_id: str
    task: str = "tool_call_warmup"
    agent: str
    messages: List[DatasetMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatmlSftRecord(BaseModel):
    sample_id: str
    task: str
    agent: str
    messages: List[DatasetMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgenticRlTraceRecord(BaseModel):
    sample_id: str
    task: str = "agentic_rl_trace"
    agent: str
    observation: List[DatasetMessage]
    tool_trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: Any = None
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


class ScenarioSpecRecord(BaseModel):
    scenario_id: str
    seed: int
    domain_type: Literal["qos_only", "mobility_only", "joint"]
    difficulty: Literal["zero_ambiguity", "ambiguous", "incomplete_context", "conflict", "dispatch_fail", "partial_success", "tradeoff"]
    retry_shape: Literal["one_shot_success", "retry_qos", "retry_mobility", "retry_cross_domain", "stop_incomplete_context"]
    user_input: str
    expected_behavior: Dict[str, Any] = Field(default_factory=dict)
    mutations: List[str] = Field(default_factory=list)
    mock_env: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class MainIntentSftRecord(BaseModel):
    sample_id: str
    task: str = "main_intent_sft"
    agent: str = "main_control"
    input: Dict[str, Any]
    target: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IeAdvisorSftRecord(BaseModel):
    sample_id: str
    task: str = "iea_advisor_sft"
    agent: str = "intent_encoding"
    input: Dict[str, Any]
    target: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OsaDecisionSftRecord(BaseModel):
    sample_id: str
    task: str = "osa_decision_sft"
    agent: str = "optimization_strategy"
    input: Dict[str, Any]
    target: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompilerSftRecord(BaseModel):
    sample_id: str
    task: str
    agent: str
    input: Dict[str, Any]
    target: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BuildReport(BaseModel):
    artifact_total: int = 0
    paired_total: int = 0
    iea_sft_samples: int = 0
    osa_sft_samples: int = 0
    tool_warmup_samples: int = 0
    rl_trace_samples: int = 0
    reject_total: int = 0


def write_jsonl(path: Path, records: Iterable[BaseModel | Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def load_build_report(path: Path) -> BuildReport:
    if not path.exists():
        return BuildReport()
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return BuildReport()
    payload = json.loads(raw_text)
    return BuildReport.model_validate(payload)


def save_build_report(path: Path, report: BuildReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
