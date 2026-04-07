from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


class MinimalTraceRecord(BaseModel):
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BuildReport.model_validate(payload)


def save_build_report(path: Path, report: BuildReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
