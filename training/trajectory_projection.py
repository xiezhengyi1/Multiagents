from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agent_runtime.trace.models import RunTreeTraceRecord
from agent_runtime.trace.projectors import project_trace_to_training_trace
from training.common import load_trace_records
from training.schemas import ChatmlSftRecord, DatasetMessage, ProjectedTraceRecord, WorkflowTrajectoryRecord

COLLABORATION_AGENT_NAMES: tuple[str, ...] = (
    "intent_encoding",
    "optimization_strategy",
    "policy_dispatch",
)
WORKFLOW_AGENT_NAMES: tuple[str, ...] = ("main_control", *COLLABORATION_AGENT_NAMES)


def read_jsonl_objects(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(payload)
    return rows


def load_projected_trace_records_for_agent(project_root: Path, agent_name: str) -> List[ProjectedTraceRecord]:
    trace_path = project_root / "training" / agent_name / "raw_traces" / f"{agent_name}.jsonl"
    run_trees = load_trace_records(trace_path, RunTreeTraceRecord)
    return [project_trace_to_training_trace(run) for run in run_trees]


def load_projected_traces_by_agent(
    project_root: Path,
    *,
    agent_names: Sequence[str],
) -> Dict[str, List[ProjectedTraceRecord]]:
    traces: Dict[str, List[ProjectedTraceRecord]] = {}
    for agent_name in agent_names:
        normalized = str(agent_name or "").strip()
        if not normalized:
            raise ValueError("agent_names must not contain empty values")
        traces[normalized] = load_projected_trace_records_for_agent(project_root, normalized)
    return traces


def index_projected_traces_by_session(
    traces_by_agent: Mapping[str, Sequence[ProjectedTraceRecord]],
) -> Dict[str, Dict[str, List[ProjectedTraceRecord]]]:
    by_session: Dict[str, Dict[str, List[ProjectedTraceRecord]]] = {}
    for agent_name, traces in traces_by_agent.items():
        for trace in traces:
            session_id = str(trace.session_id or "").strip()
            if not session_id:
                raise ValueError(f"Trace {trace.trace_id} for agent {agent_name} is missing session_id")
            by_session.setdefault(session_id, {}).setdefault(str(agent_name), []).append(trace)
    return by_session


def sort_traces(traces: Iterable[ProjectedTraceRecord]) -> List[ProjectedTraceRecord]:
    return sorted(
        list(traces),
        key=lambda trace: (
            str(trace.timestamp or ""),
            str(trace.trace_id or ""),
        ),
    )


def flatten_tool_calls(
    agent_traces: Mapping[str, Sequence[ProjectedTraceRecord]],
    *,
    agent_order: Sequence[str],
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for agent_name in agent_order:
        for trace in sort_traces(agent_traces.get(agent_name, [])):
            for call in trace.tool_calls:
                flattened.append({"agent_name": agent_name, **dict(call)})
    return flattened


def flatten_tool_results(
    agent_traces: Mapping[str, Sequence[ProjectedTraceRecord]],
    *,
    agent_order: Sequence[str],
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for agent_name in agent_order:
        for trace in sort_traces(agent_traces.get(agent_name, [])):
            for result in trace.tool_results:
                flattened.append({"agent_name": agent_name, **dict(result)})
    return flattened


def build_workflow_trajectory_record(
    record: Mapping[str, Any],
    *,
    session_traces: Mapping[str, Sequence[ProjectedTraceRecord]],
    agent_order: Sequence[str] = WORKFLOW_AGENT_NAMES,
) -> WorkflowTrajectoryRecord:
    session_id = str(record.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("workflow record is missing session_id")

    snapshot_id = str(record.get("snapshot_id") or "").strip()
    scenario_tags = [str(item).strip() for item in (record.get("scenario_tags") or []) if str(item).strip()]
    agent_trajectories = {
        agent_name: sort_traces(session_traces.get(agent_name, []))
        for agent_name in agent_order
        if session_traces.get(agent_name)
    }
    workflow_id = str(record.get("workflow_id") or "").strip() or f"workflow-{session_id}"

    return WorkflowTrajectoryRecord(
        workflow_id=workflow_id,
        record_index=int(record["record_index"]) if record.get("record_index") is not None else None,
        scenario_id=str(record.get("scenario_id") or "").strip() or None,
        scenario_tags=scenario_tags,
        session_id=session_id,
        snapshot_id=snapshot_id,
        status=str(record.get("status") or "").strip() or "unknown",
        completed=bool(record.get("completed")),
        user_input=str(record.get("user_input") or ""),
        messages=list(record.get("messages") or []),
        context=str(record.get("context") or ""),
        global_intent=record.get("global_intent"),
        unified_plan=record.get("unified_plan"),
        qos_feedback=record.get("qos_feedback"),
        mobility_feedback=record.get("mobility_feedback"),
        diagnosis=record.get("diagnosis"),
        round_count=int(record.get("round_count") or 0),
        retry_count=int(record.get("retry_count") or 0),
        round_traces=list(record.get("round_traces") or []),
        agent_trajectories=agent_trajectories,
        tool_calls=flatten_tool_calls(agent_trajectories, agent_order=agent_order),
        tool_results=flatten_tool_results(agent_trajectories, agent_order=agent_order),
    )


def build_chatml_record_from_trace(trace: ProjectedTraceRecord) -> ChatmlSftRecord:
    if not trace.message_trajectory:
        raise ValueError(f"Trace {trace.trace_id} has empty message_trajectory")

    messages = [
        DatasetMessage(role=message.role, content=message.content)
        for message in trace.message_trajectory
    ]
    metadata = {
        "trace_id": trace.trace_id,
        "session_id": trace.session_id,
        "snapshot_id": trace.snapshot_id,
        "thread_id": trace.thread_id,
        "status": trace.status,
        "scenario_id": trace.scenario_id,
        "scenario_tags": trace.scenario_tags,
        "path_label": trace.path_label,
        "tool_call_count": len(trace.tool_calls),
        "tool_result_count": len(trace.tool_results),
        "model_name": trace.model_name,
    }
    sample_id = f"{trace.agent_name}-chatml-{trace.trace_id}"
    task = f"{trace.agent_name}_chatml_trace"
    return ChatmlSftRecord(
        sample_id=sample_id,
        task=task,
        agent=trace.agent_name,
        messages=messages,
        metadata=metadata,
    )


__all__ = [
    "COLLABORATION_AGENT_NAMES",
    "WORKFLOW_AGENT_NAMES",
    "build_chatml_record_from_trace",
    "build_workflow_trajectory_record",
    "flatten_tool_calls",
    "flatten_tool_results",
    "index_projected_traces_by_session",
    "load_projected_trace_records_for_agent",
    "load_projected_traces_by_agent",
    "read_jsonl_objects",
    "sort_traces",
]
