from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.runtime.messages import (
    format_tool_call,
    format_tool_result,
    normalize_message_role,
    stringify_message_content,
)
from shared.runtime.trace.models import RunTreeTraceRecord, iter_runs_in_dotted_order

from training.artifacts import raw_trace_path, read_jsonl
from training.schemas import (
    ChatmlSftRecord,
    DatasetMessage,
    FineTuningTraceRecord,
    ProjectedTraceRecord,
    TrajectoryMessage,
    WorkflowTrajectoryRecord,
)


COLLABORATION_AGENT_NAMES = (
    "intent_encoding",
    "optimization_strategy",
    "policy_dispatch",
)
WORKFLOW_AGENT_NAMES = ("main_control", *COLLABORATION_AGENT_NAMES)


def _append_message(
    trajectory: list[TrajectoryMessage],
    role: str,
    content: str,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> None:
    trajectory.append(
        TrajectoryMessage(
            role=role,
            content=content,
            step_index=len(trajectory),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
    )


def project_trace(root_run: RunTreeTraceRecord) -> ProjectedTraceRecord:
    metadata = dict(root_run.metadata or {})
    raw_messages = root_run.inputs.get("input_messages") or []
    input_messages = list(raw_messages) if isinstance(raw_messages, list) else []
    trajectory: list[TrajectoryMessage] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    structured_response: Any = root_run.outputs.get("structured_response")

    if system_prompt := str(metadata.get("system_prompt") or "").strip():
        _append_message(trajectory, "system", system_prompt)
    for message in input_messages:
        _append_message(
            trajectory,
            normalize_message_role(message, default="user"),
            stringify_message_content(message.get("content")),
        )

    for run in iter_runs_in_dotted_order(root_run):
        if run.run_type == "llm":
            message = run.outputs.get("message")
            if not isinstance(message, dict):
                continue
            role = normalize_message_role(message)
            content = stringify_message_content(message.get("content"))
            if role == "assistant" and content.strip():
                _append_message(trajectory, role, content)
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                record = {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args"),
                    "content": message.get("content"),
                    "message_id": message.get("id"),
                }
                tool_calls.append(record)
                _append_message(
                    trajectory,
                    "assistant",
                    format_tool_call({"name": record["name"], "args": record["args"]}),
                    tool_call_id=str(record["id"] or "") or None,
                    tool_name=str(record["name"] or "") or None,
                )
        elif run.run_type == "tool":
            result = run.outputs.get("result")
            if not isinstance(result, dict):
                continue
            record = {
                "tool_call_id": result.get("tool_call_id"),
                "name": result.get("name"),
                "content": result.get("content"),
                "status": result.get("status"),
                "message_id": result.get("id"),
            }
            tool_results.append(record)
            _append_message(
                trajectory,
                "tool",
                format_tool_result({"content": record["content"]}),
                tool_call_id=str(record["tool_call_id"] or "") or None,
                tool_name=str(record["name"] or "") or None,
            )
        elif run.run_type == "parser":
            structured_response = run.outputs.get("structured_response")

    if structured_response is not None:
        final_content = json.dumps(structured_response, ensure_ascii=False)
        if not trajectory or trajectory[-1].role != "assistant" or trajectory[-1].content != final_content:
            _append_message(trajectory, "assistant", final_content)

    return ProjectedTraceRecord(
        trace_id=root_run.trace_id,
        timestamp=root_run.start_time,
        agent_name=str(metadata.get("agent_name") or root_run.name or "").strip(),
        session_id=str(metadata.get("session_id") or "").strip(),
        snapshot_id=str(metadata.get("snapshot_id") or "").strip(),
        thread_id=str(metadata.get("thread_id") or "").strip(),
        model_name=str(metadata.get("model_name") or "").strip(),
        input_messages=input_messages,
        message_trajectory=trajectory,
        tool_calls=tool_calls,
        tool_results=tool_results,
        structured_response=structured_response,
        status=root_run.status,
        error=root_run.error,
        scenario_id=str(metadata.get("scenario_id") or "").strip() or None,
        scenario_tags=[str(item).strip() for item in metadata.get("scenario_tags") or [] if str(item).strip()],
        path_label=str(metadata.get("path_label") or "").strip() or None,
        advisor_decision=metadata.get("advisor_decision"),
        compiler_output=metadata.get("compiler_output"),
    )


def load_projected_trace_records(path: Path) -> list[ProjectedTraceRecord]:
    if not path.exists():
        return []
    return [project_trace(RunTreeTraceRecord.model_validate(row)) for row in read_jsonl(path)]


def load_agent_traces(agent_name: str) -> list[ProjectedTraceRecord]:
    return load_projected_trace_records(raw_trace_path(agent_name))


def load_traces(agent_names: Sequence[str]) -> dict[str, list[ProjectedTraceRecord]]:
    return {agent_name: load_agent_traces(agent_name) for agent_name in agent_names}


def index_traces_by_session(
    traces_by_agent: Mapping[str, Sequence[ProjectedTraceRecord]],
) -> dict[str, dict[str, list[ProjectedTraceRecord]]]:
    by_session: dict[str, dict[str, list[ProjectedTraceRecord]]] = {}
    for agent_name, traces in traces_by_agent.items():
        for trace in traces:
            if not trace.session_id:
                raise ValueError(f"Trace {trace.trace_id} for agent {agent_name} is missing session_id")
            by_session.setdefault(trace.session_id, {}).setdefault(agent_name, []).append(trace)
    return by_session


def sort_traces(traces: Iterable[ProjectedTraceRecord]) -> list[ProjectedTraceRecord]:
    return sorted(traces, key=lambda trace: (trace.timestamp, trace.trace_id))


def _flatten_trace_field(
    agent_traces: Mapping[str, Sequence[ProjectedTraceRecord]],
    agent_order: Sequence[str],
    field: str,
) -> list[dict[str, Any]]:
    return [
        {"agent_name": agent_name, **dict(item)}
        for agent_name in agent_order
        for trace in sort_traces(agent_traces.get(agent_name, []))
        for item in getattr(trace, field)
    ]


def build_workflow_record(
    record: Mapping[str, Any],
    *,
    session_traces: Mapping[str, Sequence[ProjectedTraceRecord]],
    agent_order: Sequence[str] = WORKFLOW_AGENT_NAMES,
) -> WorkflowTrajectoryRecord:
    session_id = str(record.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("workflow record is missing session_id")
    agent_trajectories = {
        agent_name: sort_traces(session_traces.get(agent_name, []))
        for agent_name in agent_order
        if session_traces.get(agent_name)
    }
    return WorkflowTrajectoryRecord(
        workflow_id=str(record.get("workflow_id") or "").strip() or f"workflow-{session_id}",
        record_index=int(record["record_index"]) if record.get("record_index") is not None else None,
        scenario_id=str(record.get("scenario_id") or "").strip() or None,
        scenario_tags=[str(item).strip() for item in record.get("scenario_tags") or [] if str(item).strip()],
        session_id=session_id,
        snapshot_id=str(record.get("snapshot_id") or "").strip(),
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
        tool_calls=_flatten_trace_field(agent_trajectories, agent_order, "tool_calls"),
        tool_results=_flatten_trace_field(agent_trajectories, agent_order, "tool_results"),
    )


def build_finetuning_record(trace: ProjectedTraceRecord) -> FineTuningTraceRecord:
    return FineTuningTraceRecord(
        trace_id=trace.trace_id,
        timestamp=trace.timestamp,
        agent_name=trace.agent_name,
        session_id=trace.session_id,
        snapshot_id=trace.snapshot_id,
        input=list(trace.input_messages),
        output=trace.structured_response,
        tool_calls=list(trace.tool_calls),
        tool_results=list(trace.tool_results),
        status=trace.status,
        error=trace.error,
    )


def build_chatml_record(trace: FineTuningTraceRecord) -> ChatmlSftRecord:
    messages = [
        DatasetMessage(role=str(message.get("role") or "user"), content=str(message.get("content") or ""))
        for message in trace.input
        if isinstance(message, Mapping) and str(message.get("content") or "").strip()
    ]
    messages.extend(
        DatasetMessage(role="assistant", content=json.dumps({"tool_call": call}, ensure_ascii=False))
        for call in trace.tool_calls
    )
    messages.extend(
        DatasetMessage(role="tool", content=json.dumps(result, ensure_ascii=False))
        for result in trace.tool_results
    )
    messages.append(DatasetMessage(role="assistant", content=json.dumps(trace.output, ensure_ascii=False)))
    return ChatmlSftRecord(
        sample_id=f"{trace.agent_name}-chatml-{trace.trace_id}",
        task=f"{trace.agent_name}_chatml_trace",
        agent=trace.agent_name,
        messages=messages,
        metadata={
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "snapshot_id": trace.snapshot_id,
            "status": trace.status,
            "tool_call_count": len(trace.tool_calls),
            "tool_result_count": len(trace.tool_results),
        },
    )
