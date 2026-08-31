from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.contracts import AgentMessage, AgentTrace
from shared.runtime import extract_tool_calls, extract_tool_results

from training.artifacts import raw_trace_path, read_jsonl
from training.schemas import (
    ProjectedTraceRecord,
    Qwen3SftRecord,
    SftExampleV2,
    WorkflowTrajectoryRecord,
)


COLLABORATION_AGENT_NAMES = (
    "intent_encoding",
    "optimization_strategy",
    "policy_dispatch",
)
WORKFLOW_AGENT_NAMES = ("main_control", *COLLABORATION_AGENT_NAMES)


def load_raw_traces(path: Path) -> list[AgentTrace]:
    return [AgentTrace.model_validate(row) for row in read_jsonl(path)]


def project_trace(trace: AgentTrace) -> ProjectedTraceRecord:
    observed_targets = [target for target in trace.targets if target.kind == "observed_model_output"]
    return ProjectedTraceRecord(
        trace_id=trace.trace_id,
        timestamp=trace.started_at.isoformat(),
        agent_name=trace.agent_name,
        session_id=trace.scope.session_id,
        snapshot_id=trace.scope.snapshot_id or "",
        thread_id=trace.scope.thread_id,
        model_name=trace.model_name,
        messages=trace.messages,
        tool_calls=extract_tool_calls(trace.messages),
        tool_results=extract_tool_results(trace.messages),
        structured_response=observed_targets[0].value if observed_targets else None,
        status=trace.status,
        error=trace.error,
        scenario_id=trace.scope.scenario_id,
        scenario_tags=trace.scope.scenario_tags,
        path_label=trace.metadata.path_label,
        quality_targets=[
            target
            for target in trace.targets
            if target.kind in {"validated_output", "compiled_output"}
        ],
    )


def load_projected_trace_records(path: Path) -> list[ProjectedTraceRecord]:
    return [project_trace(trace) for trace in load_raw_traces(path)]


def load_agent_traces(agent_name: str) -> list[AgentTrace]:
    return load_raw_traces(raw_trace_path(agent_name))


def load_traces(agent_names: Sequence[str]) -> dict[str, list[AgentTrace]]:
    return {agent_name: load_agent_traces(agent_name) for agent_name in agent_names}


def is_sft_candidate(trace: AgentTrace) -> bool:
    return (
        trace.status == "success"
        and trace.source == "observed_llm"
        and len([target for target in trace.targets if target.kind == "observed_model_output"]) == 1
        and any(target.kind in {"validated_output", "compiled_output"} for target in trace.targets)
    )


def build_sft_example(trace: AgentTrace) -> SftExampleV2:
    if not is_sft_candidate(trace):
        raise ValueError(f"trace {trace.trace_id} has not passed the SFT quality gate")
    observed_target = next(
        target for target in trace.targets if target.kind == "observed_model_output"
    )
    quality_targets = [
        target
        for target in trace.targets
        if target.kind in {"validated_output", "compiled_output"}
    ]
    return SftExampleV2(
        sample_id=f"{trace.agent_name}-{trace.trace_id}",
        trace_id=trace.trace_id,
        agent_name=trace.agent_name,
        scope=trace.scope,
        messages=trace.messages,
        tool_catalog=trace.tool_catalog,
        response_schema_name=trace.response_schema_name,
        response_schema=trace.response_schema,
        observed_target=observed_target,
        quality_targets=quality_targets,
        metadata=trace.metadata,
    )


def _qwen3_message(message: AgentMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        payload["name"] = message.name
        payload["tool_call_id"] = message.tool_call_id
    return payload


def build_qwen3_record(example: SftExampleV2) -> Qwen3SftRecord:
    return Qwen3SftRecord(
        sample_id=example.sample_id,
        messages=[_qwen3_message(message) for message in example.messages],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in example.tool_catalog.tools
        ],
        metadata={
            "trace_id": example.trace_id,
            "agent_name": example.agent_name,
            "session_id": example.scope.session_id,
            "snapshot_id": example.scope.snapshot_id,
            "path_label": example.metadata.path_label,
            "prompt_name": example.metadata.prompt_name,
            "prompt_version": example.metadata.prompt_version,
            "tool_catalog_fingerprint": example.tool_catalog.fingerprint,
            "response_schema_name": example.response_schema_name,
        },
    )


def index_traces_by_session(
    traces_by_agent: Mapping[str, Sequence[AgentTrace]],
) -> dict[str, dict[str, list[ProjectedTraceRecord]]]:
    by_session: dict[str, dict[str, list[ProjectedTraceRecord]]] = {}
    for agent_name, traces in traces_by_agent.items():
        for trace in traces:
            projected = project_trace(trace)
            by_session.setdefault(trace.scope.session_id, {}).setdefault(agent_name, []).append(projected)
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
        workflow_id=str(record.get("workflow_id") or f"workflow-{session_id}"),
        record_index=int(record["record_index"]) if record.get("record_index") is not None else None,
        scenario_id=str(record.get("scenario_id") or "").strip() or None,
        scenario_tags=[str(item) for item in record.get("scenario_tags") or []],
        session_id=session_id,
        snapshot_id=str(record.get("snapshot_id") or ""),
        status=str(record.get("status") or "unknown"),
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


__all__ = [
    "COLLABORATION_AGENT_NAMES",
    "WORKFLOW_AGENT_NAMES",
    "build_qwen3_record",
    "build_sft_example",
    "build_workflow_record",
    "index_traces_by_session",
    "is_sft_candidate",
    "load_agent_traces",
    "load_projected_trace_records",
    "load_raw_traces",
    "load_traces",
    "project_trace",
    "sort_traces",
]
