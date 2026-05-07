from __future__ import annotations

import json
from typing import Any, Dict, List

from agent_runtime.messages import format_tool_call, format_tool_result, normalize_message_role, stringify_message_content
from agent_runtime.trace.models import RunTreeTraceRecord, iter_runs_in_dotted_order
from training.schemas import DatasetMessage, ProjectedTraceRecord, TrajectoryMessage


def _root_metadata(root_run: RunTreeTraceRecord) -> Dict[str, Any]:
    return dict(root_run.metadata or {})


def _root_input_messages(root_run: RunTreeTraceRecord) -> List[Dict[str, Any]]:
    raw = root_run.inputs.get("input_messages") or []
    return list(raw) if isinstance(raw, list) else []


def _tool_call_record(serialized_message: Dict[str, Any], call: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call.get("id"),
        "name": call.get("name"),
        "args": call.get("args"),
        "content": serialized_message.get("content"),
        "message_id": serialized_message.get("id"),
    }


def _tool_result_record(serialized_message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tool_call_id": serialized_message.get("tool_call_id"),
        "name": serialized_message.get("name"),
        "content": serialized_message.get("content"),
        "status": serialized_message.get("status"),
        "message_id": serialized_message.get("id"),
    }


def _append_message(
    trajectory: List[TrajectoryMessage],
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


def project_trace_to_training_trace(root_run: RunTreeTraceRecord) -> ProjectedTraceRecord:
    metadata = _root_metadata(root_run)
    input_messages = _root_input_messages(root_run)
    trajectory: List[TrajectoryMessage] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    structured_response: Any = root_run.outputs.get("structured_response")

    system_prompt = str(metadata.get("system_prompt") or "").strip()
    if system_prompt:
        _append_message(trajectory, "system", system_prompt)

    for message in input_messages:
        _append_message(
            trajectory,
            normalize_message_role(message, default="user"),
            stringify_message_content(message.get("content")),
        )

    for run in iter_runs_in_dotted_order(root_run):
        if run.run_type == "llm":
            serialized_message = run.outputs.get("message")
            if not isinstance(serialized_message, dict):
                continue
            role = normalize_message_role(serialized_message)
            content = stringify_message_content(serialized_message.get("content"))
            if role == "assistant" and content.strip():
                _append_message(trajectory, "assistant", content)
            for call in serialized_message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                record = _tool_call_record(serialized_message, call)
                tool_calls.append(record)
                _append_message(
                    trajectory,
                    "assistant",
                    format_tool_call({"name": record.get("name"), "args": record.get("args")}),
                    tool_call_id=str(record.get("id") or "") or None,
                    tool_name=str(record.get("name") or "") or None,
                )
        elif run.run_type == "tool":
            serialized_result = run.outputs.get("result")
            if not isinstance(serialized_result, dict):
                continue
            record = _tool_result_record(serialized_result)
            tool_results.append(record)
            _append_message(
                trajectory,
                "tool",
                format_tool_result({"content": record.get("content")}),
                tool_call_id=str(record.get("tool_call_id") or "") or None,
                tool_name=str(record.get("name") or "") or None,
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
        scenario_tags=[str(item).strip() for item in (metadata.get("scenario_tags") or []) if str(item).strip()],
        path_label=str(metadata.get("path_label") or "").strip() or None,
        advisor_decision=metadata.get("advisor_decision"),
        compiler_output=metadata.get("compiler_output"),
    )


def project_trace_to_chatml_messages(root_run: RunTreeTraceRecord) -> List[DatasetMessage]:
    projected = project_trace_to_training_trace(root_run)
    return [DatasetMessage(role=message.role, content=message.content) for message in projected.message_trajectory]


__all__ = ["project_trace_to_chatml_messages", "project_trace_to_training_trace"]
