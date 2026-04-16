from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage

from agent_runtime.execution.structured_tool_loop import ToolLoopExecutionError
from agent_runtime.messages import json_friendly, serialize_message
from agent_runtime.trace.models import RunTreeEvent, RunTreeTraceRecord, collect_descendant_ids

TRACE_UNSET = object()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_run_tree_record(
    *,
    agent_name: str,
    model_name: str,
    system_prompt: str,
    run_id: str,
    payload: Dict[str, Any],
    context: Any,
    result: Any,
    status: str,
    captured_error: BaseException | None,
    start_dt: datetime,
    end_dt: datetime,
    structured_response_override: Any = TRACE_UNSET,
) -> Dict[str, Any]:
    output_messages = _extract_output_messages(result, captured_error)
    structured_response = _extract_structured_response(
        result,
        captured_error,
        structured_response_override=structured_response_override,
    )
    trace_metadata = _extract_trace_metadata(payload)
    root_metadata = {
        "agent_name": agent_name,
        "model_name": model_name,
        "session_id": getattr(context, "session_id", "") if context is not None else "",
        "snapshot_id": getattr(context, "snapshot_id", "") if context is not None else "",
        "thread_id": getattr(context, "thread_id", "") if context is not None else "",
        "system_prompt": system_prompt,
        **trace_metadata,
    }
    child_runs = _build_child_runs(
        agent_name=agent_name,
        trace_id=run_id,
        parent_run_id=run_id,
        output_messages=output_messages,
        structured_response=structured_response,
        tool_error_call=_tool_error_payload(captured_error),
        error_text=None if captured_error is None else str(captured_error),
        start_dt=start_dt,
    )
    root_run = RunTreeTraceRecord(
        id=run_id,
        trace_id=run_id,
        parent_run_id=None,
        parent_run_ids=[],
        name=agent_name,
        run_type="chain",
        inputs={
            "input_messages": [serialize_message(message) for message in payload["messages"]],
            "invoke_payload": json_friendly({key: value for key, value in payload.items() if key != "messages"}),
        },
        outputs={} if structured_response is None else {"structured_response": structured_response},
        error=None if captured_error is None else str(captured_error),
        start_time=start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        events=[
            RunTreeEvent(name="invoke_started", time=start_dt.isoformat(), payload={}),
            RunTreeEvent(name="invoke_finished", time=end_dt.isoformat(), payload={"status": status}),
        ],
        tags=[agent_name, status],
        metadata=root_metadata,
        status="success" if status == "success" else "error",
        child_runs=child_runs,
        child_run_ids=[],
        direct_child_run_ids=[child.id for child in child_runs],
        dotted_order="1",
    )
    root_run.child_run_ids = collect_descendant_ids(root_run)
    return root_run.model_dump(mode="json")


def _extract_trace_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("trace_metadata")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("trace_metadata must be a mapping when present")
    return {
        "scenario_id": str(raw.get("scenario_id") or "").strip() or None,
        "scenario_tags": [str(item).strip() for item in (raw.get("scenario_tags") or []) if str(item).strip()],
        "path_label": str(raw.get("path_label") or "").strip() or None,
        "advisor_decision": json_friendly(raw.get("advisor_decision")),
        "compiler_output": json_friendly(raw.get("compiler_output")),
    }


def _extract_output_messages(result: Any, error: BaseException | None) -> List[Any]:
    if isinstance(error, ToolLoopExecutionError):
        return list(error.output_messages)
    if result is None:
        return []
    if not isinstance(result, dict):
        raise TypeError("traced structured agent result must be a dict")
    messages = result.get("messages", [])
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise TypeError("traced structured agent result field 'messages' must be a list")
    return messages


def _extract_structured_response(
    result: Any,
    error: BaseException | None,
    *,
    structured_response_override: Any = TRACE_UNSET,
) -> Any:
    if structured_response_override is not TRACE_UNSET:
        return json_friendly(structured_response_override)
    if isinstance(error, ToolLoopExecutionError) and error.structured_response is not None:
        return json_friendly(error.structured_response)
    if result is None:
        return None
    if not isinstance(result, dict):
        raise TypeError("traced structured agent result must be a dict")
    return json_friendly(result.get("structured_response"))


def _tool_error_payload(error: BaseException | None) -> Dict[str, Any] | None:
    if not isinstance(error, ToolLoopExecutionError):
        return None
    payload = error.failed_tool_call
    return dict(payload) if isinstance(payload, dict) else None


def _shifted_time(base: datetime, offset_ms: int) -> str:
    return (base + timedelta(milliseconds=offset_ms)).isoformat()


def _build_child_runs(
    *,
    agent_name: str,
    trace_id: str,
    parent_run_id: str,
    output_messages: Iterable[Any],
    structured_response: Any,
    tool_error_call: Dict[str, Any] | None,
    error_text: str | None,
    start_dt: datetime,
) -> List[RunTreeTraceRecord]:
    serialized_messages = [
        serialize_message(message) if isinstance(message, BaseMessage) else json_friendly(message)
        for message in output_messages
    ]
    llm_runs: List[RunTreeTraceRecord] = []
    tool_results_by_call_id = {
        str(message.get("tool_call_id") or "").strip(): message
        for message in serialized_messages
        if isinstance(message, dict) and str(message.get("type") or message.get("role") or "").strip().lower() == "tool"
    }

    child_index = 0
    for message in output_messages:
        if not isinstance(message, AIMessage):
            continue
        child_index += 1
        llm_run_id = f"run-{uuid4()}"
        llm_serialized = serialize_message(message)
        llm_run = RunTreeTraceRecord(
            id=llm_run_id,
            trace_id=trace_id,
            parent_run_id=parent_run_id,
            parent_run_ids=[parent_run_id],
            name=f"{agent_name}.llm",
            run_type="llm",
            inputs={},
            outputs={"message": llm_serialized},
            error=None,
            start_time=_shifted_time(start_dt, child_index),
            end_time=_shifted_time(start_dt, child_index + 1),
            events=[],
            tags=["llm"],
            metadata={"message_id": llm_serialized.get("id")},
            status="success",
            child_runs=[],
            child_run_ids=[],
            direct_child_run_ids=[],
            dotted_order=f"1.{child_index}",
        )
        for tool_index, call in enumerate(llm_serialized.get("tool_calls") or [], start=1):
            if not isinstance(call, dict):
                continue
            tool_call_id = str(call.get("id") or "").strip()
            tool_result = tool_results_by_call_id.get(tool_call_id)
            tool_status = "success"
            tool_error = None
            if tool_result is None and tool_error_call and str(tool_error_call.get("id") or "").strip() == tool_call_id:
                tool_status = "error"
                tool_error = error_text
            elif tool_result is None:
                tool_status = "error"
                tool_error = "missing tool result"
            llm_run.child_runs.append(
                RunTreeTraceRecord(
                    id=f"run-{uuid4()}",
                    trace_id=trace_id,
                    parent_run_id=llm_run_id,
                    parent_run_ids=[parent_run_id, llm_run_id],
                    name=str(call.get("name") or "tool").strip() or "tool",
                    run_type="tool",
                    inputs={
                        "tool_call_id": tool_call_id,
                        "name": call.get("name"),
                        "args": json_friendly(call.get("args")),
                    },
                    outputs={} if tool_result is None else {"result": tool_result},
                    error=tool_error,
                    start_time=_shifted_time(start_dt, child_index * 10 + tool_index),
                    end_time=_shifted_time(start_dt, child_index * 10 + tool_index + 1),
                    events=[],
                    tags=["tool"],
                    metadata={},
                    status=tool_status,
                    child_runs=[],
                    child_run_ids=[],
                    direct_child_run_ids=[],
                    dotted_order=f"1.{child_index}.{tool_index}",
                )
            )
        llm_run.direct_child_run_ids = [child.id for child in llm_run.child_runs]
        llm_run.child_run_ids = collect_descendant_ids(llm_run)
        llm_runs.append(llm_run)

    if structured_response is not None:
        parser_index = len(llm_runs) + 1
        llm_runs.append(
            RunTreeTraceRecord(
                id=f"run-{uuid4()}",
                trace_id=trace_id,
                parent_run_id=parent_run_id,
                parent_run_ids=[parent_run_id],
                name=f"{agent_name}.parser",
                run_type="parser",
                inputs={},
                outputs={"structured_response": structured_response},
                error=None,
                start_time=_shifted_time(start_dt, parser_index * 10),
                end_time=_shifted_time(start_dt, parser_index * 10 + 1),
                events=[],
                tags=["parser"],
                metadata={},
                status="success",
                child_runs=[],
                child_run_ids=[],
                direct_child_run_ids=[],
                dotted_order=f"1.{parser_index}",
            )
        )

    return llm_runs


__all__ = ["TRACE_UNSET", "build_run_tree_record", "utc_now"]
