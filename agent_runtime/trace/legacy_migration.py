from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from agent_runtime.trace.models import RunTreeTraceRecord, collect_descendant_ids


def is_legacy_minimal_trace(payload: Mapping[str, Any]) -> bool:
    return "run_type" not in payload and "message_trajectory" in payload and "trace_id" in payload


def _json_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _matches_structured_response(content: str, structured_response: Any) -> bool:
    text = str(content or "").strip()
    if not text or structured_response is None:
        return False
    try:
        return json.loads(text) == structured_response
    except Exception:
        return False


def _tool_output_payload(result: Mapping[str, Any] | None, tool_call_id: str, tool_name: str) -> Dict[str, Any]:
    if result is None:
        return {}
    return {
        "result": {
            "type": "tool",
            "content": result.get("content"),
            "tool_call_id": tool_call_id,
            "name": result.get("name") or tool_name,
            "status": result.get("status"),
            "id": result.get("message_id") or f"{tool_call_id}-result",
        }
    }


def legacy_trace_to_run_tree(payload: Mapping[str, Any]) -> RunTreeTraceRecord:
    if not is_legacy_minimal_trace(payload):
        raise ValueError("payload is not a legacy minimal trace")

    trace_id = str(payload.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("legacy trace is missing trace_id")
    root_id = f"run-{trace_id}"
    timestamp = str(payload.get("timestamp") or "").strip()
    if not timestamp:
        raise ValueError(f"legacy trace {trace_id} is missing timestamp")
    agent_name = str(payload.get("agent_name") or "").strip()
    if not agent_name:
        raise ValueError(f"legacy trace {trace_id} is missing agent_name")

    trajectory = list(payload.get("message_trajectory") or [])
    tool_calls = list(payload.get("tool_calls") or [])
    tool_results = list(payload.get("tool_results") or [])
    structured_response = payload.get("structured_response")
    input_messages = list(payload.get("input_messages") or [])

    system_prompt = ""
    for message in trajectory:
        if str(message.get("role") or "").strip() == "system":
            system_prompt = _json_string(message.get("content"))
            break

    tool_calls_by_id = {
        str(call.get("id") or "").strip(): dict(call)
        for call in tool_calls
        if str(call.get("id") or "").strip()
    }
    tool_results_by_id = {
        str(result.get("tool_call_id") or "").strip(): dict(result)
        for result in tool_results
        if str(result.get("tool_call_id") or "").strip()
    }

    used_tool_calls: set[str] = set()
    used_tool_results: set[str] = set()
    child_runs: List[RunTreeTraceRecord] = []
    root_child_index = 1

    def append_llm_run(*, content: str, tool_call: Mapping[str, Any] | None = None) -> None:
        nonlocal root_child_index
        llm_run_id = f"{root_id}-llm-{root_child_index}"
        tool_children: List[RunTreeTraceRecord] = []
        llm_message: Dict[str, Any] = {
            "type": "ai",
            "content": "" if tool_call is not None else content,
            "id": f"msg-{root_child_index}",
            "tool_calls": [],
        }
        if tool_call is not None:
            tool_call_id = str(tool_call.get("id") or "").strip()
            tool_name = str(tool_call.get("name") or "").strip() or "tool"
            llm_message["tool_calls"] = [
                {
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": tool_call.get("args") or {},
                }
            ]
            used_tool_calls.add(tool_call_id)
            result = tool_results_by_id.get(tool_call_id)
            if result is not None:
                used_tool_results.add(tool_call_id)
            tool_children.append(
                RunTreeTraceRecord(
                    id=f"{llm_run_id}-tool",
                    trace_id=root_id,
                    parent_run_id=llm_run_id,
                    parent_run_ids=[root_id, llm_run_id],
                    name=tool_name,
                    run_type="tool",
                    inputs={
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "args": tool_call.get("args") or {},
                    },
                    outputs=_tool_output_payload(result, tool_call_id, tool_name),
                    error=None if result is not None else "missing tool result",
                    start_time=timestamp,
                    end_time=timestamp,
                    events=[],
                    tags=["tool"],
                    metadata={},
                    status="success" if result is not None else "error",
                    child_runs=[],
                    child_run_ids=[],
                    direct_child_run_ids=[],
                    dotted_order=f"1.{root_child_index}.1",
                )
            )
        llm_run = RunTreeTraceRecord(
            id=llm_run_id,
            trace_id=root_id,
            parent_run_id=root_id,
            parent_run_ids=[root_id],
            name=f"{agent_name}.llm",
            run_type="llm",
            inputs={},
            outputs={"message": llm_message},
            error=None,
            start_time=timestamp,
            end_time=timestamp,
            events=[],
            tags=["llm"],
            metadata={},
            status="success",
            child_runs=tool_children,
            child_run_ids=[],
            direct_child_run_ids=[child.id for child in tool_children],
            dotted_order=f"1.{root_child_index}",
        )
        llm_run.child_run_ids = collect_descendant_ids(llm_run)
        child_runs.append(llm_run)
        root_child_index += 1

    def append_orphan_tool(result: Mapping[str, Any]) -> None:
        nonlocal root_child_index
        tool_call_id = str(result.get("tool_call_id") or "").strip()
        tool_name = str(result.get("name") or "").strip() or "tool"
        used_tool_results.add(tool_call_id)
        child_runs.append(
            RunTreeTraceRecord(
                id=f"{root_id}-tool-{root_child_index}",
                trace_id=root_id,
                parent_run_id=root_id,
                parent_run_ids=[root_id],
                name=tool_name,
                run_type="tool",
                inputs={
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "args": (tool_calls_by_id.get(tool_call_id) or {}).get("args") or {},
                },
                outputs=_tool_output_payload(result, tool_call_id, tool_name),
                error=None,
                start_time=timestamp,
                end_time=timestamp,
                events=[],
                tags=["tool"],
                metadata={},
                status="success",
                child_runs=[],
                child_run_ids=[],
                direct_child_run_ids=[],
                dotted_order=f"1.{root_child_index}",
            )
        )
        root_child_index += 1

    for message in trajectory:
        role = str(message.get("role") or "").strip()
        content = _json_string(message.get("content"))
        if role == "assistant":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            tool_name = str(message.get("tool_name") or "").strip()
            if tool_call_id and tool_name:
                tool_call = dict(tool_calls_by_id.get(tool_call_id) or {})
                tool_call.setdefault("id", tool_call_id)
                tool_call.setdefault("name", tool_name)
                tool_call.setdefault("args", {})
                append_llm_run(content="", tool_call=tool_call)
            elif content.strip() and not _matches_structured_response(content, structured_response):
                append_llm_run(content=content)
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if tool_call_id and tool_call_id in used_tool_results:
                continue
            append_orphan_tool(
                {
                    "tool_call_id": tool_call_id,
                    "name": message.get("tool_name"),
                    "content": message.get("content"),
                    "status": "success",
                    "message_id": message.get("message_id"),
                }
            )

    for tool_call_id, tool_call in tool_calls_by_id.items():
        if tool_call_id in used_tool_calls:
            continue
        append_llm_run(content="", tool_call=tool_call)

    for tool_call_id, result in tool_results_by_id.items():
        if tool_call_id in used_tool_results:
            continue
        append_orphan_tool(result)

    if structured_response is not None:
        child_runs.append(
            RunTreeTraceRecord(
                id=f"{root_id}-parser",
                trace_id=root_id,
                parent_run_id=root_id,
                parent_run_ids=[root_id],
                name=f"{agent_name}.parser",
                run_type="parser",
                inputs={},
                outputs={"structured_response": structured_response},
                error=None,
                start_time=timestamp,
                end_time=timestamp,
                events=[],
                tags=["parser"],
                metadata={},
                status="success",
                child_runs=[],
                child_run_ids=[],
                direct_child_run_ids=[],
                dotted_order=f"1.{root_child_index}",
            )
        )

    root_run = RunTreeTraceRecord(
        id=root_id,
        trace_id=root_id,
        parent_run_id=None,
        parent_run_ids=[],
        name=agent_name,
        run_type="chain",
        inputs={
            "input_messages": input_messages,
            "invoke_payload": {},
        },
        outputs={} if structured_response is None else {"structured_response": structured_response},
        error=None if payload.get("error") in (None, "") else str(payload.get("error")),
        start_time=timestamp,
        end_time=timestamp,
        events=[],
        tags=[agent_name, str(payload.get("status") or "success")],
        metadata={
            "agent_name": agent_name,
            "model_name": payload.get("model_name"),
            "session_id": payload.get("session_id"),
            "snapshot_id": payload.get("snapshot_id"),
            "thread_id": payload.get("thread_id"),
            "system_prompt": system_prompt,
            "scenario_id": payload.get("scenario_id"),
            "scenario_tags": list(payload.get("scenario_tags") or []),
            "path_label": payload.get("path_label"),
            "advisor_decision": payload.get("advisor_decision"),
            "compiler_output": payload.get("compiler_output"),
        },
        status=str(payload.get("status") or "success"),
        child_runs=child_runs,
        child_run_ids=[],
        direct_child_run_ids=[child.id for child in child_runs],
        dotted_order="1",
    )
    root_run.child_run_ids = collect_descendant_ids(root_run)
    return root_run


__all__ = ["is_legacy_minimal_trace", "legacy_trace_to_run_tree"]
