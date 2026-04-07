from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, get_args, get_origin
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from sft_data.common import raw_trace_dir
from sft_data.formatting import format_tool_call, format_tool_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FILE_LOCKS: dict[Path, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()
_TRACE_UNSET = object()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_lock(path: Path) -> threading.Lock:
    with _LOCK_GUARD:
        lock = _FILE_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[path] = lock
        return lock


def _json_friendly(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type) and issubclass(value, BaseModel):
        return _serialize_base_model_schema(value)
    if isinstance(value, BaseModel):
        return _json_friendly(value.model_dump(mode="json"))
    if hasattr(value, "model_dump"):
        return _json_friendly(value.model_dump(mode="json"))
    if isinstance(value, BaseMessage):
        return _serialize_message(value)
    if isinstance(value, Mapping):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_friendly(item) for item in value]
    return str(value)


def _serialize_message(message: BaseMessage | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(message, Mapping):
        return {str(key): _json_friendly(value) for key, value in message.items()}

    payload: Dict[str, Any] = {
        "type": message.type,
        "content": _json_friendly(message.content),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
        "additional_kwargs": _json_friendly(getattr(message, "additional_kwargs", {})),
        "response_metadata": _json_friendly(getattr(message, "response_metadata", {})),
    }

    if isinstance(message, AIMessage):
        payload["tool_calls"] = _json_friendly(getattr(message, "tool_calls", []))
        payload["invalid_tool_calls"] = _json_friendly(getattr(message, "invalid_tool_calls", []))
        payload["usage_metadata"] = _json_friendly(getattr(message, "usage_metadata", None))

    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = getattr(message, "tool_call_id", None)
        payload["status"] = getattr(message, "status", None)

    return payload


def _annotation_to_string(annotation: Any) -> str:
    if annotation is None:
        return "None"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = ", ".join(_annotation_to_string(item) for item in get_args(annotation))
    origin_name = getattr(origin, "__name__", str(origin))
    return f"{origin_name}[{args}]"


def _serialize_base_model_schema(model_cls: type[BaseModel]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for field_name, field_info in model_cls.model_fields.items():
        if field_name == "runtime":
            continue
        field_payload: Dict[str, Any] = {
            "annotation": _annotation_to_string(field_info.annotation),
        }
        if field_info.description:
            field_payload["description"] = str(field_info.description)
        if field_info.is_required():
            required.append(field_name)
        elif field_info.default is not None:
            field_payload["default"] = _json_friendly(field_info.default)
        properties[field_name] = field_payload
    return {
        "title": model_cls.__name__,
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_tool_specs(tools: Iterable[Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name and hasattr(tool, "__name__"):
            name = tool.__name__
        description = getattr(tool, "description", None) or getattr(tool, "__doc__", None) or ""
        args_schema = getattr(tool, "args_schema", None)
        specs.append(
            {
                "name": str(name or ""),
                "description": str(description).strip(),
                "args_schema": _json_friendly(args_schema),
            }
        )
    return specs


def extract_tool_calls(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in messages:
        serialized = _serialize_message(item) if isinstance(item, BaseMessage) else _json_friendly(item)
        if not isinstance(serialized, dict):
            continue
        raw_calls = serialized.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise TypeError("tool_calls must be a list when present in traced messages")
        for call in raw_calls:
            if not isinstance(call, dict):
                raise TypeError("each tool_call must be a dict")
            calls.append(
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": _json_friendly(call.get("args")),
                    "content": serialized.get("content"),
                    "message_id": serialized.get("id"),
                }
            )
    return calls


def extract_tool_results(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in messages:
        serialized = _serialize_message(item) if isinstance(item, BaseMessage) else _json_friendly(item)
        if not isinstance(serialized, dict):
            continue
        message_type = str(serialized.get("type") or serialized.get("role") or "")
        if message_type != "tool":
            continue
        results.append(
            {
                "tool_call_id": serialized.get("tool_call_id"),
                "name": serialized.get("name"),
                "content": serialized.get("content"),
                "status": serialized.get("status"),
                "message_id": serialized.get("id"),
            }
        )
    return results


def _normalize_message_role(serialized: Dict[str, Any]) -> str:
    raw_role = str(serialized.get("role") or serialized.get("type") or "").strip().lower()
    if raw_role == "human":
        return "user"
    if raw_role == "ai":
        return "assistant"
    if raw_role:
        return raw_role
    return "assistant"


def _stringify_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _extract_think_text(call: Dict[str, Any]) -> str:
    args = call.get("args")
    if not isinstance(args, dict):
        return ""
    message = args.get("message")
    if message is None:
        return ""
    return str(message)


def build_message_trajectory(
    system_prompt: str,
    input_messages: Iterable[Any],
    output_messages: Iterable[Any],
    *,
    structured_response: Any,
) -> List[Dict[str, Any]]:
    trajectory: List[Dict[str, Any]] = []

    def append(role: str, content: str, *, tool_call_id: str | None = None, tool_name: str | None = None) -> None:
        trajectory.append(
            {
                "role": role,
                "content": content,
                "step_index": len(trajectory),
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
            }
        )

    if str(system_prompt or "").strip():
        append("system", str(system_prompt))

    for message in input_messages:
        serialized = _serialize_message(message) if isinstance(message, BaseMessage) else _json_friendly(message)
        if not isinstance(serialized, dict):
            continue
        append(
            _normalize_message_role(serialized),
            _stringify_message_content(serialized.get("content")),
        )

    for message in output_messages:
        serialized = _serialize_message(message) if isinstance(message, BaseMessage) else _json_friendly(message)
        if not isinstance(serialized, dict):
            continue

        role = _normalize_message_role(serialized)
        content = _stringify_message_content(serialized.get("content"))
        tool_calls = serialized.get("tool_calls", [])

        if role == "assistant" and content.strip():
            append(role, content)

        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_name = str(call.get("name") or "").strip()
                if tool_name == "think":
                    think_text = _extract_think_text(call).strip()
                    if think_text:
                        append(
                            "assistant",
                            f"<think>{think_text}</think>",
                            tool_call_id=str(call.get("id") or "") or None,
                            tool_name=tool_name,
                        )
                    continue
                append(
                    "assistant",
                    format_tool_call(
                        {
                            "name": tool_name,
                            "args": call.get("args"),
                        }
                    ),
                    tool_call_id=str(call.get("id") or "") or None,
                    tool_name=tool_name or None,
                )
            continue

        if role == "tool":
            tool_name = str(serialized.get("name") or "") or None
            if tool_name == "think":
                continue
            append(
                "tool",
                format_tool_result(
                    {
                        "content": serialized.get("content"),
                    }
                ),
                tool_call_id=str(serialized.get("tool_call_id") or "") or None,
                tool_name=tool_name,
            )
            continue

        if role != "assistant" and content.strip():
            append(role, content)

    if structured_response is not None:
        final_content = json.dumps(_json_friendly(structured_response), ensure_ascii=False)
        if not trajectory or trajectory[-1]["role"] != "assistant" or trajectory[-1]["content"] != final_content:
            append("assistant", final_content)

    return trajectory


class JsonlTraceWriter:
    def __init__(self, agent_name: str, root: Path | None = None) -> None:
        normalized_agent = str(agent_name or "").strip()
        if not normalized_agent:
            raise ValueError("agent_name is required for trace writing")
        self.agent_name = normalized_agent
        self.root = Path(root) if root is not None else raw_trace_dir(normalized_agent)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{self.agent_name}.jsonl"

    def write(self, record: Dict[str, Any]) -> Path:
        payload = json.dumps(record, ensure_ascii=False)
        lock = _file_lock(self.path)
        with lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
        return self.path


class TracedStructuredAgent:
    def __init__(
        self,
        agent_name: str,
        model_name: str,
        system_prompt: str,
        tool_specs: List[Dict[str, Any]],
        runnable: Any,
        writer: JsonlTraceWriter,
    ) -> None:
        self.agent_name = str(agent_name or "").strip()
        self.model_name = str(model_name or "").strip()
        self.system_prompt = str(system_prompt or "")
        self.tool_specs = list(tool_specs)
        self.runnable = runnable
        self.writer = writer

    def __getattr__(self, item: str) -> Any:
        return getattr(self.runnable, item)

    def invoke(self, payload: Dict[str, Any], *, context: Any = None, **kwargs: Any) -> Any:
        if "messages" not in payload:
            raise KeyError("traced agent invoke payload must contain 'messages'")
        trace_id = f"trace-{uuid4()}"
        status = "success"
        error = None
        result = None
        try:
            result = self.runnable.invoke(payload, context=context, **kwargs)
            return result
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            self.writer.write(
                self._build_trace_record(
                    trace_id=trace_id,
                    payload=payload,
                    context=context,
                    result=result,
                    status=status,
                    error=error,
                )
            )

    async def ainvoke(self, payload: Dict[str, Any], *, context: Any = None, **kwargs: Any) -> Any:
        if "messages" not in payload:
            raise KeyError("traced agent ainvoke payload must contain 'messages'")
        trace_id = f"trace-{uuid4()}"
        status = "success"
        error = None
        result = None
        try:
            result = await self.runnable.ainvoke(payload, context=context, **kwargs)
            return result
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            self.writer.write(
                self._build_trace_record(
                    trace_id=trace_id,
                    payload=payload,
                    context=context,
                    result=result,
                    status=status,
                    error=error,
                )
            )

    def write_trace(
        self,
        *,
        payload: Dict[str, Any],
        context: Any = None,
        result: Any = None,
        status: str = "success",
        error: str | None = None,
        structured_response_override: Any = _TRACE_UNSET,
    ) -> Path:
        trace_id = f"trace-{uuid4()}"
        return self.writer.write(
            self._build_trace_record(
                trace_id=trace_id,
                payload=payload,
                context=context,
                result=result,
                status=status,
                error=error,
                structured_response_override=structured_response_override,
            )
        )

    def _build_trace_record(
        self,
        *,
        trace_id: str,
        payload: Dict[str, Any],
        context: Any,
        result: Any,
        status: str,
        error: str | None,
        structured_response_override: Any = _TRACE_UNSET,
    ) -> Dict[str, Any]:
        output_messages = self._extract_output_messages(result)
        record = {
            "trace_id": trace_id,
            "timestamp": _utc_now_iso(),
            "agent_name": self.agent_name,
            "session_id": getattr(context, "session_id", "") if context is not None else "",
            "snapshot_id": getattr(context, "snapshot_id", "") if context is not None else "",
            "thread_id": getattr(context, "thread_id", "") if context is not None else "",
            "model_name": self.model_name,
            "input_messages": [_serialize_message(message) for message in payload["messages"]],
            "message_trajectory": build_message_trajectory(
                self.system_prompt,
                payload["messages"],
                output_messages,
                structured_response=self._extract_structured_response(
                    result,
                    structured_response_override=structured_response_override,
                ),
            ),
            "tool_calls": extract_tool_calls(output_messages),
            "tool_results": extract_tool_results(output_messages),
            "structured_response": self._extract_structured_response(
                result,
                structured_response_override=structured_response_override,
            ),
            "status": status,
            "error": error,
        }
        required = (
            "trace_id",
            "timestamp",
            "agent_name",
            "session_id",
            "snapshot_id",
            "thread_id",
            "model_name",
            "input_messages",
            "message_trajectory",
            "tool_calls",
            "tool_results",
            "structured_response",
            "status",
            "error",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise RuntimeError(f"trace record missing required fields: {missing}")
        return record

    @staticmethod
    def _extract_output_messages(result: Any) -> List[Any]:
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

    @staticmethod
    def _extract_structured_response(result: Any, *, structured_response_override: Any = _TRACE_UNSET) -> Any:
        if structured_response_override is not _TRACE_UNSET:
            return _json_friendly(structured_response_override)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise TypeError("traced structured agent result must be a dict")
        return _json_friendly(result.get("structured_response"))
