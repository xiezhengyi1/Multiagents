from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, get_args, get_origin
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = PROJECT_ROOT / "sft_data" / "intent_encoding" / "raw_traces"

_FILE_LOCKS: dict[Path, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


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


class JsonlTraceWriter:
    def __init__(self, agent_name: str, root: Path | None = None) -> None:
        normalized_agent = str(agent_name or "").strip()
        if not normalized_agent:
            raise ValueError("agent_name is required for trace writing")
        self.agent_name = normalized_agent
        self.root = Path(root) if root is not None else TRACE_ROOT
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
        started_at = time.perf_counter()
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
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    status=status,
                    error=error,
                )
            )

    async def ainvoke(self, payload: Dict[str, Any], *, context: Any = None, **kwargs: Any) -> Any:
        if "messages" not in payload:
            raise KeyError("traced agent ainvoke payload must contain 'messages'")
        trace_id = f"trace-{uuid4()}"
        started_at = time.perf_counter()
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
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    status=status,
                    error=error,
                )
            )

    def _build_trace_record(
        self,
        *,
        trace_id: str,
        payload: Dict[str, Any],
        context: Any,
        result: Any,
        latency_ms: float,
        status: str,
        error: str | None,
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
            "system_prompt": self.system_prompt,
            "input_messages": [_serialize_message(message) for message in payload["messages"]],
            "tool_specs": self.tool_specs,
            "tool_calls": extract_tool_calls(output_messages),
            "tool_results": extract_tool_results(output_messages),
            "output_messages": [_serialize_message(message) for message in output_messages],
            "structured_response": self._extract_structured_response(result),
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "context": _json_friendly(context),
        }
        required = (
            "trace_id",
            "timestamp",
            "agent_name",
            "session_id",
            "snapshot_id",
            "thread_id",
            "model_name",
            "system_prompt",
            "input_messages",
            "tool_specs",
            "tool_calls",
            "tool_results",
            "structured_response",
            "latency_ms",
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
    def _extract_structured_response(result: Any) -> Any:
        if result is None:
            return None
        if not isinstance(result, dict):
            raise TypeError("traced structured agent result must be a dict")
        return _json_friendly(result.get("structured_response"))
