"""Canonical serialization, ChatML formatting, and message extraction utilities.

This module is the single source of truth for:
- JSON-safe recursive serialization (json_friendly, serialize_message, etc.)
- ChatML tag rendering (format_tool_call, format_tool_result)
- Message role normalization and content stringification
- Tool spec building and tool call / result extraction from message sequences

Other modules should import shared message formatting from here
rather than re-implementing these functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, get_args, get_origin

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel


# ── Generic JSON Serialization ─────────────────────────────────────


def json_friendly(value: Any) -> Any:
    """Recursively convert a value to JSON-serializable types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type) and issubclass(value, BaseModel):
        return serialize_base_model_schema(value)
    if isinstance(value, BaseModel):
        return json_friendly(value.model_dump(mode="json"))
    if hasattr(value, "model_dump"):
        return json_friendly(value.model_dump(mode="json"))
    if isinstance(value, BaseMessage):
        return serialize_message(value)
    if isinstance(value, Mapping):
        return {str(key): json_friendly(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_friendly(item) for item in value]
    return str(value)


# ── Message Serialization ─────────────────────────────────────


def serialize_message(message: BaseMessage | Mapping[str, Any]) -> Dict[str, Any]:
    """Serialize a LangChain message or raw dict to a JSON-friendly dict."""
    if isinstance(message, Mapping):
        return {str(key): json_friendly(value) for key, value in message.items()}

    payload: Dict[str, Any] = {
        "type": message.type,
        "content": json_friendly(message.content),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
        "additional_kwargs": json_friendly(getattr(message, "additional_kwargs", {})),
        "response_metadata": json_friendly(getattr(message, "response_metadata", {})),
    }

    if isinstance(message, AIMessage):
        payload["tool_calls"] = json_friendly(getattr(message, "tool_calls", []))
        payload["invalid_tool_calls"] = json_friendly(getattr(message, "invalid_tool_calls", []))
        payload["usage_metadata"] = json_friendly(getattr(message, "usage_metadata", None))

    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = getattr(message, "tool_call_id", None)
        payload["status"] = getattr(message, "status", None)

    return payload


# ── Pydantic Schema Serialization ─────────────────────────────────────


def _annotation_to_string(annotation: Any) -> str:
    if annotation is None:
        return "None"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = ", ".join(_annotation_to_string(item) for item in get_args(annotation))
    origin_name = getattr(origin, "__name__", str(origin))
    return f"{origin_name}[{args}]"


def serialize_base_model_schema(model_cls: type[BaseModel]) -> Dict[str, Any]:
    """Serialize a Pydantic model class to a JSON-friendly schema dict."""
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
            field_payload["default"] = json_friendly(field_info.default)
        properties[field_name] = field_payload
    return {
        "title": model_cls.__name__,
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ── ChatML Formatting ─────────────────────────────────────


def format_tool_call(tool_call: Dict[str, Any]) -> str:
    """Render a tool call as a ChatML XML tag."""
    name = str(tool_call.get("name") or "").strip()
    args = tool_call.get("args")
    return f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>'


def format_tool_result(tool_result: Dict[str, Any]) -> str:
    """Render a tool result as a ChatML XML tag."""
    content = tool_result.get("content")
    rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return f"<tool_result>{rendered}</tool_result>"


# ── Message Role & Content Normalization ─────────────────────────────────────


def normalize_message_role(serialized: Dict[str, Any], *, default: str = "assistant") -> str:
    """Normalize a message role string to standard ChatML roles."""
    raw_role = str(serialized.get("role") or serialized.get("type") or "").strip().lower()
    if raw_role == "human":
        return "user"
    if raw_role == "ai":
        return "assistant"
    if raw_role:
        return raw_role
    return default


def stringify_message_content(value: Any) -> str:
    """Convert message content to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# ── Tool Spec & Extraction ─────────────────────────────────────


# Empty by default — application-layer 6G-specific tool names are injected via
# control_runtime at startup. Keeping the map here preserves backward compat
# during migration; new code should pass capability_aliases explicitly.
_TOOL_CAPABILITY_ALIASES: Dict[str, List[str]] = {}


def register_capability_aliases(aliases: Dict[str, List[str]]) -> None:
    """Register application-layer tool capability aliases at startup.

    Called once by the control-runtime layer to inject 6G-specific tool names
    without hardcoding them in the generic runtime library.
    """
    _TOOL_CAPABILITY_ALIASES.clear()
    _TOOL_CAPABILITY_ALIASES.update(aliases)


def resolve_tool_capabilities(tool_name: str) -> List[str]:
    normalized = str(tool_name or "").strip()
    if not normalized:
        return []
    return list(_TOOL_CAPABILITY_ALIASES.get(normalized, []))


def build_tool_specs(tools: Iterable[Any]) -> List[Dict[str, Any]]:
    """Build a list of tool spec dicts for trace recording."""
    specs: List[Dict[str, Any]] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name and hasattr(tool, "__name__"):
            name = tool.__name__
        description = getattr(tool, "description", None) or getattr(tool, "__doc__", None) or ""
        args_schema = getattr(tool, "args_schema", None)
        capabilities = getattr(tool, "capabilities", None)
        if capabilities is None:
            capabilities = resolve_tool_capabilities(str(name or ""))
        specs.append(
            {
                "name": str(name or ""),
                "description": str(description).strip(),
                "args_schema": json_friendly(args_schema),
                "capabilities": json_friendly(capabilities),
            }
        )
    return specs


def extract_tool_calls(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    """Extract structured tool call records from a sequence of messages."""
    calls: List[Dict[str, Any]] = []
    for item in messages:
        serialized = serialize_message(item) if isinstance(item, BaseMessage) else json_friendly(item)
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
                    "args": json_friendly(call.get("args")),
                    "content": serialized.get("content"),
                    "message_id": serialized.get("id"),
                }
            )
    return calls


def extract_tool_results(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    """Extract structured tool result records from a sequence of messages."""
    results: List[Dict[str, Any]] = []
    for item in messages:
        serialized = serialize_message(item) if isinstance(item, BaseMessage) else json_friendly(item)
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


__all__ = [
    "build_tool_specs",
    "extract_tool_calls",
    "extract_tool_results",
    "format_tool_call",
    "format_tool_result",
    "json_friendly",
    "normalize_message_role",
    "register_capability_aliases",
    "resolve_tool_capabilities",
    "serialize_base_model_schema",
    "serialize_message",
    "stringify_message_content",
]
