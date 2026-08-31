"""Serialization adapters between LangChain and the canonical V2 contracts.

This module is the single source of truth for:
- JSON-safe recursive serialization (json_friendly, serialize_message, etc.)
- LangChain-to-contract message conversion
- strict tool catalog construction from explicit tool metadata

Other modules should import shared message formatting from here
rather than re-implementing these functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, get_args, get_origin

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from shared.contracts import AgentMessage, ToolCall, ToolCatalog, ToolContract


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
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


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


def build_tool_catalog(
    tools: Iterable[Any],
    *,
    capability_aliases: Mapping[str, Iterable[Any]] | None = None,
) -> ToolCatalog:
    """Build a strict, content-addressed catalog from model-visible tools."""

    aliases = {
        str(tool_name).strip(): [str(item).strip() for item in capabilities]
        for tool_name, capabilities in (capability_aliases or {}).items()
    }
    contracts: list[ToolContract] = []
    for tool in tools:
        name = str(getattr(tool, "name", "")).strip()
        if not name:
            raise ValueError("every model tool requires a name")
        description = str(getattr(tool, "description", "")).strip()
        if not description:
            raise ValueError(f"tool {name} requires a description")
        metadata = dict(getattr(tool, "metadata", None) or {})
        contract_metadata = metadata.get("contract")
        if not isinstance(contract_metadata, Mapping):
            raise ValueError(f"tool {name} is missing explicit contract metadata")
        input_model = tool.get_input_schema()
        input_schema = input_model.model_json_schema()
        tool_capabilities = metadata.get("capabilities", aliases.get(name, []))
        contracts.append(
            ToolContract(
                name=name,
                description=description,
                input_schema=json_friendly(input_schema),
                output_schema=json_friendly(contract_metadata.get("output_schema")),
                capabilities=[str(item).strip() for item in tool_capabilities],
                side_effect=str(contract_metadata.get("side_effect")),
                result_format=str(contract_metadata.get("result_format")),
            )
        )
    return ToolCatalog.create(contracts)


def message_to_contract(message: BaseMessage | Mapping[str, Any]) -> AgentMessage:
    """Convert one runtime message without changing tool-call semantics."""

    if isinstance(message, SystemMessage):
        return AgentMessage(role="system", content=stringify_message_content(message.content))
    if isinstance(message, HumanMessage):
        return AgentMessage(role="user", content=stringify_message_content(message.content))
    if isinstance(message, AIMessage):
        calls = [
            ToolCall(
                id=str(call.get("id") or "").strip(),
                name=str(call.get("name") or "").strip(),
                arguments=dict(call.get("args") or {}),
            )
            for call in message.tool_calls
        ]
        return AgentMessage(
            role="assistant",
            content=stringify_message_content(message.content),
            tool_calls=calls,
        )
    if isinstance(message, ToolMessage):
        return AgentMessage(
            role="tool",
            content=stringify_message_content(message.content),
            name=str(message.name or "").strip(),
            tool_call_id=str(message.tool_call_id or "").strip(),
            tool_status=str(message.status or "").strip(),
        )
    if not isinstance(message, Mapping):
        raise TypeError(f"unsupported message type: {type(message).__name__}")
    role = normalize_message_role(dict(message))
    calls = [
        ToolCall(
            id=str(call.get("id") or "").strip(),
            name=str(call.get("name") or "").strip(),
            arguments=dict(call.get("arguments") or call.get("args") or {}),
        )
        for call in message.get("tool_calls") or []
    ]
    return AgentMessage(
        role=role,
        content=stringify_message_content(message.get("content")),
        name=str(message.get("name") or "").strip() or None,
        tool_calls=calls,
        tool_call_id=str(message.get("tool_call_id") or "").strip() or None,
        tool_status=str(message.get("tool_status") or message.get("status") or "").strip() or None,
    )


def contract_to_message(message: AgentMessage) -> BaseMessage:
    """Convert one canonical message to the LangChain transport boundary."""

    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "user":
        return HumanMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(
            content=message.content,
            tool_calls=[
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.arguments,
                    "type": "tool_call",
                }
                for call in message.tool_calls
            ],
        )
    return ToolMessage(
        content=message.content,
        tool_call_id=str(message.tool_call_id),
        name=message.name,
        status=str(message.tool_status),
    )


def extract_tool_calls(messages: Iterable[AgentMessage]) -> List[Dict[str, Any]]:
    """Project canonical assistant tool calls for application-level evidence logic."""
    calls: List[Dict[str, Any]] = []
    for message in messages:
        for call in message.tool_calls:
            calls.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.arguments,
                    "content": message.content,
                }
            )
    return calls


def extract_tool_results(messages: Iterable[AgentMessage]) -> List[Dict[str, Any]]:
    """Project canonical tool results for application-level evidence logic."""
    results: List[Dict[str, Any]] = []
    for message in messages:
        if message.role != "tool":
            continue
        results.append(
            {
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "content": message.content,
                "status": message.tool_status,
            }
        )
    return results


__all__ = [
    "build_tool_catalog",
    "contract_to_message",
    "extract_tool_calls",
    "extract_tool_results",
    "format_tool_call",
    "format_tool_result",
    "json_friendly",
    "message_to_contract",
    "normalize_message_role",
    "serialize_base_model_schema",
    "serialize_message",
    "stringify_message_content",
]
