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
from typing import Any, Iterable, Mapping

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
        return json_friendly(value.model_json_schema())
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


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Serialize one LangChain transport message to a JSON-friendly dict."""
    payload: dict[str, Any] = {
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


def message_to_contract(message: BaseMessage) -> AgentMessage:
    """Convert one runtime message without changing tool-call semantics."""

    if isinstance(message, SystemMessage):
        return AgentMessage(role="system", content=stringify_message_content(message.content))
    if isinstance(message, HumanMessage):
        return AgentMessage(role="user", content=stringify_message_content(message.content))
    if isinstance(message, AIMessage):
        calls: list[ToolCall] = []
        for call in message.tool_calls:
            arguments = call.get("args")
            if not isinstance(arguments, dict):
                raise TypeError("LangChain tool_call args must be an object")
            calls.append(
                ToolCall(
                    id=str(call.get("id") or "").strip(),
                    name=str(call.get("name") or "").strip(),
                    arguments=arguments,
                )
            )
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
    raise TypeError(f"unsupported message type: {type(message).__name__}")


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


def extract_tool_calls(messages: Iterable[AgentMessage]) -> list[dict[str, Any]]:
    """Project canonical assistant tool calls for application-level evidence logic."""
    calls: list[dict[str, Any]] = []
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


def extract_tool_results(messages: Iterable[AgentMessage]) -> list[dict[str, Any]]:
    """Project canonical tool results for application-level evidence logic."""
    results: list[dict[str, Any]] = []
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
    "json_friendly",
    "message_to_contract",
    "serialize_message",
    "stringify_message_content",
]
