from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, get_args, get_origin

from pydantic import BaseModel

_TOOL_CAPABILITY_ALIASES: Dict[str, List[str]] = {
    "get_sm_ue_context": ["sm_ue_context"],
    "get_sm_ue_flow_catalog": ["sm_flow_catalog"],
    "get_ue_flow_catalog": ["sm_flow_catalog"],
    "search_sm_flow_targets": ["sm_flow_target_resolution"],
    "search_flow_targets_by_name": ["sm_flow_target_resolution"],
    "get_am_policy_context": ["am_policy_context"],
    "search_am_policy_targets": ["am_policy_target_resolution"],
    "preview_qos_optimizer": ["optimizer_counterfactual", "qos_runtime_evidence"],
    "preview_optimizer": ["optimizer_counterfactual", "qos_runtime_evidence"],
    "fetch_qos_network_status": ["qos_runtime_evidence"],
    "fetch_network_status": ["qos_runtime_evidence"],
    "inspect_mobility_ue_policies": ["ue_policy_context", "mobility_policy_context"],
    "inspect_ue_policies": ["ue_policy_context", "mobility_policy_context"],
}


def _annotation_to_string(annotation: Any) -> str:
    if annotation is None:
        return "None"
    if annotation is Ellipsis:
        return "..."
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
    if isinstance(value, Mapping):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_friendly(item) for item in value]
    return str(value)


def _serialize_message_like(message: Any) -> Dict[str, Any]:
    if isinstance(message, Mapping):
        return {str(key): _json_friendly(value) for key, value in message.items()}

    payload: Dict[str, Any] = {
        "type": getattr(message, "type", None),
        "role": getattr(message, "role", None),
        "content": _json_friendly(getattr(message, "content", None)),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
        "tool_calls": _json_friendly(getattr(message, "tool_calls", [])),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "status": getattr(message, "status", None),
    }
    return payload


def resolve_tool_capabilities(tool_name: str) -> List[str]:
    normalized = str(tool_name or "").strip()
    if not normalized:
        return []
    return list(_TOOL_CAPABILITY_ALIASES.get(normalized, []))


def build_tool_specs(tools: Iterable[Any]) -> List[Dict[str, Any]]:
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
                "args_schema": _json_friendly(args_schema),
                "capabilities": _json_friendly(capabilities),
            }
        )
    return specs


def extract_tool_calls(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in messages:
        serialized = _serialize_message_like(item)
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
        serialized = _serialize_message_like(item)
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
    "resolve_tool_capabilities",
]
