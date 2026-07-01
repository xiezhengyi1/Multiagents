from __future__ import annotations

import json
from typing import Any, Dict, List

from shared.runtime import extract_tool_calls, extract_tool_results

_STRUCTURED_PLANNING_TOOLS = {
    "preview_qos_optimizer",
    "fetch_qos_network_status",
    "inspect_mobility_ue_policies",
}


_TRUNCATION_MARKER = "\n... [truncated]"


def _parse_json_object(content: Any) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise ValueError("tool result content is empty")
    # Strip truncation marker that context_policy may have appended after
    # JSON-safe truncation (the suffix sits outside the JSON structure).
    if text.endswith(_TRUNCATION_MARKER):
        text = text[:-len(_TRUNCATION_MARKER)].strip()
    if not text:
        raise ValueError("tool result content is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool result content is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool result content must be a JSON object")
    return payload


def extract_planning_tool_evidence(
    *,
    advisor_result: Dict[str, Any],
    tool_payload_cache: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tool_calls = {
        str(call.get("id") or "").strip(): call
        for call in extract_tool_calls(advisor_result.get("messages") or [])
        if str(call.get("id") or "").strip()
    }
    optimizer_previews: List[Dict[str, Any]] = []
    network_statuses: List[Dict[str, Any]] = []
    mobility_contexts: List[Dict[str, Any]] = []

    for result in extract_tool_results(advisor_result.get("messages") or []):
        tool_name = str(result.get("name") or "").strip()
        if tool_name not in _STRUCTURED_PLANNING_TOOLS:
            continue
        tool_call_id = str(result.get("tool_call_id") or "").strip()
        call_args = tool_calls.get(tool_call_id, {}).get("args") if tool_call_id else {}
        if not isinstance(call_args, dict):
            call_args = {}
        try:
            payload = _parse_json_object(result.get("content"))
        except ValueError as exc:
            # ToolMessage content may be truncated by context_policy.compact_tool_result.
            # Fall back to tool_payload_cache if available; otherwise surface the error.
            cached = (tool_payload_cache or {}).get("latest_optimizer_preview") if tool_name == "preview_qos_optimizer" else None
            if cached is not None:
                payload = dict(cached)
            else:
                raise ValueError(f"invalid planning tool result from {tool_name or '<unknown>'}: {exc}") from exc
        record = {
            "tool_name": tool_name,
            "call_args": dict(call_args),
            "payload": dict(payload),
        }
        if tool_name == "preview_qos_optimizer":
            optimizer_previews.append(record)
        elif tool_name == "fetch_qos_network_status":
            network_statuses.append(record)
        elif tool_name == "inspect_mobility_ue_policies":
            mobility_contexts.append(record)

    # Prefer the full cached payload for optimizer results (avoids truncated-ToolMessage parse issues).
    cached_optimizer = dict((tool_payload_cache or {}).get("latest_optimizer_preview") or {})
    return {
        "optimizer_previews": optimizer_previews,
        "latest_optimizer_preview": (
            cached_optimizer
            or dict((optimizer_previews[-1]["payload"] if optimizer_previews else {}))
        ),
        "network_statuses": network_statuses,
        "latest_network_status": dict((network_statuses[-1]["payload"] if network_statuses else {})),
        "mobility_contexts": mobility_contexts,
        "latest_mobility_context": dict((mobility_contexts[-1]["payload"] if mobility_contexts else {})),
    }


__all__ = ["extract_planning_tool_evidence"]
