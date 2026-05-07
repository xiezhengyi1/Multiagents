from __future__ import annotations

import json
from typing import Any, Dict, List

from shared.runtime import extract_tool_calls, extract_tool_results


def _parse_json_object(content: Any) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_planning_tool_evidence(*, advisor_result: Dict[str, Any]) -> Dict[str, Any]:
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
        tool_call_id = str(result.get("tool_call_id") or "").strip()
        call_args = tool_calls.get(tool_call_id, {}).get("args") if tool_call_id else {}
        if not isinstance(call_args, dict):
            call_args = {}
        payload = _parse_json_object(result.get("content"))
        if not payload:
            continue
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

    return {
        "optimizer_previews": optimizer_previews,
        "latest_optimizer_preview": dict((optimizer_previews[-1]["payload"] if optimizer_previews else {})),
        "network_statuses": network_statuses,
        "latest_network_status": dict((network_statuses[-1]["payload"] if network_statuses else {})),
        "mobility_contexts": mobility_contexts,
        "latest_mobility_context": dict((mobility_contexts[-1]["payload"] if mobility_contexts else {})),
    }


__all__ = ["extract_planning_tool_evidence"]
