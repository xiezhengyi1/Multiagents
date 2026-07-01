from __future__ import annotations

import json
from typing import Any, Dict, List

from shared.runtime import extract_tool_calls, extract_tool_results


TOOL_RESULT_MARKERS: Dict[str, str] = {
    "get_sm_ue_flow_catalog": "SM UE Flow Catalog Retrieved:",
    "get_sm_ue_context": "SM UE Context Retrieved:",
    "search_sm_flow_targets": "SM Flow Target Search Retrieved:",
    "get_am_policy_context": "AM Policy Context Retrieved:",
    "search_am_policy_targets": "AM Policy Target Search Retrieved:",
}


_TRUNCATION_MARKER = "\n... [truncated]"


def parse_json_payload_from_tool_result(content: Any, *, marker: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}
    # Strip truncation marker that context_policy may have appended after
    # JSON-safe truncation (the suffix sits outside the JSON structure).
    if text.endswith(_TRUNCATION_MARKER):
        text = text[:-len(_TRUNCATION_MARKER)].strip()
    if not text:
        return {}
    payload_text = text
    marker_index = text.find(marker)
    if marker_index >= 0:
        payload_text = text[marker_index + len(marker):].strip()
    start = payload_text.find("{")
    end = payload_text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        payload = json.loads(payload_text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_grounding_tool_payloads(*, advisor_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = {
        str(call.get("id") or "").strip(): call
        for call in extract_tool_calls(advisor_result.get("messages") or [])
        if str(call.get("id") or "").strip()
    }
    payloads: List[Dict[str, Any]] = []
    for result in extract_tool_results(advisor_result.get("messages") or []):
        tool_name = str(result.get("name") or "").strip()
        marker = TOOL_RESULT_MARKERS.get(tool_name)
        if not marker:
            continue
        tool_call_id = str(result.get("tool_call_id") or "").strip()
        call_args = tool_calls.get(tool_call_id, {}).get("args") if tool_call_id else {}
        if not isinstance(call_args, dict):
            call_args = {}
        payload = parse_json_payload_from_tool_result(
            result.get("content"),
            marker=marker,
        )
        if not payload:
            continue
        payloads.append(
            {
                "tool_name": tool_name,
                "call_args": dict(call_args),
                "payload": dict(payload),
            }
        )
    return payloads


__all__ = ["TOOL_RESULT_MARKERS", "extract_grounding_tool_payloads", "parse_json_payload_from_tool_result"]
