from __future__ import annotations

from typing import Any, Dict, List

from shared.runtime import extract_tool_calls, extract_tool_results


TOOL_RESULT_MARKERS: Dict[str, str] = {
    "get_sm_ue_flow_catalog": "SM UE Flow Catalog Retrieved:",
    "get_sm_ue_context": "SM UE Context Retrieved:",
    "search_sm_flow_targets": "SM Flow Target Search Retrieved:",
    "get_am_policy_context": "AM Policy Context Retrieved:",
    "search_am_policy_targets": "AM Policy Target Search Retrieved:",
}


def extract_grounding_tool_payloads(*, advisor_result: Dict[str, Any], compiler: Any) -> List[Dict[str, Any]]:
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
        payload = compiler.parse_json_payload_from_tool_result(
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


__all__ = ["TOOL_RESULT_MARKERS", "extract_grounding_tool_payloads"]
