"""Planning evidence derived from the IEA handoff."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from shared.contracts import AgentRunResult
from shared.runtime import extract_tool_calls, extract_tool_results

from .contracts import PlanningRequest


_STRUCTURED_PLANNING_TOOLS = {
    "preview_qos_optimizer",
    "fetch_qos_network_status",
    "inspect_mobility_ue_policies",
}
_TRUNCATION_MARKER = "\n... [truncated]"


def build_planning_evidence(planning_request: PlanningRequest) -> Dict[str, Any]:
    operation_constraints = derive_qos_operation_constraints(
        planning_request.grounding_decision.model_dump(mode="json"),
    )
    return {"derived_operation_constraints": operation_constraints} if operation_constraints else {}


def derive_qos_operation_constraints(
    grounding_decision_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    flow_by_id = {
        str(flow.get("flow_id") or "").strip(): flow
        for flow in (grounding_decision_payload.get("flows") or [])
        if isinstance(flow, dict) and str(flow.get("flow_id") or "").strip()
    }
    constraints: List[Dict[str, Any]] = []
    for raw_constraint in grounding_decision_payload.get("qos_operation_constraints") or []:
        if not isinstance(raw_constraint, dict):
            continue
        constraint = _normalize_qos_operation_constraint(raw_constraint, flow_by_id)
        if constraint:
            constraints.append(constraint)
    return constraints


def _normalize_qos_operation_constraint(
    raw_constraint: Dict[str, Any],
    flow_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    flow_id = str(raw_constraint.get("flow_id") or "").strip()
    if not flow_id or flow_id not in flow_by_id:
        return None
    flow = flow_by_id[flow_id]
    source_slice = str(raw_constraint.get("source_slice_snssai") or flow.get("current_slice_snssai") or "").strip()
    excluded = [
        str(item or "").strip()
        for item in (raw_constraint.get("excluded_slice_snssais") or [])
        if str(item or "").strip()
    ]
    require_slice_change = bool(raw_constraint.get("require_slice_change", False))
    no_op_allowed = bool(raw_constraint.get("no_op_allowed", not require_slice_change))
    if require_slice_change and source_slice and source_slice not in excluded:
        excluded.append(source_slice)
    return {
        "flow_id": flow_id,
        "app_id": str(raw_constraint.get("app_id") or flow.get("app_id") or "").strip(),
        "operation_type": str(raw_constraint.get("operation_type") or "").strip() or "qos_reallocation",
        "require_slice_change": require_slice_change,
        "source_slice_snssai": source_slice or None,
        "excluded_slice_snssais": excluded,
        "target_slice_preference": str(raw_constraint.get("target_slice_preference") or "").strip(),
        "no_op_allowed": no_op_allowed,
        "rationale": list(raw_constraint.get("rationale") or []),
    }


def extract_planning_tool_evidence(
    *,
    advisor_result: AgentRunResult,
    tool_payload_cache: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Collect the structured planning-tool evidence used by validation and compilation."""
    tool_calls = {
        str(call.get("id") or "").strip(): call
        for call in extract_tool_calls(advisor_result.messages)
        if str(call.get("id") or "").strip()
    }
    optimizer_previews: List[Dict[str, Any]] = []
    network_statuses: List[Dict[str, Any]] = []
    mobility_contexts: List[Dict[str, Any]] = []

    for result in extract_tool_results(advisor_result.messages):
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
            cached = (
                (tool_payload_cache or {}).get("latest_optimizer_preview")
                if tool_name == "preview_qos_optimizer"
                else None
            )
            if cached is None:
                raise ValueError(
                    f"invalid planning tool result from {tool_name or '<unknown>'}: {exc}"
                ) from exc
            payload = dict(cached)
        record = {
            "tool_name": tool_name,
            "call_args": dict(call_args),
            "payload": dict(payload),
        }
        if tool_name == "preview_qos_optimizer":
            optimizer_previews.append(record)
        elif tool_name == "fetch_qos_network_status":
            network_statuses.append(record)
        else:
            mobility_contexts.append(record)

    cached_optimizer = dict((tool_payload_cache or {}).get("latest_optimizer_preview") or {})
    latest_tool_payload = dict(optimizer_previews[-1]["payload"] if optimizer_previews else {})
    if cached_optimizer and latest_tool_payload:
        cached_optimizer = {
            **cached_optimizer,
            **{
                key: value
                for key, value in latest_tool_payload.items()
                if key in {"summary", "status", "qos_flow_assignments", "qos_meta_status"}
            },
        }
    return {
        "optimizer_previews": optimizer_previews,
        "latest_optimizer_preview": cached_optimizer or latest_tool_payload,
        "network_statuses": network_statuses,
        "latest_network_status": dict(network_statuses[-1]["payload"] if network_statuses else {}),
        "mobility_contexts": mobility_contexts,
        "latest_mobility_context": dict(mobility_contexts[-1]["payload"] if mobility_contexts else {}),
    }


def _parse_json_object(content: Any) -> Dict[str, Any]:
    text = str(content or "").strip()
    if text.endswith(_TRUNCATION_MARKER):
        text = text[: -len(_TRUNCATION_MARKER)].strip()
    if not text:
        raise ValueError("tool result content is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool result content is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool result content must be a JSON object")
    return payload


__all__ = [
    "build_planning_evidence",
    "derive_qos_operation_constraints",
    "extract_planning_tool_evidence",
]
