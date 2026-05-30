from __future__ import annotations

from typing import Any, Iterable


_FLOW_PROMPT_FIELDS = (
    "flow_id",
    "app_id",
    "supi",
    "name",
    "flow_name",
    "service_type",
    "service_type_id",
    "priority",
    "resolution_status",
    "requested_domains",
    "dnn",
)
_OPERATION_PROMPT_FIELDS = (
    "session_id",
    "snapshot_id",
    "supi",
    "app_id",
    "operation_type",
    "urgency",
    "raw_input",
    "requested_domains",
    "grounded_requested_domains",
    "domain_resolution",
    "domain_revision_needed",
    "domain_revision_rationale",
    "retry_scope",
    "objective_profile_hint",
    "control_semantics",
    "mobility_intent",
)
_COLLABORATION_PROMPT_FIELDS = (
    "round_index",
    "session_id",
    "snapshot_id",
    "snapshot_metadata",
    "memory_context",
    "feedback_context",
    "active_domains",
    "main_round_strategy",
    "main_retry_scope",
    "main_investigation_targets",
    "main_uncertainty_flags",
    "main_routing_decision",
    "main_routing_rationale",
    "main_routing_confidence",
    "main_reuse_contract",
    "main_handoff_expectations",
)
_GLOBAL_INTENT_PROMPT_FIELDS = (
    "session_id",
    "snapshot_id",
    "supi",
    "round_strategy",
    "next_agent",
    "requested_domains",
    "domain_evidence",
    "control_semantics",
    "objective_profile",
    "investigation_targets",
    "uncertainty_flags",
    "retry_scope",
    "required_evidence",
    "forbidden_assumptions",
    "intent_encoding_guidance",
    "routing_decision",
    "routing_rationale",
    "routing_confidence",
    "reuse_contract",
    "handoff_expectations",
)


def _json_mapping(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return dict(payload) if isinstance(payload, dict) else {}


def _without_empty_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", None, [], {})
    }


def _select_fields(payload: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return _without_empty_values({key: payload.get(key) for key in fields})


def project_operation_intent_for_prompt(operation_intent: Any) -> dict[str, Any]:
    """Return the binding-level OperationIntent view needed by the planning LLM."""
    raw = _json_mapping(operation_intent)
    projected = _select_fields(raw, _OPERATION_PROMPT_FIELDS)
    projected["flows"] = [
        _select_fields(flow, _FLOW_PROMPT_FIELDS)
        for flow in (raw.get("flows") or [])
        if isinstance(flow, dict)
    ]
    if not projected["flows"]:
        projected.pop("flows")
    return projected


def project_global_intent_for_prompt(global_intent: Any) -> dict[str, Any]:
    """Return Main's routing contract without replaying trace-only fields."""
    return _select_fields(_json_mapping(global_intent), _GLOBAL_INTENT_PROMPT_FIELDS)


def _project_handoff_summary(handoff: Any) -> dict[str, Any]:
    raw = _json_mapping(handoff)
    summary = _select_fields(raw, ("round_index", "summary"))
    for source_key, summary_key in (
        ("diagnosis", "diagnosis_summary"),
        ("planning_blocker", "planning_blocker_summary"),
        ("execution_reentry", "execution_reentry_summary"),
        ("negotiation_request", "negotiation_summary"),
    ):
        nested = _json_mapping(raw.get(source_key))
        text = str(nested.get("summary") or nested.get("reason_summary") or nested.get("root_cause") or "").strip()
        if text:
            summary[summary_key] = text
    return summary


def project_collaboration_context_for_prompt(context: Any) -> dict[str, Any]:
    """Return collaboration guidance without replaying full prior round artifacts."""
    raw = _json_mapping(context)
    projected = _select_fields(raw, _COLLABORATION_PROMPT_FIELDS)
    handoffs = [
        item
        for item in (_project_handoff_summary(handoff) for handoff in (raw.get("handoff_history") or [])[-2:])
        if item
    ]
    if handoffs:
        projected["handoff_history"] = handoffs
    return projected


def project_intent_evidence_for_prompt(evidence: Any) -> dict[str, Any]:
    """Keep pre-grounded candidates while excluding cached full tool payloads."""
    return _without_empty_values(_json_mapping(evidence))


def project_memory_payload(role: str, payload: Any) -> dict[str, Any]:
    """Persist reusable operational facts, not full trace artifacts."""
    raw = _json_mapping(payload)
    normalized_role = str(role or "").strip().upper()
    if normalized_role == "IEA":
        return project_operation_intent_for_prompt(raw)
    if normalized_role == "MAIN":
        return project_global_intent_for_prompt(raw)
    if normalized_role == "OSA":
        projected = _select_fields(
            raw,
            (
                "supi",
                "session_id",
                "snapshot_id",
                "planning_status",
                "missing_evidence",
                "blocked_targets",
                "upstream_requests",
                "planner_conflicts",
                "planning_rationale",
            ),
        )
        policies = [
            _select_fields(
                policy,
                (
                    "policy_id",
                    "policy_type",
                    "supi",
                    "app_id",
                    "flow_id",
                    "target_type",
                    "resource_keys",
                ),
            )
            for policy in (raw.get("all_policies") or [])
            if isinstance(policy, dict)
        ]
        if policies:
            projected["policies"] = policies
        return projected
    if normalized_role == "AD":
        return _select_fields(
            raw,
            (
                "status",
                "root_cause_category",
                "root_cause",
                "reason_summary",
                "affected_policy_ids",
                "affected_flow_ids",
                "recommended_actions",
            ),
        )
    return _select_fields(raw, ("session_id", "snapshot_id", "supi", "status", "summary"))


__all__ = [
    "project_collaboration_context_for_prompt",
    "project_global_intent_for_prompt",
    "project_intent_evidence_for_prompt",
    "project_memory_payload",
    "project_operation_intent_for_prompt",
]
