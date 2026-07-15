"""Pure round-transition decisions used by the control orchestrators."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..domain.collaboration import DomainNegotiationRequest, ExecutionReentryRequest
from ..domain.control_plane import GlobalControlIntent
from ..domain.policy_plan import OperationIntent


def build_planning_failure_payload(
    exc: Exception,
    *,
    debug_context: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    message = str(exc or "").strip() or exc.__class__.__name__
    report_payload = {
        "execution_status": "Failed",
        "violation_details": message,
        "feedback_payload": {
            "phase": "planning",
            "error": message,
            "target_bindings_at_risk": [],
            "policy_objects_at_risk": [],
            "reason_by_domain": {},
            "debug_context": dict(debug_context or {}),
        },
    }
    diagnosis = {
        "root_cause_category": "planning_failure",
        "root_cause": message,
        "reason_summary": message,
        "recommended_actions": [],
        "debug_context": dict(debug_context or {}),
    }
    return report_payload, diagnosis


def has_supi_scope(global_intent: GlobalControlIntent) -> bool:
    if str(global_intent.supi or "").strip():
        return True
    return any(
        str(target.supi or "").strip()
        for stage in global_intent.control_semantics.stages or []
        for target in stage.targets or []
    )


def build_negotiation_request(operation_intent: OperationIntent, *, round_index: int) -> DomainNegotiationRequest:
    issues = []
    for question in operation_intent.open_questions:
        payload = question.model_dump(mode="json") if hasattr(question, "model_dump") else dict(question)
        issues.append(
            {
                "source_agent": "intent_encoding",
                "issue_type": "domain_boundary",
                "domain": str((payload.get("related_domains") or [""])[0] or ""),
                "binding_keys": [],
                "policy_objects": [],
                "missing_evidence": [str(payload.get("question") or "").strip()] if str(payload.get("question") or "").strip() else [],
                "rationale": str(payload.get("question") or "").strip(),
            }
        )
    domain_resolution = str(operation_intent.domain_resolution or "cannot_confirm").strip() or "cannot_confirm"
    revision_needed = domain_resolution != "confirmed" or bool(issues)
    summary = "; ".join(item["rationale"] for item in issues if str(item.get("rationale") or "").strip())
    return DomainNegotiationRequest(
        round_index=round_index,
        source_agent="intent_encoding",
        main_requested_domains=[],
        grounded_requested_domains=list(operation_intent.requested_domains or []),
        domain_resolution=domain_resolution,
        domain_revision_needed=revision_needed,
        issues=issues,
        recommended_consumers=["main_control", "intent_encoding"],
        summary=summary or domain_resolution,
    )


def build_negotiation_diagnosis(request: DomainNegotiationRequest) -> Dict[str, Any]:
    issue_reasons = [
        str(item.rationale or "").strip()
        for item in (request.issues or [])
        if str(item.rationale or "").strip()
    ]
    return {
        "root_cause_category": "domain_negotiation_required",
        "root_cause": request.summary,
        "reason_summary": request.summary,
        "recommended_actions": issue_reasons,
    }


def build_reentry_report_payload(request: ExecutionReentryRequest) -> Dict[str, Any]:
    return {
        "execution_status": "Failed",
        "violation_details": request.summary,
        "feedback_payload": {
            "failure_scope": request.failure_scope,
            "target_bindings_at_risk": list(request.target_bindings_at_risk or []),
            "policy_objects_at_risk": list(request.policy_objects_at_risk or []),
            "reason_by_domain": dict(request.reason_by_domain or {}),
            "failures": list(request.failures or []),
        },
    }


def should_reuse_operation_intent(
    *,
    global_intent: GlobalControlIntent,
    previous_operation_intent: Optional[OperationIntent],
    previous_report_payload: Dict[str, Any],
    previous_mediator_decision: Optional[Dict[str, Any]],
) -> bool:
    if previous_operation_intent is None:
        return False
    if str(global_intent.next_agent or "").strip().lower() != "optimization_strategy":
        return False
    contract = global_intent.reuse_contract
    if not contract.allowed:
        return False
    if contract.preserve_bindings and str(previous_operation_intent.supi or "").strip() != str(global_intent.supi or "").strip():
        return False
    previous_domains = {str(item or "").strip().lower() for item in (previous_operation_intent.requested_domains or []) if str(item or "").strip()}
    current_domains = {item.value for item in global_intent.requested_domains}
    if contract.preserve_domains and previous_domains != current_domains:
        return False
    if contract.preserve_stage_scope:
        stages = previous_operation_intent.control_semantics.stages or []
        active_stage = next(
            (stage for stage in stages if int(stage.stage_index or 0) == int(previous_operation_intent.control_semantics.current_stage or 1)),
            None,
        )
        if active_stage is None:
            return False
        if not [flow_id for flow_id in (active_stage.active_flow_ids or []) if str(flow_id or "").strip()]:
            return False
    invalidate_text = json.dumps(
        {
            "report": previous_report_payload or {},
            "mediator": previous_mediator_decision or {},
            "open_questions": previous_operation_intent.open_questions,
        },
        ensure_ascii=False,
    ).lower()
    for token in contract.invalidate_on:
        normalized = str(token or "").strip().lower()
        if normalized and normalized in invalidate_text:
            return False
    return not previous_operation_intent.open_questions


def activate_control_stage(*, operation_intent: OperationIntent, round_index: int) -> OperationIntent:
    semantics = operation_intent.control_semantics
    if not semantics.stages:
        return operation_intent
    activated = operation_intent.model_copy(deep=True)
    max_stage = max(stage.stage_index for stage in semantics.stages)
    activated.control_semantics.current_stage = min(max(1, round_index), max_stage)
    return activated


__all__ = [
    "activate_control_stage",
    "build_negotiation_diagnosis",
    "build_negotiation_request",
    "build_planning_failure_payload",
    "build_reentry_report_payload",
    "has_supi_scope",
    "should_reuse_operation_intent",
]
