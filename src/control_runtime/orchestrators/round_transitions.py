"""Pure round-transition decisions used by the control orchestrators."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..domain.collaboration import DomainNegotiationRequest, ExecutionReentryRequest
from ..domain.control_plane import GlobalControlIntent
from ..domain.policy_plan import GroundingDecision


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


def build_negotiation_request(
    grounding_decision: GroundingDecision,
    *,
    round_index: int,
) -> DomainNegotiationRequest:
    issues = []
    for question in grounding_decision.open_questions:
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
    summary = "; ".join(item["rationale"] for item in issues if str(item.get("rationale") or "").strip())
    return DomainNegotiationRequest(
        round_index=round_index,
        source_agent="intent_encoding",
        requires_domain_review=bool(issues),
        issues=issues,
        recommended_consumers=["main_control", "intent_encoding"],
        summary=summary or "IEA reported unresolved grounding questions",
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


def should_reuse_grounding_decision(
    *,
    global_intent: GlobalControlIntent,
    previous_grounding_decision: Optional[GroundingDecision],
    previous_report_payload: Dict[str, Any],
    previous_mediator_decision: Optional[Dict[str, Any]],
) -> bool:
    if previous_grounding_decision is None:
        return False
    # An unresolved question is an IEA-owned semantic boundary. Do not try to
    # serialize and reuse it as an OSA-only retry; Main may instead choose a
    # fresh grounding pass with the latest execution feedback.
    if previous_grounding_decision.open_questions:
        return False
    if str(global_intent.next_agent or "").strip().lower() != "optimization_strategy":
        return False
    contract = global_intent.reuse_contract
    if not contract.allowed:
        return False
    previous_supis = {
        str(flow.supi or "").strip()
        for flow in previous_grounding_decision.flows or []
        if str(flow.supi or "").strip()
    }
    if contract.preserve_bindings and str(global_intent.supi or "").strip() and previous_supis and previous_supis != {str(global_intent.supi).strip()}:
        return False
    invalidate_text = json.dumps(
        {
            "report": previous_report_payload or {},
            "mediator": previous_mediator_decision or {},
            "open_questions": [],
        },
        ensure_ascii=False,
    ).lower()
    for token in contract.invalidate_on:
        normalized = str(token or "").strip().lower()
        if normalized and normalized in invalidate_text:
            return False
    return not previous_grounding_decision.open_questions


__all__ = [
    "build_negotiation_diagnosis",
    "build_negotiation_request",
    "build_planning_failure_payload",
    "build_reentry_report_payload",
    "has_supi_scope",
    "should_reuse_grounding_decision",
]
