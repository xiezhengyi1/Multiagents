"""Pure round-transition decisions used by the control orchestrators."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..agents.grounding.contracts import GroundingDecision
from ..agents.main.contracts import GlobalControlIntent
from .contracts import (
    DomainNegotiationRequest,
    OrchestrationFailureReport,
)


def build_planning_failure_report(
    exc: Exception,
    *,
    round_index: int,
    debug_context: Optional[Dict[str, Any]] = None,
) -> tuple[OrchestrationFailureReport, Dict[str, Any]]:
    message = str(exc or "").strip() or exc.__class__.__name__
    report = OrchestrationFailureReport(
        failure_stage="planning",
        round_index=round_index,
        violation_details=message,
        error_type=exc.__class__.__name__,
        error=message,
        correction_suggestion="Retry planning after correcting the reported agent failure.",
        recommended_consumer="main_control",
        debug_context=dict(debug_context or {}),
    )
    diagnosis = {
        "root_cause_category": "planning_failure",
        "root_cause": message,
        "reason_summary": message,
        "recommended_actions": [],
        "debug_context": dict(debug_context or {}),
    }
    return report, diagnosis


def scope_global_intent_for_intent_encoding(
    *,
    global_intent: GlobalControlIntent,
    round_index: int,
) -> GlobalControlIntent:
    semantics = global_intent.control_semantics
    stages = list(semantics.stages or [])
    if not stages:
        return global_intent

    ordered_stages = sorted(stages, key=lambda stage: int(stage.stage_index or 0))
    if round_index < 1 or round_index > len(ordered_stages):
        raise ValueError("round_index must select an existing control semantic stage")
    scoped_position = round_index - 1
    scoped_stage = ordered_stages[scoped_position].model_copy(deep=True)
    scoped_semantics = semantics.model_copy(
        update={
            "current_stage": 1,
            "stages": [scoped_stage.model_copy(update={"stage_index": 1}, deep=True)],
        },
        deep=True,
    )
    return global_intent.model_copy(
        update={"control_semantics": scoped_semantics},
        deep=True,
    )


def has_supi_scope(global_intent: GlobalControlIntent) -> bool:
    if global_intent.scope.supi is not None:
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
    if (
        contract.preserve_bindings
        and global_intent.scope.supi is not None
        and previous_supis
        and previous_supis != {global_intent.scope.supi}
    ):
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
    "build_planning_failure_report",
    "has_supi_scope",
    "scope_global_intent_for_intent_encoding",
    "should_reuse_grounding_decision",
]
