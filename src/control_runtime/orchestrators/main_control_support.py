from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shared.runtime import ContextPolicy

from ..agents.dispatch.contracts import FeedbackReport
from ..domain.collaboration import DomainNegotiationRequest, ExecutionReentryRequest, PlanningBlockerReport, PlanningContext
from ..domain.control_plane import GlobalControlIntent
from ..integrations.storage import get_latest_snapshot_metadata, get_snapshot_data_by_id


@dataclass
class ControlRoundTrace:
    round_index: int
    global_intent: Dict[str, Any] = field(default_factory=dict)
    operation_intent: Dict[str, Any] = field(default_factory=dict)
    policy_plan: Dict[str, Any] = field(default_factory=dict)
    domain_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    pda_feedback: Dict[str, Any] = field(default_factory=dict)
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlRoundResult:
    session_id: str
    snapshot_id: str
    completed: bool
    global_intent: Dict[str, Any]
    unified_plan: Dict[str, Any]
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)
    round_count: int = 1
    retry_count: int = 0
    round_traces: List[Dict[str, Any]] = field(default_factory=list)


def build_main_context(
    snapshot_id: str,
    *,
    round_index: int,
    memory_context: str = "",
    feedback_context: str = "",
    external_routing_hint: Optional[Dict[str, Any]] = None,
    previous_diagnosis: Optional[Dict[str, Any]] = None,
    previous_execution_feedback: Optional[Dict[str, Any]] = None,
    previous_operation_intent: Optional[Dict[str, Any]] = None,
    previous_negotiation_request: Optional[Dict[str, Any]] = None,
    previous_planning_blocker: Optional[Dict[str, Any]] = None,
    previous_execution_reentry: Optional[Dict[str, Any]] = None,
) -> str:
    snapshot = get_snapshot_data_by_id(snapshot_id) or {}
    if not snapshot:
        raise LookupError(f"bound snapshot not found: snapshot_id={snapshot_id}")
    retry_source = previous_execution_feedback if isinstance(previous_execution_feedback, dict) and previous_execution_feedback else previous_diagnosis or {}
    return (
        "## Snapshot Summary\n"
        f"{_render_mapping({'snapshot_id': snapshot_id, 'round_index': round_index})}\n"
        f"{_render_snapshot_summary(_build_snapshot_summary(snapshot))}\n\n"
        "## Previous Round Diagnosis\n"
        f"{_render_mapping(previous_diagnosis or {})}\n\n"
        "## Previous Operation Intent\n"
        f"{_render_mapping(_build_previous_operation_intent_summary(previous_operation_intent or {}))}\n\n"
        "## Retry Hints\n"
        f"{_render_json_list(_build_execution_retry_hints(retry_source))}\n\n"
        "## Conflict And Assurance Signals\n"
        f"{_render_mapping({'mediator_conflict_summary': _build_mediator_conflict_summary(previous_execution_feedback or previous_diagnosis or {}), 'assurance_failure_summary': _build_assurance_failure_summary(previous_execution_feedback or {})})}\n\n"
        "## Collaboration Requests\n"
        f"{_render_mapping({'negotiation_request': previous_negotiation_request or {}, 'planning_blocker': previous_planning_blocker or {}, 'execution_reentry': previous_execution_reentry or {}})}\n\n"
        "## External Routing Hint\n"
        f"{_render_mapping(external_routing_hint or {})}\n\n"
        "## Memory Context\n"
        f"{memory_context or 'N/A'}\n\n"
        "## Feedback\n"
        f"{feedback_context or 'N/A'}"
    )


def build_planning_context(
    global_intent: GlobalControlIntent,
    session_id: str,
    snapshot_id: str,
    *,
    active_domains: Optional[List[str]] = None,
    round_index: int,
    memory_context: str = "",
    feedback_context: str = "",
    handoff_history: Optional[List[Dict[str, Any]]] = None,
    revision_requests: Optional[List[Dict[str, Any]]] = None,
    unified_constraints: Optional[Dict[str, Any]] = None,
) -> PlanningContext:
    return PlanningContext(
        round_index=round_index,
        session_id=session_id,
        snapshot_id=snapshot_id,
        snapshot_metadata={**(get_latest_snapshot_metadata() or {}), "snapshot_id": snapshot_id},
        memory_context=memory_context,
        feedback_context=feedback_context,
        handoff_history=list(handoff_history or [])[-2:],
        active_domains=list(active_domains or [item.value for item in global_intent.requested_domains]),
        main_round_strategy=global_intent.round_strategy.value,
        main_retry_scope=(
            global_intent.retry_scope.value
            if getattr(global_intent, "retry_scope", None) is not None and hasattr(global_intent.retry_scope, "value")
            else str(getattr(global_intent, "retry_scope", "") or "").strip()
        ),
        main_investigation_targets=[item.value for item in global_intent.investigation_targets],
        main_uncertainty_flags=[item.value for item in global_intent.uncertainty_flags],
        main_routing_decision=str(global_intent.routing_decision or "").strip(),
        main_routing_rationale=str(global_intent.routing_rationale or "").strip(),
        main_reuse_contract=global_intent.reuse_contract.model_dump(mode="json"),
        objective_profile=global_intent.objective_profile.model_dump(mode="json"),
        forbidden_assumptions=list(global_intent.forbidden_assumptions or []),
        required_evidence=list(global_intent.required_evidence or []),
        revision_requests=list(revision_requests or []),
        unified_constraints=dict(unified_constraints or {}),
    )


def parse_pda_metrics(report: FeedbackReport) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_metrics = str(report.performance_metrics or "").strip()
    if not raw_metrics:
        return [], []
    payload = json.loads(raw_metrics)
    if not isinstance(payload, dict):
        raise TypeError("performance_metrics must be a JSON object")
    return (
        payload.get("dispatch_results", []) if isinstance(payload.get("dispatch_results"), list) else [],
        payload.get("assurance_results", []) if isinstance(payload.get("assurance_results"), list) else [],
    )


def build_round_feedback_block(
    *,
    pda_feedback: Optional[Dict[str, Any]] = None,
    diagnosis: Optional[Dict[str, Any]] = None,
    domain_verdicts: Optional[List[Dict[str, Any]]] = None,
    mediator_decision: Optional[Dict[str, Any]] = None,
    negotiation_request: Optional[Dict[str, Any]] = None,
    planning_blocker: Optional[Dict[str, Any]] = None,
    execution_reentry: Optional[Dict[str, Any]] = None,
    round_index: int,
) -> str:
    blocks: List[str] = []
    if pda_feedback:
        retry_hints = _build_execution_retry_hints(pda_feedback)
        blocks.append(
            "[Round Feedback]\n"
            f"round_index: {round_index}\n"
            f"execution_status: {pda_feedback.get('execution_status', '')}\n"
            f"violation_details: {pda_feedback.get('violation_details', '')}\n"
            + (f"retry_hints: {json.dumps(retry_hints, ensure_ascii=False)}\n" if retry_hints else "")
        )
    if diagnosis:
        blocks.append(
            "[Diagnosis]\n"
            f"root_cause_category: {diagnosis.get('root_cause_category', '')}\n"
            f"root_cause: {diagnosis.get('root_cause', '')}\n"
            f"reason_summary: {diagnosis.get('reason_summary', '')}\n"
        )
    if negotiation_request:
        blocks.append(
            "[Negotiation]\n"
            f"domain_resolution: {negotiation_request.get('domain_resolution', '')}\n"
            f"summary: {negotiation_request.get('summary', '')}\n"
            f"recommended_consumers: {json.dumps(negotiation_request.get('recommended_consumers') or [], ensure_ascii=False)}\n"
        )
    if planning_blocker:
        blocks.append(
            "[Planning Blocker]\n"
            f"planning_status: {planning_blocker.get('planning_status', '')}\n"
            f"summary: {planning_blocker.get('summary', '')}\n"
            f"upstream_requests: {json.dumps(planning_blocker.get('upstream_requests') or [], ensure_ascii=False)}\n"
        )
    if execution_reentry:
        blocks.append(
            "[Execution Reentry]\n"
            f"failure_scope: {execution_reentry.get('failure_scope', '')}\n"
            f"summary: {execution_reentry.get('summary', '')}\n"
            f"recommended_consumers: {json.dumps(execution_reentry.get('recommended_consumers') or [], ensure_ascii=False)}\n"
        )
    if domain_verdicts:
        lines: List[str] = []
        for verdict in domain_verdicts:
            if not isinstance(verdict, dict):
                continue
            domain = str(verdict.get("domain") or "").strip()
            status = str(verdict.get("status") or "").strip()
            for item in verdict.get("hard_conflicts") or []:
                text = str(item or "").strip()
                if text:
                    lines.append(f"{domain}:{status}: {text}")
        if lines:
            blocks.append("[Conflict Evidence]\n" + "\n".join(lines))
    if mediator_decision:
        mediator_lines: List[str] = []
        mediator_lines.append(f"mediator_status: {mediator_decision.get('status', '')}")
        mediator_lines.append(f"reason_summary: {mediator_decision.get('reason_summary', '')}")
        for item in mediator_decision.get("revision_requests") or []:
            if not isinstance(item, dict):
                continue
            target_domain = str(item.get("target_domain") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if target_domain or reason:
                mediator_lines.append(f"revision:{target_domain}: {reason}")
        hard_constraints = ((mediator_decision.get("unified_constraints") or {}).get("hard_constraints") or []) if isinstance(mediator_decision.get("unified_constraints"), dict) else []
        for item in hard_constraints:
            text = str(item or "").strip()
            if text:
                mediator_lines.append(f"constraint: {text}")
        if mediator_lines:
            blocks.append("[Mediator]\n" + "\n".join(mediator_lines))
    return "\n".join(block.strip() for block in blocks if block.strip())


def build_feedback_context_from_snapshots(
    snapshots: List[Any],
    *,
    token_counter: Any = None,
    summarizer_llm: Any = None,
) -> str:
    recent = list(snapshots[-2:])
    older = list(snapshots[:-2])
    blocks: List[str] = []
    if older:
        blocks.append(_summarize_old_feedback(older, summarizer_llm=summarizer_llm))
    blocks.extend(
        str(getattr(snap, "feedback_added", "") or "").strip()
        for snap in recent
        if str(getattr(snap, "feedback_added", "") or "").strip()
    )
    return _truncate_feedback_context("\n".join(block for block in blocks if block.strip()), token_counter=token_counter)


_MAX_FEEDBACK_CHARS = 4000
_MAX_FEEDBACK_TOKENS = 2000
_CONTEXT_POLICY = ContextPolicy()


def _truncate_feedback_context(feedback: str, token_counter: Any = None) -> str:
    """Limit feedback context, keeping the most recent blocks when truncated."""
    return _CONTEXT_POLICY.compact_text(
        feedback,
        max_chars=_MAX_FEEDBACK_CHARS,
        max_tokens=_MAX_FEEDBACK_TOKENS,
        token_counter=token_counter,
    )


def _summarize_old_feedback(old_snapshots: List[Any], *, summarizer_llm: Any = None) -> str:
    source_blocks = [
        {
            "round_index": getattr(snap, "round_index", ""),
            "diagnosis": getattr(snap, "diagnosis", {}) or {},
            "feedback": str(getattr(snap, "feedback_added", "") or "").strip(),
        }
        for snap in old_snapshots
    ]
    if summarizer_llm is not None and source_blocks:
        prompt = (
            "Summarize older control-loop feedback in 1-2 concise sentences. "
            "Preserve root cause categories, affected bindings, and retry-relevant constraints.\n\n"
            f"{json.dumps(source_blocks, ensure_ascii=False)}"
        )
        try:
            response = summarizer_llm.invoke(prompt) if hasattr(summarizer_llm, "invoke") else summarizer_llm(prompt)
            text = str(getattr(response, "content", response) or "").strip()
            if text:
                return "[Older Feedback Summary]\n" + text
        except Exception:
            pass

    lines: List[str] = []
    for item in source_blocks:
        diagnosis = item["diagnosis"] if isinstance(item["diagnosis"], dict) else {}
        category = str(diagnosis.get("root_cause_category") or "").strip()
        reason = str(diagnosis.get("reason_summary") or diagnosis.get("root_cause") or "").strip()
        if category or reason:
            lines.append(
                f"round {item['round_index']}: "
                + "; ".join(part for part in [category, reason] if part)
            )
    if not lines:
        lines = [f"round {item['round_index']}: prior feedback present" for item in source_blocks]
    return "[Older Feedback Summary]\n" + "\n".join(lines[:4])


def _render_mapping(payload: Dict[str, Any]) -> str:
    if not payload:
        return "N/A"
    lines: List[str] = []
    for key, value in payload.items():
        if value in ("", None, [], {}):
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines) if lines else "N/A"


def _render_json_list(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "N/A"
    return "\n".join(f"- {json.dumps(item, ensure_ascii=False)}" for item in items)


def _render_snapshot_summary(summary: Dict[str, Any]) -> str:
    sections: List[str] = []
    for key in (
        "counts",
        "candidate_apps",
        "candidate_flows",
        "candidate_slices",
        "candidate_nodes",
        "mobility_targets",
    ):
        value = summary.get(key)
        if value in (None, {}, []):
            continue
        sections.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(sections) if sections else "- counts: {}"


def _build_execution_retry_hints(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    feedback_payload = payload.get("feedback_payload")
    if not isinstance(feedback_payload, dict):
        controller_feedback = {}
    else:
        controller_feedback = feedback_payload.get("controller_feedback")
        if not isinstance(controller_feedback, dict):
            controller_feedback = {}

    failures = controller_feedback.get("failures")
    normalized_failures = [item for item in (failures or []) if isinstance(item, dict)]
    if not normalized_failures and isinstance(feedback_payload, dict) and feedback_payload:
        normalized_failures = [feedback_payload]

    results: List[Dict[str, Any]] = []
    for failure in normalized_failures:
        last_dispatch = failure.get("last_dispatch_result")
        if not isinstance(last_dispatch, dict):
            nested = failure.get("feedback_payload")
            last_dispatch = nested.get("last_dispatch_result") if isinstance(nested, dict) else {}
        if not isinstance(last_dispatch, dict):
            last_dispatch = {}
        result = {
            "policy_id": str(failure.get("policy_id") or feedback_payload.get("policy_id") or "").strip(),
            "policy_type": str(failure.get("policy_type") or feedback_payload.get("policy_type") or "").strip(),
            "flow_id": str(failure.get("flow_id") or feedback_payload.get("flow_id") or "").strip(),
            "phase": str(failure.get("phase") or feedback_payload.get("phase") or "").strip(),
            "error": str(failure.get("error") or feedback_payload.get("reason") or payload.get("violation_details") or "").strip(),
            "recommended_consumers": _normalize_recommended_consumers(
                failure.get("recommended_consumers")
                or feedback_payload.get("recommended_consumers")
            ),
            "response_code": last_dispatch.get("response_code"),
        }
        compact = {key: value for key, value in result.items() if value not in ("", None, [])}
        if compact:
            results.append(compact)
    return results


def _normalize_recommended_consumers(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _build_snapshot_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    apps = snapshot.get("apps") if isinstance(snapshot.get("apps"), list) else []
    slices = snapshot.get("slices") if isinstance(snapshot.get("slices"), list) else []
    nodes = snapshot.get("nodes") if isinstance(snapshot.get("nodes"), list) else []
    mobility = snapshot.get("mobility") if isinstance(snapshot.get("mobility"), list) else []
    raw_flows = snapshot.get("flows") if isinstance(snapshot.get("flows"), list) else []
    candidate_flows = [item for item in raw_flows if isinstance(item, dict)]
    if not candidate_flows:
        for app in apps:
            if not isinstance(app, dict):
                continue
            flows = app.get("flows")
            if isinstance(flows, list):
                candidate_flows.extend(item for item in flows if isinstance(item, dict))
    return {
        "counts": {
            "apps": len(apps),
            "slices": len(slices),
            "nodes": len(nodes),
            "mobility_ues": len(mobility),
        },
        "candidate_apps": [
            {
                "app_id": str(item.get("app_id") or "").strip(),
                "app_name": str(item.get("app_name") or "").strip(),
            }
            for item in apps[:5]
            if isinstance(item, dict)
        ],
        "candidate_flows": [
            {
                "flow_id": str(item.get("flow_id") or item.get("id") or "").strip(),
                "app_id": str(item.get("app_id") or "").strip(),
                "flow_name": str(item.get("flow_name") or item.get("name") or "").strip(),
                "priority": item.get("priority"),
            }
            for item in candidate_flows[:8]
        ],
        "candidate_slices": [
            {
                "slice_id": str(item.get("id") or item.get("slice_id") or "").strip(),
                "snssai": item.get("snssai"),
                "status": item.get("status"),
            }
            for item in slices[:5]
            if isinstance(item, dict)
        ],
        "candidate_nodes": [
            {
                "node_id": str(item.get("id") or item.get("node_id") or "").strip(),
                "status": item.get("status"),
                "load": item.get("load"),
            }
            for item in nodes[:5]
            if isinstance(item, dict)
        ],
        "mobility_targets": [
            {
                "supi": str(item.get("supi") or "").strip(),
                "currentAssociationId": item.get("currentAssociationId"),
                "rfsp": item.get("currentRfsp"),
            }
            for item in mobility[:5]
            if isinstance(item, dict)
        ],
    }


def _build_previous_operation_intent_summary(previous_operation_intent: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(previous_operation_intent, dict):
        return {}
    flows = previous_operation_intent.get("flows") if isinstance(previous_operation_intent.get("flows"), list) else []
    return {
        "supi": str(previous_operation_intent.get("supi") or "").strip(),
        "requested_domains": list(previous_operation_intent.get("requested_domains") or []),
        "grounded_requested_domains": list(previous_operation_intent.get("grounded_requested_domains") or []),
        "domain_revision_needed": bool(previous_operation_intent.get("domain_revision_needed") or False),
        "control_semantics": previous_operation_intent.get("control_semantics") or {},
        "flow_bindings": [
            {
                "flow_id": str(item.get("flow_id") or "").strip(),
                "app_id": str(item.get("app_id") or "").strip(),
                "resolution_status": str(item.get("resolution_status") or "").strip(),
            }
            for item in flows
            if isinstance(item, dict)
        ],
    }


def _build_mediator_conflict_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    feedback_payload = payload.get("feedback_payload")
    controller_feedback = (
        feedback_payload.get("controller_feedback")
        if isinstance(feedback_payload, dict) and isinstance(feedback_payload.get("controller_feedback"), dict)
        else {}
    )
    failures = controller_feedback.get("failures") if isinstance(controller_feedback.get("failures"), list) else []
    reason_by_domain: Dict[str, List[str]] = {}
    for item in failures:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain") or item.get("target_domain") or "").strip() or "unknown"
        reason = str(item.get("error") or item.get("reason") or "").strip()
        if reason:
            reason_by_domain.setdefault(domain, []).append(reason)
    return {"reason_by_domain": reason_by_domain}


def _build_assurance_failure_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    feedback_payload = payload.get("feedback_payload")
    controller_feedback = (
        feedback_payload.get("controller_feedback")
        if isinstance(feedback_payload, dict) and isinstance(feedback_payload.get("controller_feedback"), dict)
        else {}
    )
    failures = controller_feedback.get("failures") if isinstance(controller_feedback.get("failures"), list) else []
    return {
        "failures": [
            {
                "policy_id": str(item.get("policy_id") or "").strip(),
                "policy_type": str(item.get("policy_type") or "").strip(),
                "flow_id": str(item.get("flow_id") or "").strip(),
                "phase": str(item.get("phase") or "").strip(),
                "error": str(item.get("error") or "").strip(),
            }
            for item in failures
            if isinstance(item, dict)
        ]
    }
