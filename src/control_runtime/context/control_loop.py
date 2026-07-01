from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shared.runtime import ContextPolicy

from ..domain.collaboration import DomainNegotiationRequest, ExecutionReentryRequest, PlanningBlockerReport, PlanningContext
from ..domain.control_plane import GlobalControlIntent
from ..domain.policy_plan import OperationIntent
from .projectors import project_global_intent_for_prompt


def get_snapshot_data_by_id(snapshot_id: str) -> Dict[str, Any]:
    from ..integrations.storage import get_snapshot_data_by_id as _get_snapshot_data_by_id

    return _get_snapshot_data_by_id(snapshot_id) or {}


def get_latest_snapshot_metadata() -> Dict[str, Any]:
    from ..integrations.storage import get_latest_snapshot_metadata as _get_latest_snapshot_metadata

    return _get_latest_snapshot_metadata() or {}


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


def parse_pda_metrics(report: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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


def build_memory_context(
    user_input: str,
    *,
    memory_manager: Any,
    diagnosis_hint: str = "",
    routing_hint: str = "",
    context_policy: Optional[ContextPolicy] = None,
    token_counter: Any = None,
) -> str:
    bundle = memory_manager.retrieve(user_input)
    short_term = bundle.get("short_term", []) if isinstance(bundle, dict) else []
    long_term = bundle.get("long_term", []) if isinstance(bundle, dict) else []
    short_term = rerank_by_context_hints(short_term, diagnosis_hint, routing_hint)[:5]
    long_term = rerank_by_context_hints(long_term, diagnosis_hint, routing_hint)[:5]
    blocks: List[str] = []
    if short_term:
        blocks.append(
            "[Memory][Short-Term]\n"
            + "\n".join(
                f"{item.get('role', 'unknown')}: {item.get('content', '')}"
                for item in short_term
                if isinstance(item, dict)
            )
        )
    if long_term:
        blocks.append("[Memory][Long-Term]\n" + "\n".join(str(item) for item in long_term))
    rendered = "\n\n".join(block for block in blocks if block.strip())
    policy = context_policy or ContextPolicy()
    return policy.compact_text(
        rendered,
        max_chars=6000,
        max_tokens=1500,
        token_counter=token_counter,
    )


def rerank_by_context_hints(items: List[Any], diagnosis_hint: str = "", routing_hint: str = "") -> List[Any]:
    hint_text = f"{diagnosis_hint} {routing_hint}".lower()
    hint_tokens = {
        token
        for token in re.split(r"[^a-zA-Z0-9_]+", hint_text)
        if len(token) >= 3
    }
    if not hint_tokens:
        return list(items)

    def score_item(index_item: tuple[int, Any]) -> tuple[int, int]:
        index, item = index_item
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False).lower()
        else:
            text = str(item or "").lower()
        item_tokens = {
            token
            for token in re.split(r"[^a-zA-Z0-9_]+", text)
            if len(token) >= 3
        }
        return (len(hint_tokens & item_tokens), -index)

    ranked = sorted(enumerate(items), key=score_item, reverse=True)
    return [item for _, item in ranked]


def build_intent_encoding_context(
    *,
    global_intent: Dict[str, Any],
    snapshot_id: str,
    round_index: int,
    diagnosis: Dict[str, Any],
    feedback_context: str,
) -> str:
    snapshot = get_snapshot_data_by_id(snapshot_id) or {}
    projected_global_intent = project_global_intent_for_prompt(global_intent)
    return (
        "## Guidance\n"
        f"- round_index: {round_index}\n"
        f"- intent_encoding_guidance: {global_intent.get('intent_encoding_guidance', '') or 'N/A'}\n"
        f"- retry_scope: {global_intent.get('retry_scope') or 'N/A'}\n"
        f"- routing_decision: {global_intent.get('routing_decision') or 'N/A'}\n"
        f"- routing_rationale: {global_intent.get('routing_rationale') or 'N/A'}\n\n"
        "## Evidence\n"
        f"- main_intent: {json.dumps(projected_global_intent, ensure_ascii=False)}\n"
        f"- snapshot_summary: {json.dumps(_build_snapshot_summary(snapshot) if snapshot else {}, ensure_ascii=False)}\n"
        "\n"
        "## Previous Diagnosis\n"
        f"{json.dumps(diagnosis or {}, ensure_ascii=False)}\n\n"
        "## Feedback\n"
        f"{feedback_context or 'N/A'}"
    )


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
    scoped_position = min(max(1, int(round_index or 1)), len(ordered_stages)) - 1
    scoped_stage = ordered_stages[scoped_position].model_copy(deep=True)
    scoped_semantics = semantics.model_copy(
        update={
            "current_stage": 1,
            "stages": [
                scoped_stage.model_copy(
                    update={"stage_index": 1},
                    deep=True,
                )
            ],
        },
        deep=True,
    )
    return global_intent.model_copy(
        update={"control_semantics": scoped_semantics},
        deep=True,
    )


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
    for stage in global_intent.control_semantics.stages or []:
        for target in stage.targets or []:
            if str(target.supi or "").strip():
                return True
    return False


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
    return DomainNegotiationRequest(
        round_index=round_index,
        source_agent="intent_encoding",
        main_requested_domains=list(operation_intent.main_requested_domains or []),
        grounded_requested_domains=list(operation_intent.grounded_requested_domains or operation_intent.requested_domains or []),
        domain_resolution=str(operation_intent.domain_resolution or "cannot_confirm"),
        domain_revision_needed=bool(operation_intent.domain_revision_needed),
        issues=issues,
        recommended_consumers=["main_control", "intent_encoding"],
        summary=str(operation_intent.domain_revision_rationale or "").strip(),
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
            (
                stage
                for stage in stages
                if int(stage.stage_index or 0) == int(previous_operation_intent.control_semantics.current_stage or 1)
            ),
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
    if previous_operation_intent.open_questions:
        return False
    return True


def activate_control_stage(
    *,
    operation_intent: OperationIntent,
    round_index: int,
) -> OperationIntent:
    semantics = operation_intent.control_semantics
    if not semantics.stages:
        return operation_intent
    activated = operation_intent.model_copy(deep=True)
    max_stage = max(stage.stage_index for stage in semantics.stages)
    activated.control_semantics.current_stage = min(max(1, round_index), max_stage)
    return activated


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


__all__ = [
    "ControlRoundResult",
    "ControlRoundTrace",
    "activate_control_stage",
    "build_feedback_context_from_snapshots",
    "build_intent_encoding_context",
    "build_main_context",
    "build_memory_context",
    "build_negotiation_diagnosis",
    "build_negotiation_request",
    "build_planning_context",
    "build_planning_failure_payload",
    "build_reentry_report_payload",
    "build_round_feedback_block",
    "get_latest_snapshot_metadata",
    "get_snapshot_data_by_id",
    "has_supi_scope",
    "parse_pda_metrics",
    "rerank_by_context_hints",
    "scope_global_intent_for_intent_encoding",
    "should_reuse_operation_intent",
]
