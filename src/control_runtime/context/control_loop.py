from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from shared.runtime import ContextPolicy



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
    from ..integrations.storage import get_snapshot_data_by_id

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
    retry_contract = _build_retry_contract(
        pda_feedback=pda_feedback or {},
        diagnosis=diagnosis or {},
        mediator_decision=mediator_decision or {},
        round_index=round_index,
    )
    if retry_contract:
        blocks.append("[Retry Contract]\n" f"retry_contract: {json.dumps(retry_contract, ensure_ascii=False)}")
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


def _build_retry_contract(
    *,
    pda_feedback: Dict[str, Any],
    diagnosis: Dict[str, Any],
    mediator_decision: Dict[str, Any],
    round_index: int,
) -> Dict[str, Any]:
    if not pda_feedback and not diagnosis and not mediator_decision:
        return {}

    root_cause_category = str(diagnosis.get("root_cause_category") or "").strip()
    root_cause = str(diagnosis.get("root_cause") or "").strip()
    reason_summary = str(diagnosis.get("reason_summary") or "").strip()
    correction_suggestion = str(pda_feedback.get("correction_suggestion") or "").strip()
    recommended_actions = [
        str(item or "").strip()
        for item in (diagnosis.get("recommended_actions") or [])
        if str(item or "").strip()
    ]
    affected_flow_ids = _unique_text_values(
        [
            *(diagnosis.get("affected_flow_ids") or []),
            *_extract_ids_from_text(root_cause, prefix="flow"),
            *_extract_ids_from_text(reason_summary, prefix="flow"),
            *_extract_ids_from_text(str(pda_feedback.get("violation_details") or ""), prefix="flow"),
        ]
    )
    affected_policy_ids = _unique_text_values(
        [
            *(diagnosis.get("affected_policy_ids") or []),
            *_extract_ids_from_text(root_cause, prefix="smp"),
            *_extract_ids_from_text(str(pda_feedback.get("violation_details") or ""), prefix="smp"),
        ]
    )
    hard_constraints = []
    unified_constraints = mediator_decision.get("unified_constraints")
    if isinstance(unified_constraints, dict):
        hard_constraints = [
            str(item or "").strip()
            for item in (unified_constraints.get("hard_constraints") or [])
            if str(item or "").strip()
        ]

    retry_goal_parts = [*recommended_actions]
    if correction_suggestion:
        retry_goal_parts.append(correction_suggestion)
    if not retry_goal_parts and reason_summary:
        retry_goal_parts.append(reason_summary)

    contract = {
        "retry_round": int(round_index) + 1,
        "previous_failure_type": root_cause_category or "unknown",
        "repair_goal": "; ".join(retry_goal_parts),
        "must_reuse": {
            "flow_ids": affected_flow_ids,
            "policy_ids": affected_policy_ids,
        },
        "must_call_next": _infer_retry_tool(
            root_cause_category=root_cause_category,
            root_cause=root_cause,
            reason_summary=reason_summary,
            affected_flow_ids=affected_flow_ids,
        ),
        "forbidden_changes": hard_constraints,
    }
    return contract


def _infer_retry_tool(
    *,
    root_cause_category: str,
    root_cause: str,
    reason_summary: str,
    affected_flow_ids: List[str],
) -> str:
    joined = " ".join([root_cause_category, root_cause, reason_summary]).lower()
    if affected_flow_ids or "qos" in joined or "sm policy" in joined or "latency" in joined or "throughput" in joined:
        return "preview_qos_optimizer"
    if "mobility" in joined or "am policy" in joined or "rfsp" in joined or "nssai" in joined:
        return "inspect_mobility_ue_policies"
    return ""


def _extract_ids_from_text(text: str, *, prefix: str) -> List[str]:
    if prefix == "flow":
        pattern = r"\bflow-[A-Za-z0-9_-]+\b"
    elif prefix == "smp":
        pattern = r"\bsmp-[A-Za-z0-9_-]+\b"
    else:
        return []
    return re.findall(pattern, str(text or ""))


def _unique_text_values(values: List[Any]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


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
        "domain_resolution": str(previous_operation_intent.get("domain_resolution") or "").strip(),
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
    "build_feedback_context_from_snapshots",
    "build_main_context",
    "build_memory_context",
    "build_round_feedback_block",
    "rerank_by_context_hints",
]
