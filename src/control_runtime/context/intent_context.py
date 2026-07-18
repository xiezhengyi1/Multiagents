"""Stable context-construction boundary for Main -> IEA -> OSA handoffs.

The orchestrator depends on this interface rather than on individual context
fields.  Alternative context policies can therefore be developed and tested
without changing orchestration or agent implementations.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Protocol

from ..domain.collaboration import PlanningContext, QosRetryConstraint, SharedControlContext
from ..domain.control_plane import GlobalControlIntent
from .projectors import project_global_intent_for_prompt


class IntentContextBuilder(Protocol):
    """Build the two bounded contexts consumed by IEA and OSA."""

    def build_intent_encoding_context(
        self,
        *,
        global_intent: Dict[str, Any],
        round_index: int,
        diagnosis: Dict[str, Any],
        feedback_context: str,
    ) -> str: ...

    def build_planning_context(
        self,
        global_intent: GlobalControlIntent,
        session_id: str,
        snapshot_id: str,
        *,
        round_index: int,
        memory_context: str = "",
        feedback_context: str = "",
        handoff_history: Optional[List[Dict[str, Any]]] = None,
        revision_requests: Optional[List[Dict[str, Any]]] = None,
        unified_constraints: Optional[Dict[str, Any]] = None,
        previous_execution_feedback: Optional[Dict[str, Any]] = None,
    ) -> PlanningContext: ...

    def scope_global_intent_for_intent_encoding(
        self,
        *,
        global_intent: GlobalControlIntent,
        round_index: int,
    ) -> GlobalControlIntent: ...


class DefaultIntentContextBuilder:
    """Canonical, minimal context policy for the CoreAgents pipeline."""

    def build_intent_encoding_context(
        self,
        *,
        global_intent: Dict[str, Any],
        round_index: int,
        diagnosis: Dict[str, Any],
        feedback_context: str,
    ) -> str:
        payload: Dict[str, Any] = {
            "round_index": round_index,
            "main_intent": _project_iea_routing_contract(global_intent),
        }
        retry_delta = _project_iea_retry_delta(
            diagnosis=diagnosis,
            feedback_context=feedback_context,
        )
        if int(round_index or 1) > 1 and retry_delta:
            payload["retry_delta"] = retry_delta
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def build_planning_context(
        self,
        global_intent: GlobalControlIntent,
        session_id: str,
        snapshot_id: str,
        *,
        round_index: int,
        memory_context: str = "",
        feedback_context: str = "",
        handoff_history: Optional[List[Dict[str, Any]]] = None,
        revision_requests: Optional[List[Dict[str, Any]]] = None,
        unified_constraints: Optional[Dict[str, Any]] = None,
        previous_execution_feedback: Optional[Dict[str, Any]] = None,
    ) -> PlanningContext:
        return PlanningContext(
            round_index=round_index,
            session_id=session_id,
            snapshot_id=snapshot_id,
            memory_context=memory_context,
            shared_context=_build_shared_control_context(global_intent),
            feedback_context=feedback_context,
            handoff_history=list(handoff_history or [])[-2:],
            revision_requests=list(revision_requests or []),
            unified_constraints=dict(unified_constraints or {}),
            qos_retry_constraints=_qos_retry_constraints_from_feedback(previous_execution_feedback),
        )

    def scope_global_intent_for_intent_encoding(
        self,
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
                "stages": [scoped_stage.model_copy(update={"stage_index": 1}, deep=True)],
            },
            deep=True,
        )
        return global_intent.model_copy(
            update={"control_semantics": scoped_semantics},
            deep=True,
        )


def _build_shared_control_context(global_intent: GlobalControlIntent) -> SharedControlContext:
    return SharedControlContext(main_intent=global_intent.model_copy(deep=True))


def _project_iea_routing_contract(global_intent: Dict[str, Any]) -> Dict[str, Any]:
    projected = project_global_intent_for_prompt(global_intent)
    return {
        key: projected[key]
        for key in (
            "supi",
            "requested_domains",
            "domain_evidence",
            "control_semantics",
            "retry_scope",
            "required_evidence",
            "forbidden_assumptions",
            "intent_encoding_guidance",
        )
        if projected.get(key) not in (None, "", [], {})
    }


def _project_iea_retry_delta(*, diagnosis: Dict[str, Any], feedback_context: str) -> Dict[str, Any]:
    diagnosis_payload = dict(diagnosis or {})
    compact_diagnosis = {
        key: diagnosis_payload[key]
        for key in (
            "root_cause_category",
            "reason_summary",
            "root_cause",
            "affected_flow_ids",
            "recommended_actions",
        )
        if diagnosis_payload.get(key) not in (None, "", [], {})
    }
    feedback = str(feedback_context or "").strip()
    if len(feedback) > 1800:
        feedback = "... [older feedback omitted]\n" + feedback[-1800:].lstrip()
    return {
        key: value
        for key, value in {"diagnosis": compact_diagnosis, "feedback_summary": feedback}.items()
        if value not in (None, "", [], {})
    }


def _qos_retry_constraints_from_feedback(
    previous_execution_feedback: Optional[Dict[str, Any]],
) -> list[QosRetryConstraint]:
    """Promote PDA assurance facts to typed, enforceable OSA retry bounds.

    This intentionally consumes only the structured assurance verdict, never
    the human-readable violation message.  The floor is the smallest integral
    PDB whose configured tolerance admits the observed latency.
    """

    root = dict(previous_execution_feedback or {})
    feedback = root.get("feedback_payload")
    if not isinstance(feedback, dict):
        return []
    failures = feedback.get("failures")
    if not isinstance(failures, list):
        return []

    strongest_by_flow: dict[str, QosRetryConstraint] = {}
    for failure in failures:
        if not isinstance(failure, dict) or str(failure.get("phase") or "").strip() != "assurance":
            continue
        verdict = failure.get("assurance_result")
        metrics = verdict.get("metrics") if isinstance(verdict, dict) else None
        if not isinstance(metrics, dict) or bool(metrics.get("latency_within_tolerance", True)):
            continue
        flow_id = str(failure.get("flow_id") or (verdict or {}).get("flow_id") or "").strip()
        observed = _positive_float(metrics.get("latency"))
        requirement = _positive_float(metrics.get("latency_requirement"))
        tolerance = _nonnegative_float(metrics.get("tolerance_ratio"))
        if not flow_id or observed is None or requirement is None or tolerance is None:
            continue
        allowed = requirement * (1.0 + tolerance)
        if observed <= allowed:
            continue
        constraint = QosRetryConstraint(
            flow_id=flow_id,
            observed_latency_ms=observed,
            allowed_latency_ms=allowed,
            minimum_target_latency_ms=max(1, math.ceil(observed / (1.0 + tolerance))),
            tolerance_ratio=tolerance,
            policy_id=str(failure.get("policy_id") or (verdict or {}).get("policy_id") or "").strip(),
        )
        previous = strongest_by_flow.get(flow_id)
        if previous is None or constraint.minimum_target_latency_ms > previous.minimum_target_latency_ms:
            strongest_by_flow[flow_id] = constraint
    return [strongest_by_flow[flow_id] for flow_id in sorted(strongest_by_flow)]


def _positive_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0.0 else None


def _nonnegative_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0.0 else None


__all__ = ["DefaultIntentContextBuilder", "IntentContextBuilder"]
