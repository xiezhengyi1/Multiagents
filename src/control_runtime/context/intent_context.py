"""Stable context-construction boundary for Main -> IEA -> OSA handoffs.

The orchestrator depends on this interface rather than on individual context
fields.  Alternative context policies can therefore be developed and tested
without changing orchestration or agent implementations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol

from ..domain.collaboration import InitialIntentContext, PlanningContext, SharedControlContext
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
        active_domains: Optional[List[str]] = None,
        round_index: int,
        memory_context: str = "",
        feedback_context: str = "",
        handoff_history: Optional[List[Dict[str, Any]]] = None,
        revision_requests: Optional[List[Dict[str, Any]]] = None,
        unified_constraints: Optional[Dict[str, Any]] = None,
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
            memory_context=memory_context,
            shared_context=_build_shared_control_context(global_intent),
            feedback_context=feedback_context,
            handoff_history=list(handoff_history or [])[-2:],
            active_domains=list(active_domains or [item.value for item in global_intent.requested_domains]),
            retry_scope=(
                global_intent.retry_scope.value
                if getattr(global_intent, "retry_scope", None) is not None and hasattr(global_intent.retry_scope, "value")
                else str(getattr(global_intent, "retry_scope", "") or "").strip()
            ),
            revision_requests=list(revision_requests or []),
            unified_constraints=dict(unified_constraints or {}),
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
    raw_input = str(global_intent.raw_input or "").strip()
    semantics = global_intent.control_semantics.model_dump(mode="json")
    constraints: List[Dict[str, Any]] = []
    if _has_qos_slice_migration_goal(global_intent):
        constraints.append(
            {
                "type": "qos_slice_migration",
                "required": True,
                "no_op_allowed": False,
                "source": "main_control",
                "target_slice_policy": {
                    "exclude_current": True,
                    "preference": _infer_slice_preference(raw_input, semantics),
                },
                "semantic_cues": _slice_migration_cues(raw_input, semantics),
            }
        )
    target_supis: List[str] = []
    target_names: List[str] = []
    if str(global_intent.supi or "").strip():
        target_supis.append(str(global_intent.supi).strip())
    for stage in semantics.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        for target in stage.get("targets") or []:
            if not isinstance(target, dict):
                continue
            target_supi = str(target.get("supi") or "").strip()
            target_name = str(target.get("semantic_name") or "").strip()
            if target_supi and target_supi not in target_supis:
                target_supis.append(target_supi)
            if target_name and target_name not in target_names:
                target_names.append(target_name)
    return SharedControlContext(
        initial_intent=InitialIntentContext(
            request_summary=raw_input,
            requested_domains=[item.value for item in global_intent.requested_domains],
            target_supis=target_supis,
            target_names=target_names,
            objective_profile=global_intent.objective_profile.model_dump(mode="json"),
            required_evidence=list(global_intent.required_evidence or []),
            forbidden_assumptions=list(global_intent.forbidden_assumptions or []),
            global_constraints=constraints,
        )
    )


def _has_qos_slice_migration_goal(global_intent: GlobalControlIntent) -> bool:
    domains = {str(item.value if hasattr(item, "value") else item).strip().lower() for item in global_intent.requested_domains}
    if "qos" not in domains:
        return False
    text = " ".join(
        [
            str(global_intent.raw_input or ""),
            str(global_intent.routing_rationale or ""),
            json.dumps(global_intent.control_semantics.model_dump(mode="json"), ensure_ascii=False),
        ]
    ).lower()
    return any(token in text for token in ("迁移", "迁出", "切换", "换到", "migrate", "migration", "switch", "move away"))


def _infer_slice_preference(raw_input: str, semantics: Dict[str, Any]) -> str:
    text = f"{raw_input} {json.dumps(semantics, ensure_ascii=False)}".lower()
    if any(token in text for token in ("低时延", "更低时延", "lower latency", "lower-latency", "latency")):
        return "lower_latency"
    if any(token in text for token in ("高吞吐", "throughput", "bandwidth")):
        return "higher_throughput"
    return "runtime_feasible"


def _slice_migration_cues(raw_input: str, semantics: Dict[str, Any]) -> List[str]:
    text = f"{raw_input} {json.dumps(semantics, ensure_ascii=False)}".lower()
    return [
        token
        for token in ("迁移", "迁出", "切换", "换到", "migrate", "migration", "switch", "move away")
        if token in text
    ]


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


__all__ = ["DefaultIntentContextBuilder", "IntentContextBuilder"]
