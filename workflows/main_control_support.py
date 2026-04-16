from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.policy_dispatch.contracts import FeedbackReport
from agents.tools.db_tool import get_latest_snapshot_data, get_latest_snapshot_metadata
from domain.collaboration import PlanningContext
from domain.control_plane import GlobalControlIntent


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


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
    round_count: int = 1
    retry_count: int = 0
    round_traces: List[Dict[str, Any]] = field(default_factory=list)


def build_main_context(
    snapshot_id: str,
    *,
    round_index: int,
    memory_context: str = "",
    feedback_context: str = "",
    previous_diagnosis: Optional[Dict[str, Any]] = None,
) -> str:
    snapshot = get_latest_snapshot_data() or {}
    payload = {
        "snapshot_id": snapshot_id,
        "round_index": round_index,
        "apps": len(snapshot.get("apps", [])),
        "slices": len(snapshot.get("slices", [])),
        "nodes": len(snapshot.get("nodes", [])),
        "mobility_ues": len(snapshot.get("mobility", [])),
        "memory_context": memory_context,
        "feedback_context": feedback_context,
        "previous_diagnosis": previous_diagnosis or {},
    }
    return json.dumps(payload, ensure_ascii=False)


def build_planning_context(
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
) -> PlanningContext:
    return PlanningContext(
        round_index=round_index,
        session_id=session_id,
        snapshot_id=snapshot_id,
        snapshot_metadata=get_latest_snapshot_metadata() or {},
        memory_context=memory_context,
        feedback_context=feedback_context,
        handoff_history=handoff_history or [],
        active_domains=[item.value for item in global_intent.requested_domains],
        main_agent_guidance=global_intent.prompt_injections.get("optimization_strategy", ""),
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


def build_feedback_context(
    previous_context: str,
    *,
    pda_feedback: Optional[Dict[str, Any]] = None,
    diagnosis: Optional[Dict[str, Any]] = None,
    domain_verdicts: Optional[List[Dict[str, Any]]] = None,
    mediator_decision: Optional[Dict[str, Any]] = None,
    round_index: int,
) -> str:
    blocks: List[str] = []
    if previous_context:
        blocks.append(previous_context)
    if pda_feedback:
        blocks.append(
            "[Round Feedback]\n"
            f"round_index: {round_index}\n"
            f"execution_status: {pda_feedback.get('execution_status', '')}\n"
            f"violation_details: {pda_feedback.get('violation_details', '')}\n"
            f"correction_suggestion: {pda_feedback.get('correction_suggestion', '')}\n"
            f"recommended_consumer: {pda_feedback.get('recommended_consumer', '')}\n"
        )
    if diagnosis:
        blocks.append(
            "[Diagnosis]\n"
            f"root_cause_category: {diagnosis.get('root_cause_category', '')}\n"
            f"root_cause: {diagnosis.get('root_cause', '')}\n"
            f"reason_summary: {diagnosis.get('reason_summary', '')}\n"
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
