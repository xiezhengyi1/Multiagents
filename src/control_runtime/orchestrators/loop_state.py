from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts import ControlRoundResult
from ..integrations.storage import create_session_context, get_latest_snapshot_metadata, update_session_context


@dataclass(frozen=True)
class ControlRoundSnapshot:
    round_index: int
    global_intent: Dict[str, Any]
    operation_intent: Dict[str, Any]
    policy_plan: Dict[str, Any]
    diagnosis: Dict[str, Any]
    feedback_added: str = ""
    domain_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    pda_feedback: Dict[str, Any] = field(default_factory=dict)
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    mediator_decision: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trace_payload(cls, trace_payload: Dict[str, Any], *, feedback_added: str = "") -> "ControlRoundSnapshot":
        return cls(
            round_index=int(trace_payload.get("round_index") or 0),
            global_intent=dict(trace_payload.get("global_intent") or {}),
            operation_intent=dict(trace_payload.get("operation_intent") or {}),
            policy_plan=dict(trace_payload.get("policy_plan") or {}),
            diagnosis=dict(trace_payload.get("diagnosis") or {}),
            feedback_added=str(feedback_added or ""),
            domain_verdicts=[
                dict(item)
                for item in (trace_payload.get("domain_verdicts") or [])
                if isinstance(item, dict)
            ],
            pda_feedback=dict(trace_payload.get("pda_feedback") or {}),
            qos_feedback=dict(trace_payload.get("qos_feedback") or {}),
            mobility_feedback=dict(trace_payload.get("mobility_feedback") or {}),
            mediator_decision=dict(trace_payload.get("mediator_decision") or {}),
            negotiation_request=dict(trace_payload.get("negotiation_request") or {}),
            planning_blocker=dict(trace_payload.get("planning_blocker") or {}),
            execution_reentry=dict(trace_payload.get("execution_reentry") or {}),
        )

    def to_trace_payload(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "global_intent": dict(self.global_intent),
            "operation_intent": dict(self.operation_intent),
            "policy_plan": dict(self.policy_plan),
            "domain_verdicts": [dict(item) for item in self.domain_verdicts],
            "pda_feedback": dict(self.pda_feedback),
            "qos_feedback": dict(self.qos_feedback),
            "mobility_feedback": dict(self.mobility_feedback),
            "diagnosis": dict(self.diagnosis),
            "mediator_decision": dict(self.mediator_decision),
            "negotiation_request": dict(self.negotiation_request),
            "planning_blocker": dict(self.planning_blocker),
            "execution_reentry": dict(self.execution_reentry),
        }


@dataclass
class OrchestratorLoopState:
    rounds: List[ControlRoundSnapshot] = field(default_factory=list)
    completed: bool = False
    latest_result: Optional[ControlRoundResult] = None

    @property
    def previous_diagnosis(self) -> Dict[str, Any]:
        return dict(self.rounds[-1].diagnosis) if self.rounds else {}

    @property
    def previous_report_payload(self) -> Dict[str, Any]:
        return dict(self.rounds[-1].pda_feedback) if self.rounds else {}

    @property
    def previous_mediator_decision(self) -> Optional[Dict[str, Any]]:
        if not self.rounds or not self.rounds[-1].mediator_decision:
            return None
        return dict(self.rounds[-1].mediator_decision)

    @property
    def previous_negotiation_request(self) -> Dict[str, Any]:
        return dict(self.rounds[-1].negotiation_request) if self.rounds else {}

    @property
    def previous_planning_blocker(self) -> Dict[str, Any]:
        return dict(self.rounds[-1].planning_blocker) if self.rounds else {}

    @property
    def previous_execution_reentry(self) -> Dict[str, Any]:
        return dict(self.rounds[-1].execution_reentry) if self.rounds else {}

    @property
    def feedback_context(self) -> str:
        return "\n".join(snap.feedback_added.strip() for snap in self.rounds if snap.feedback_added.strip())

    @property
    def round_traces(self) -> List[Dict[str, Any]]:
        return [snap.to_trace_payload() for snap in self.rounds]


def start_control_session(*, step_name: str, user_input: str, snapshot_id: str = "") -> Tuple[str, str]:
    bound_snapshot_id = str(snapshot_id or "").strip()
    if not bound_snapshot_id:
        snapshot_metadata = get_latest_snapshot_metadata() or {}
        bound_snapshot_id = str(snapshot_metadata.get("snapshot_id") or "").strip()
    if not bound_snapshot_id:
        raise RuntimeError("failed to bind control session to a graph snapshot_id")
    session_id = create_session_context(current_step=step_name, intent_data={"raw_input": user_input}) or ""
    if not session_id:
        raise RuntimeError("failed to create session_context")
    update_session_context(session_id, current_step=step_name, current_snapshot_id=bound_snapshot_id, status="active")
    return session_id, bound_snapshot_id


def append_round_trace(
    state: OrchestratorLoopState,
    *,
    trace_payload: Dict[str, Any],
    feedback_added: str = "",
) -> None:
    snapshot = ControlRoundSnapshot.from_trace_payload(trace_payload, feedback_added=feedback_added)
    state.rounds.append(snapshot)
    if len(state.rounds) > 3:
        del state.rounds[:-3]
    if state.latest_result is not None:
        state.latest_result.round_traces = state.round_traces


def finish_control_session(
    *,
    session_id: str,
    snapshot_id: str,
    state: OrchestratorLoopState,
) -> None:
    update_session_context(
        session_id,
        current_step="completed" if state.completed else "failed",
        current_snapshot_id=str(
            (
                state.latest_result.qos_feedback.get("committed_snapshot_id")
                if state.latest_result is not None and isinstance(state.latest_result.qos_feedback, dict)
                else ""
            )
            or snapshot_id
        ).strip(),
        status="completed" if state.completed else "failed",
    )


__all__ = [
    "ControlRoundSnapshot",
    "OrchestratorLoopState",
    "append_round_trace",
    "finish_control_session",
    "start_control_session",
]
