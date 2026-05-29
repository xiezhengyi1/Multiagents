from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..integrations.storage import create_session_context, get_latest_snapshot_metadata, update_session_context
from .main_control_support import ControlRoundResult


@dataclass
class OrchestratorLoopState:
    feedback_context: str = ""
    previous_diagnosis: Dict[str, Any] = field(default_factory=dict)
    previous_report_payload: Dict[str, Any] = field(default_factory=dict)
    previous_mediator_decision: Optional[Dict[str, Any]] = None
    previous_negotiation_request: Dict[str, Any] = field(default_factory=dict)
    previous_planning_blocker: Dict[str, Any] = field(default_factory=dict)
    previous_execution_reentry: Dict[str, Any] = field(default_factory=dict)
    latest_result: Optional[ControlRoundResult] = None
    round_traces: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = False


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
) -> None:
    state.round_traces.append(dict(trace_payload))
    if len(state.round_traces) > 3:
        state.round_traces = state.round_traces[-3:]
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


__all__ = ["OrchestratorLoopState", "append_round_trace", "finish_control_session", "start_control_session"]
