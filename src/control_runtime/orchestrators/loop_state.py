from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts import ControlRoundResult, ControlRoundTrace
from ..integrations.storage import create_session_context, get_latest_snapshot_metadata, update_session_context


@dataclass(frozen=True)
class ControlRoundSnapshot:
    trace: ControlRoundTrace
    feedback_added: str = ""


@dataclass
class OrchestratorLoopState:
    rounds: List[ControlRoundSnapshot] = field(default_factory=list)
    completed: bool = False
    latest_result: Optional[ControlRoundResult] = None

    @property
    def previous_diagnosis(self) -> Dict[str, Any]:
        return deepcopy(self.rounds[-1].trace.diagnosis) if self.rounds else {}

    @property
    def previous_report_payload(self) -> Dict[str, Any]:
        if not self.rounds:
            return {}
        trace = self.rounds[-1].trace
        if trace.dispatch_report is not None:
            return trace.dispatch_report.model_dump(mode="json")
        if trace.orchestration_failure is not None:
            return trace.orchestration_failure.model_dump(mode="json")
        return {}

    @property
    def previous_mediator_decision(self) -> Optional[Dict[str, Any]]:
        if not self.rounds or not self.rounds[-1].trace.mediator_decision:
            return None
        return self.rounds[-1].trace.mediator_decision.model_dump(mode="json")

    @property
    def previous_negotiation_request(self) -> Dict[str, Any]:
        if not self.rounds or self.rounds[-1].trace.negotiation_request is None:
            return {}
        return self.rounds[-1].trace.negotiation_request.model_dump(mode="json")

    @property
    def previous_planning_blocker(self) -> Dict[str, Any]:
        if not self.rounds or self.rounds[-1].trace.planning_blocker is None:
            return {}
        return self.rounds[-1].trace.planning_blocker.model_dump(mode="json")

    @property
    def previous_execution_reentry(self) -> Dict[str, Any]:
        if not self.rounds or self.rounds[-1].trace.execution_reentry is None:
            return {}
        return self.rounds[-1].trace.execution_reentry.model_dump(mode="json")

    @property
    def round_traces(self) -> List[Dict[str, Any]]:
        return [snap.trace.to_payload() for snap in self.rounds]


def start_control_session(*, step_name: str, user_input: str, snapshot_id: str = "") -> Tuple[str, str]:
    bound_snapshot_id = str(snapshot_id or "").strip()
    if not bound_snapshot_id:
        snapshot_metadata = get_latest_snapshot_metadata() or {}
        bound_snapshot_id = str(snapshot_metadata.get("snapshot_id") or "").strip()
    if not bound_snapshot_id:
        raise RuntimeError("failed to bind control session to a graph snapshot_id")
    session_id = create_session_context(current_stage=step_name, intent_data={"raw_input": user_input}) or ""
    if not session_id:
        raise RuntimeError("failed to create session_context")
    update_session_context(session_id, current_stage=step_name, current_snapshot_id=bound_snapshot_id, status="active")
    return session_id, bound_snapshot_id


def append_round_trace(
    state: OrchestratorLoopState,
    *,
    trace: ControlRoundTrace,
    feedback_added: str = "",
) -> None:
    snapshot = ControlRoundSnapshot(trace=trace, feedback_added=str(feedback_added or ""))
    state.rounds.append(snapshot)
    if state.latest_result is not None:
        state.latest_result = state.latest_result.model_copy(
            update={
                "round_traces": state.round_traces,
                "single_agent_details": [
                    deepcopy(item.trace.single_agent_detail)
                    for item in state.rounds
                    if item.trace.single_agent_detail
                ],
            }
        )


def finish_control_session(
    *,
    session_id: str,
    snapshot_id: str,
    state: OrchestratorLoopState,
) -> None:
    update_session_context(
        session_id,
        current_stage="completed" if state.completed else "failed",
        current_snapshot_id=str(snapshot_id).strip(),
        status="completed" if state.completed else "failed",
    )


__all__ = [
    "ControlRoundSnapshot",
    "OrchestratorLoopState",
    "append_round_trace",
    "finish_control_session",
    "start_control_session",
]
