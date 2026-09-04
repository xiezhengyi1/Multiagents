from __future__ import annotations

from control_runtime.execution import ExecutionController
from control_runtime.graph import EpisodeGraphState, EventEmission
from control_runtime.simulation import (
    ExecutionAction,
    ExecutionFeedback,
    SimulatorAdapter,
)
from shared.contracts.coordination import CoordinationDecision
from shared.runtime.journal import EpisodeJournal

from .common import find_latest_event_payload
from .diagnostics import ConflictAssessment


class ExecutionDispatcher:
    actor_id = "execution_dispatcher"

    def __init__(
        self, *, journal: EpisodeJournal, simulator_adapter: SimulatorAdapter
    ) -> None:
        self.journal = journal
        self.controller = ExecutionController(backend=simulator_adapter)

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ConflictAssessment):
            raise TypeError("ExecutionDispatcher requires ConflictAssessment")
        if current.payload.status != "approved":
            raise ValueError("ExecutionDispatcher requires an approved assessment")
        action = find_latest_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=ExecutionAction,
        )
        feedback = self.controller.execute(action).feedback
        return EventEmission(
            decision=CoordinationDecision[ExecutionFeedback](
                target_agent="assurance_diagnoser",
                message_type="execution_feedback",
                payload=feedback,
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
            round_delta=1,
        )


__all__ = ["ExecutionDispatcher"]
