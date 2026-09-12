from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from control_runtime.graph import EventEmission, RepairBudget
from shared.contracts.coordination import (
    CoordinationDecision,
    RepairRequestPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.runtime.journal import EpisodeJournal


class ValidationActor:
    """Common event emission for deterministic contract validators."""

    def __init__(self, *, journal: EpisodeJournal, repair_budget: RepairBudget) -> None:
        self.journal = journal
        self.repair_budget = repair_budget

    def accept(
        self,
        *,
        current: Any,
        target_agent: str,
        notes: Sequence[str],
        resolved_actor: str,
    ) -> EventEmission:
        self.repair_budget.resolved(resolved_actor)
        return EventEmission(
            decision=CoordinationDecision[ValidationResultPayload](
                target_agent=target_agent,
                message_type="validation_result",
                payload=ValidationResultPayload(notes=list(notes)),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )

    def repair(
        self,
        *,
        current: Any,
        issues: list[ValidationIssue],
        repair_owner: str,
        failed_trace_id: str | None,
        invalidated_artifact_ids: tuple[str, ...],
    ) -> EventEmission:
        route = self.repair_budget.request(owner_actor=repair_owner, issues=issues)
        return EventEmission(
            decision=CoordinationDecision[RepairRequestPayload](
                target_agent=route.target_actor,
                message_type="repair_request",
                payload=RepairRequestPayload(
                    issues=list(route.issues),
                    attempt=route.attempt,
                    max_attempts=route.max_attempts,
                    failed_trace_id=failed_trace_id,
                ),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,)
            if not invalidated_artifact_ids
            else (),
            invalidated_artifact_ids=invalidated_artifact_ids,
        )


__all__ = ["ValidationActor"]
