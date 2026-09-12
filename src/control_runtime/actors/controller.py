from __future__ import annotations

from typing import Protocol

from pydantic import Field

from control_runtime.graph import EpisodeGraphState, EventEmission, RepairBudget
from control_runtime.simulation import ExecutionFeedback
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import (
    CoordinationDecision,
    RepairRequestPayload,
    TerminalPayload,
    ValidationIssue,
)

from .common import PlanningBlocker


class OutcomeEvaluation(ContractModel):
    achieved: bool
    reason: NonEmptyStr
    reward_components: dict[NonEmptyStr, float] = Field(min_length=1)


class OutcomeEvaluator(Protocol):
    def evaluate(self, feedback: ExecutionFeedback) -> OutcomeEvaluation: ...


class EpisodeController:
    actor_id = "episode_controller"

    def __init__(
        self,
        *,
        repair_budget: RepairBudget,
        budget_exhausted_reward_components: dict[str, float],
    ) -> None:
        if not budget_exhausted_reward_components:
            raise ValueError("budget_exhausted_reward_components cannot be empty")
        self.repair_budget = repair_budget
        self.budget_exhausted_reward_components = dict(
            budget_exhausted_reward_components
        )
        self._execution_failure_keys: set[tuple[str, tuple[str, ...]]] = set()

    @staticmethod
    def _terminal(
        *,
        current: object,
        outcome: str,
        resolution: str,
        reason_code: str,
        reason: str,
        reward_components: dict[str, float],
    ) -> EventEmission:
        event_id = current.event_id
        return EventEmission(
            decision=CoordinationDecision[TerminalPayload](
                target_agent="episode_controller",
                message_type="terminal",
                payload=TerminalPayload(
                    outcome=outcome,
                    resolution=resolution,
                    reason_code=reason_code,
                    reason=reason,
                    reward_total=sum(reward_components.values()),
                    reward_components=reward_components,
                ),
            ),
            in_reply_to=event_id,
            input_artifact_ids=(event_id,),
            invalidated_artifact_ids=(),
        )

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        from .diagnostics import (
            AssuranceDiagnosis,
            ConflictAssessment,
            conflict_issues,
        )

        current = state["current_event"]
        if isinstance(current.payload, PlanningBlocker):
            if current.payload.resolution == "infeasible":
                return self._terminal(
                    current=current,
                    outcome="success",
                    resolution="infeasible",
                    reason_code=current.payload.reason_code,
                    reason=current.payload.rationale,
                    reward_components={"infeasible_request": 1.0},
                )
            details = [
                current.payload.rationale,
                *current.payload.missing_evidence,
                *current.payload.blocked_targets,
                *current.payload.upstream_requests,
                *current.payload.planner_conflicts,
            ]
            issues = [
                ValidationIssue(code="planning_blocked", message="; ".join(details))
            ]
            repair_owner = current.payload.repair_owner
            reward_components = self.budget_exhausted_reward_components
        elif isinstance(current.payload, RepairRequestPayload):
            reason = "; ".join(issue.message for issue in current.payload.issues)
            return self._terminal(
                current=current,
                outcome="failure",
                resolution="failed",
                reason_code="repair_budget_exhausted",
                reason=reason,
                reward_components=self.budget_exhausted_reward_components,
            )
        elif isinstance(current.payload, ConflictAssessment):
            if current.payload.status != "blocked":
                raise ValueError(
                    "approved conflict assessment cannot target controller"
                )
            issues = conflict_issues(current.payload)
            repair_owner = current.payload.revision_directives[0].owner_agent
            reward_components = self.budget_exhausted_reward_components
        elif isinstance(current.payload, AssuranceDiagnosis):
            if current.payload.status == "satisfied":
                return self._terminal(
                    current=current,
                    outcome="success",
                    resolution="executed",
                    reason_code="objective_achieved",
                    reason=current.payload.reason_summary,
                    reward_components=current.payload.reward_components,
                )
            if current.payload.status == "execution_failed":
                failure_key = (
                    current.payload.reason_summary,
                    tuple(sorted(current.payload.affected_operation_ids)),
                )
                if failure_key in self._execution_failure_keys:
                    return self._terminal(
                        current=current,
                        outcome="failure",
                        resolution="failed",
                        reason_code="repeated_execution_failure",
                        reason=current.payload.reason_summary,
                        reward_components=current.payload.reward_components,
                    )
                self._execution_failure_keys.add(failure_key)
            issues = [
                ValidationIssue(
                    code=current.payload.root_cause_category,
                    message=(
                        current.payload.reason_summary
                        + "; "
                        + "; ".join(current.payload.recommended_actions)
                    ),
                )
            ]
            repair_owner = current.payload.repair_owner
            reward_components = current.payload.reward_components
        else:
            raise TypeError(
                "EpisodeController requires PlanningBlocker, RepairRequestPayload, "
                "ConflictAssessment, or AssuranceDiagnosis"
            )

        route = self.repair_budget.request(
            owner_actor=repair_owner,
            issues=issues,
        )
        if route.target_actor == self.actor_id:
            return self._terminal(
                current=current,
                outcome="failure",
                resolution="failed",
                reason_code="repair_budget_exhausted",
                reason="; ".join(issue.message for issue in route.issues),
                reward_components=reward_components,
            )
        return EventEmission(
            decision=CoordinationDecision[RepairRequestPayload](
                target_agent=route.target_actor,
                message_type="repair_request",
                payload=RepairRequestPayload(
                    issues=list(route.issues),
                    attempt=route.attempt,
                    max_attempts=route.max_attempts,
                ),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )


__all__ = ["EpisodeController", "OutcomeEvaluation", "OutcomeEvaluator"]
