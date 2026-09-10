from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from control_runtime.actors import (
    AssuranceDiagnoser,
    AssuranceDiagnosis,
    ConflictAssessment,
    EpisodeController,
    OutcomeEvaluation,
    PlanningBlocker,
    assess_policy_conflicts,
)
from control_runtime.graph import RepairBudget, RepairLimits
from control_runtime.simulation import (
    EnvironmentSnapshot,
    ExecutionAction,
    ExecutionFeedback,
    ExecutionOperation,
    OperationReceipt,
)
from model.PcfAmPolicyControl import (
    Ambr,
    PcfAmPolicyControlPolicyAssociation,
    PcfAmPolicyControlPolicyAssociationRequest,
    Snssai,
)
from model.SmPolicyDecision import QosData, SmPolicyDecision
from shared.contracts.coordination import TerminalPayload
from shared.runtime.journal import EpisodeJournal
from training import EpisodeRecords, project_episode_metrics


def _sm_payload(*, rate_mbps: float = 10.0) -> SmPolicyDecision:
    return SmPolicyDecision(
        qosDecs={
            "qos-1": QosData(
                qosId="qos-1",
                maxbrUl=f"{rate_mbps:g} Mbps",
                maxbrDl=f"{rate_mbps:g} Mbps",
            )
        }
    )


def _operation(
    *, operation_id: str, resource_id: str, rate_mbps: float = 10.0
) -> ExecutionOperation:
    return ExecutionOperation(
        operation_id=operation_id,
        domain="qos",
        operation="update",
        resource_type="SmPolicyDecision",
        resource_id=resource_id,
        payload=_sm_payload(rate_mbps=rate_mbps),
    )


def test_policy_mediator_blocks_duplicate_targets_and_ambr_conflicts() -> None:
    duplicate = assess_policy_conflicts(
        ExecutionAction(
            operations=[
                _operation(operation_id="op-1", resource_id="sm-1"),
                _operation(operation_id="op-2", resource_id="sm-1"),
            ]
        )
    )
    assert duplicate.status == "blocked"
    assert [item.code for item in duplicate.conflicts] == ["duplicate_resource_target"]

    request = PcfAmPolicyControlPolicyAssociationRequest(
        notificationUri="https://pcf.example/notify",
        supi="imsi-12345",
        rfsp=1,
        ueAmbr=Ambr(uplink="5 Mbps", downlink="5 Mbps"),
        allowedSnssais=[Snssai(sst=1, sd="000001")],
        targetSnssais=[Snssai(sst=1, sd="000001")],
        suppFeat="1",
    )
    cross_domain = assess_policy_conflicts(
        ExecutionAction(
            operations=[
                ExecutionOperation(
                    operation_id="op-am",
                    domain="mobility",
                    operation="update",
                    resource_type="PcfAmPolicyControlPolicyAssociation",
                    resource_id="am-1",
                    payload=PcfAmPolicyControlPolicyAssociation(
                        request=request,
                        rfsp=1,
                        ueAmbr=request.ueAmbr,
                        suppFeat="1",
                    ),
                ),
                _operation(operation_id="op-sm", resource_id="sm-1", rate_mbps=10.0),
            ]
        )
    )
    assert cross_domain.status == "blocked"
    assert {item.code for item in cross_domain.conflicts} == {
        "ambr_qos_uplink_inconsistency",
        "ambr_qos_downlink_inconsistency",
    }
    assert all(
        directive.owner_agent == "ppa" for directive in cross_domain.revision_directives
    )


class _FailedOutcomeEvaluator:
    def evaluate(self, feedback: ExecutionFeedback) -> OutcomeEvaluation:
        return OutcomeEvaluation(
            achieved=False,
            reason="execution did not produce a valid objective state",
            reward_components={"objective": -1.0},
        )


def test_assurance_diagnosis_routes_repair_and_projects_metrics(
    tmp_path: Path,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="diagnosis-episode")
    action = ExecutionAction(
        operations=[_operation(operation_id="op-1", resource_id="sm-1")]
    )
    action_event = journal.append_event(
        round_index=0,
        source_agent="ppa_compiler",
        target_agent="policy_mediator",
        message_type="handoff",
        payload=action,
    )
    assessment_event = journal.append_event(
        round_index=0,
        source_agent="policy_mediator",
        target_agent="execution_dispatcher",
        message_type="diagnosis",
        in_reply_to=action_event.event_id,
        input_artifact_ids=[action_event.event_id],
        payload=ConflictAssessment(
            status="approved", conflicts=[], revision_directives=[]
        ),
    )
    feedback_event = journal.append_event(
        round_index=1,
        source_agent="execution_dispatcher",
        target_agent="assurance_diagnoser",
        message_type="execution_feedback",
        in_reply_to=assessment_event.event_id,
        input_artifact_ids=[assessment_event.event_id],
        payload=ExecutionFeedback(
            status="failed",
            receipts=[
                OperationReceipt(
                    operation_id="op-1",
                    status="failed",
                    error="policy rejected",
                )
            ],
            snapshot=EnvironmentSnapshot(
                snapshot_id="snapshot-2",
                observed_at=datetime.now(UTC),
                entities=[],
                relations=[],
                metrics=[],
            ),
            objective_metrics={"sla_satisfaction": 0.0},
        ),
    )
    emission = AssuranceDiagnoser(
        journal=journal,
        outcome_evaluator=_FailedOutcomeEvaluator(),
        repair_owner="ppa",
    )({"current_event": feedback_event, "pending_model_failure": None})
    diagnosis = emission.decision.payload
    assert isinstance(diagnosis, AssuranceDiagnosis)
    assert diagnosis.status == "execution_failed"
    assert diagnosis.affected_operation_ids == ["op-1"]
    diagnosis_event = journal.append_event(
        round_index=1,
        source_agent="assurance_diagnoser",
        target_agent="episode_controller",
        message_type="diagnosis",
        in_reply_to=emission.in_reply_to,
        input_artifact_ids=emission.input_artifact_ids,
        payload=diagnosis,
    )

    controller = EpisodeController(
        repair_budget=RepairBudget(
            RepairLimits(
                local_limits={"ppa": 2},
                upstream={"ppa": "sba"},
                total_limit=2,
                controller_actor="episode_controller",
            )
        ),
        budget_exhausted_reward_components={"repair_budget": -1.0},
    )
    repair = controller(
        {"current_event": diagnosis_event, "pending_model_failure": None}
    )
    assert repair.decision.message_type == "repair_request"
    assert repair.decision.target_agent == "ppa"
    repair_event = journal.append_event(
        round_index=1,
        source_agent="episode_controller",
        target_agent="ppa",
        message_type="repair_request",
        in_reply_to=repair.in_reply_to,
        input_artifact_ids=repair.input_artifact_ids,
        payload=repair.decision.payload,
    )
    journal.append_event(
        round_index=1,
        source_agent="episode_controller",
        target_agent="episode_controller",
        message_type="terminal",
        in_reply_to=repair_event.event_id,
        input_artifact_ids=[repair_event.event_id],
        payload=TerminalPayload(
            outcome="failure",
            resolution="failed",
            reason_code="test_terminal",
            reason="test terminal",
            reward_total=-1.0,
            reward_components={"objective": -1.0},
        ),
    )

    metrics = project_episode_metrics(EpisodeRecords.load(journal.directory))
    assert metrics.round_count == 2
    assert metrics.execution_attempt_count == 1
    assert metrics.execution_failure_count == 1
    assert metrics.repair_count == 1
    assert not metrics.first_attempt_success


def test_partial_plan_reenters_the_responsible_agent(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="planning-reentry")
    blocker = journal.append_event(
        round_index=0,
        source_agent="ppa_validator",
        target_agent="episode_controller",
        message_type="handoff",
        payload=PlanningBlocker(
            rationale="fresh binding required",
            resolution="repairable",
            reason_code="upstream_binding_required",
            missing_evidence=["flow binding"],
            blocked_targets=["flow-1"],
            upstream_requests=["rebind flow-1"],
            planner_conflicts=[],
            repair_owner="sba",
        ),
    )
    controller = EpisodeController(
        repair_budget=RepairBudget(
            RepairLimits(
                local_limits={"sba": 2},
                upstream={"sba": "main"},
                total_limit=2,
                controller_actor="episode_controller",
            )
        ),
        budget_exhausted_reward_components={"repair_budget": -1.0},
    )

    result = controller({"current_event": blocker, "pending_model_failure": None})

    assert result.decision.message_type == "repair_request"
    assert result.decision.target_agent == "sba"
    assert result.decision.payload.issues[0].code == "planning_blocked"


def test_certified_infeasible_plan_terminates_without_repair(tmp_path: Path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="infeasible-plan")
    blocker = journal.append_event(
        round_index=0,
        source_agent="ppa_validator",
        target_agent="episode_controller",
        message_type="handoff",
        payload=PlanningBlocker(
            rationale="subscription does not authorize a different slice",
            resolution="infeasible",
            reason_code="blocked_by_subscription_entitlement",
            missing_evidence=[],
            blocked_targets=["slice_migration_authorization"],
            upstream_requests=[],
            planner_conflicts=["blocked_by_subscription_entitlement"],
            repair_owner="ppa",
        ),
    )
    controller = EpisodeController(
        repair_budget=RepairBudget(
            RepairLimits(
                local_limits={"ppa": 2},
                upstream={"ppa": "sba"},
                total_limit=2,
                controller_actor="episode_controller",
            )
        ),
        budget_exhausted_reward_components={"repair_budget": -1.0},
    )

    result = controller({"current_event": blocker, "pending_model_failure": None})

    assert result.decision.message_type == "terminal"
    assert result.decision.payload.outcome == "success"
    assert result.decision.payload.resolution == "infeasible"
    assert result.decision.payload.reason_code == "blocked_by_subscription_entitlement"
