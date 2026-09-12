from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from control_runtime.graph import EpisodeGraphState, EventEmission
from control_runtime.simulation import ExecutionAction, ExecutionFeedback
from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import (
    CoordinationDecision,
    DiagnosisPayload,
    ValidationIssue,
)
from shared.runtime.journal import EpisodeJournal

from .common import find_latest_event_payload
from .controller import OutcomeEvaluator

DiagnosisOwner = Literal["main", "sba", "ppa", "single_control"]
PolicyDomain = Literal["qos", "mobility"]


class PolicyConflict(ContractModel):
    code: Literal[
        "duplicate_resource_target",
        "snssai_alignment_conflict",
        "ambr_qos_uplink_inconsistency",
        "ambr_qos_downlink_inconsistency",
    ]
    domains: list[PolicyDomain] = Field(min_length=1)
    operation_ids: list[NonEmptyStr] = Field(min_length=1)
    resource_ids: list[NonEmptyStr] = Field(min_length=1)
    detail: NonEmptyStr

    @field_validator("domains", "operation_ids", "resource_ids")
    @classmethod
    def validate_unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("policy conflict references must be unique")
        return values


class RevisionDirective(ContractModel):
    owner_agent: DiagnosisOwner
    code: NonEmptyStr
    instruction: NonEmptyStr
    affected_operation_ids: list[NonEmptyStr] = Field(min_length=1)

    @field_validator("affected_operation_ids")
    @classmethod
    def validate_unique_operations(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("revision operation ids must be unique")
        return values


class ConflictAssessment(DiagnosisPayload):
    status: Literal["approved", "blocked"]
    conflicts: list[PolicyConflict]
    revision_directives: list[RevisionDirective]

    @model_validator(mode="after")
    def validate_status(self) -> ConflictAssessment:
        if self.status == "approved":
            if self.conflicts or self.revision_directives:
                raise ValueError("approved assessment cannot contain conflicts")
        elif not self.conflicts or not self.revision_directives:
            raise ValueError("blocked assessment requires conflicts and revisions")
        return self


class AssuranceDiagnosis(DiagnosisPayload):
    status: Literal["satisfied", "objective_violated", "execution_failed"]
    root_cause_category: Literal[
        "objective_achieved", "objective_violation", "execution_failure"
    ]
    reason_summary: NonEmptyStr
    affected_operation_ids: list[NonEmptyStr]
    affected_domains: list[PolicyDomain]
    recommended_actions: list[NonEmptyStr]
    repair_owner: DiagnosisOwner
    reward_components: dict[NonEmptyStr, float] = Field(min_length=1)

    @field_validator(
        "affected_operation_ids", "affected_domains", "recommended_actions"
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("assurance diagnosis lists must be unique")
        return values

    @model_validator(mode="after")
    def validate_diagnosis(self) -> AssuranceDiagnosis:
        expected_category = {
            "satisfied": "objective_achieved",
            "objective_violated": "objective_violation",
            "execution_failed": "execution_failure",
        }[self.status]
        if self.root_cause_category != expected_category:
            raise ValueError(
                f"{self.status} diagnosis requires {expected_category} category"
            )
        if self.status == "satisfied":
            if self.affected_operation_ids or self.affected_domains:
                raise ValueError(
                    "satisfied diagnosis cannot identify affected operations"
                )
            if self.recommended_actions:
                raise ValueError("satisfied diagnosis cannot recommend repairs")
        elif not self.affected_operation_ids or not self.recommended_actions:
            raise ValueError(
                "failed diagnosis requires affected operations and actions"
            )
        return self


def _mbps(value: str) -> float:
    amount, unit = value.split()
    if unit != "Mbps":
        raise ValueError(f"unsupported bitrate unit: {unit}")
    return float(amount)


def assess_policy_conflicts(action: ExecutionAction) -> ConflictAssessment:
    conflicts: list[PolicyConflict] = []
    operations_by_resource: dict[tuple[str, str], list[str]] = {}
    for operation in action.operations:
        key = (operation.resource_type, operation.resource_id)
        operations_by_resource.setdefault(key, []).append(operation.operation_id)
    for (resource_type, resource_id), operation_ids in operations_by_resource.items():
        if len(operation_ids) > 1:
            domains = list(
                dict.fromkeys(
                    operation.domain
                    for operation in action.operations
                    if operation.operation_id in operation_ids
                )
            )
            conflicts.append(
                PolicyConflict(
                    code="duplicate_resource_target",
                    domains=domains,
                    operation_ids=operation_ids,
                    resource_ids=[resource_id],
                    detail=(
                        f"Multiple operations target {resource_type} resource "
                        f"{resource_id} in one atomic action"
                    ),
                )
            )

    am_operations = [
        operation
        for operation in action.operations
        if isinstance(operation.payload, PcfAmPolicyControlPolicyAssociation)
    ]
    sm_operations = [
        operation
        for operation in action.operations
        if isinstance(operation.payload, SmPolicyDecision)
    ]
    ursp_operations = [
        operation
        for operation in action.operations
        if isinstance(operation.payload, UrspRuleRequest)
    ]

    if am_operations:
        am_operation = am_operations[0]
        association = am_operation.payload
        if association.request is None:
            raise ValueError("compiled AM policy requires its association request")
        allowed = {
            (item.sst, item.sd) for item in association.request.allowedSnssais or []
        }
        for operation in ursp_operations:
            requested = {
                (route.snssai.sst, route.snssai.sd)
                for route in operation.payload.routeSelParamSets
                if route.snssai is not None
            }
            unknown = sorted(requested - allowed)
            if unknown:
                conflicts.append(
                    PolicyConflict(
                        code="snssai_alignment_conflict",
                        domains=["qos", "mobility"],
                        operation_ids=[
                            am_operation.operation_id,
                            operation.operation_id,
                        ],
                        resource_ids=[
                            am_operation.resource_id,
                            operation.resource_id,
                        ],
                        detail=(
                            "URSP selects S-NSSAIs outside the AM allowed set: "
                            + ", ".join(f"{sst}:{sd or ''}" for sst, sd in unknown)
                        ),
                    )
                )

        if association.ueAmbr is not None and sm_operations:
            requested_ul = 0.0
            requested_dl = 0.0
            for operation in sm_operations:
                for qos in (operation.payload.qosDecs or {}).values():
                    if qos.maxbrUl is not None:
                        requested_ul += _mbps(qos.maxbrUl)
                    if qos.maxbrDl is not None:
                        requested_dl += _mbps(qos.maxbrDl)
            ambr_ul = _mbps(association.ueAmbr.uplink)
            ambr_dl = _mbps(association.ueAmbr.downlink)
            operation_ids = [
                am_operation.operation_id,
                *(operation.operation_id for operation in sm_operations),
            ]
            resource_ids = [
                am_operation.resource_id,
                *(operation.resource_id for operation in sm_operations),
            ]
            if requested_ul > ambr_ul:
                conflicts.append(
                    PolicyConflict(
                        code="ambr_qos_uplink_inconsistency",
                        domains=["qos", "mobility"],
                        operation_ids=operation_ids,
                        resource_ids=resource_ids,
                        detail=(
                            f"Aggregate SM uplink MBR {requested_ul:g} Mbps exceeds "
                            f"UE-AMBR {ambr_ul:g} Mbps"
                        ),
                    )
                )
            if requested_dl > ambr_dl:
                conflicts.append(
                    PolicyConflict(
                        code="ambr_qos_downlink_inconsistency",
                        domains=["qos", "mobility"],
                        operation_ids=operation_ids,
                        resource_ids=resource_ids,
                        detail=(
                            f"Aggregate SM downlink MBR {requested_dl:g} Mbps exceeds "
                            f"UE-AMBR {ambr_dl:g} Mbps"
                        ),
                    )
                )

    directives = [
        RevisionDirective(
            owner_agent="ppa",
            code=conflict.code,
            instruction=conflict.detail,
            affected_operation_ids=conflict.operation_ids,
        )
        for conflict in conflicts
    ]
    return ConflictAssessment(
        status="blocked" if conflicts else "approved",
        conflicts=conflicts,
        revision_directives=directives,
    )


class PolicyMediator:
    actor_id = "policy_mediator"

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ExecutionAction):
            raise TypeError("PolicyMediator requires ExecutionAction")
        assessment = assess_policy_conflicts(current.payload)
        target = (
            "execution_dispatcher"
            if assessment.status == "approved"
            else "episode_controller"
        )
        return EventEmission(
            decision=CoordinationDecision[ConflictAssessment](
                target_agent=target,
                message_type="diagnosis",
                payload=assessment,
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )


class AssuranceDiagnoser:
    actor_id = "assurance_diagnoser"

    def __init__(
        self,
        *,
        journal: EpisodeJournal,
        outcome_evaluator: OutcomeEvaluator,
        repair_owner: DiagnosisOwner,
    ) -> None:
        self.journal = journal
        self.outcome_evaluator = outcome_evaluator
        self.repair_owner = repair_owner

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ExecutionFeedback):
            raise TypeError("AssuranceDiagnoser requires ExecutionFeedback")
        action = find_latest_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=ExecutionAction,
        )
        operations = {item.operation_id: item for item in action.operations}
        failed_ids = [
            receipt.operation_id
            for receipt in current.payload.receipts
            if receipt.status == "failed"
        ]
        evaluation = self.outcome_evaluator.evaluate(current.payload)

        if failed_ids:
            affected_domains = list(
                dict.fromkeys(operations[item].domain for item in failed_ids)
            )
            errors = [
                receipt.error
                for receipt in current.payload.receipts
                if receipt.status == "failed" and receipt.error is not None
            ]
            diagnosis = AssuranceDiagnosis(
                status="execution_failed",
                root_cause_category="execution_failure",
                reason_summary="; ".join(errors),
                affected_operation_ids=failed_ids,
                affected_domains=affected_domains,
                recommended_actions=[
                    "Revise the failed policy payloads using their execution receipts"
                ],
                repair_owner=self.repair_owner,
                reward_components=evaluation.reward_components,
            )
        elif evaluation.achieved:
            diagnosis = AssuranceDiagnosis(
                status="satisfied",
                root_cause_category="objective_achieved",
                reason_summary=evaluation.reason,
                affected_operation_ids=[],
                affected_domains=[],
                recommended_actions=[],
                repair_owner=self.repair_owner,
                reward_components=evaluation.reward_components,
            )
        else:
            operation_ids = list(operations)
            diagnosis = AssuranceDiagnosis(
                status="objective_violated",
                root_cause_category="objective_violation",
                reason_summary=evaluation.reason,
                affected_operation_ids=operation_ids,
                affected_domains=list(
                    dict.fromkeys(operations[item].domain for item in operation_ids)
                ),
                recommended_actions=[
                    "Revise policy parameters from the observed objective metrics"
                ],
                repair_owner=self.repair_owner,
                reward_components=evaluation.reward_components,
            )
        return EventEmission(
            decision=CoordinationDecision[AssuranceDiagnosis](
                target_agent="episode_controller",
                message_type="diagnosis",
                payload=diagnosis,
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )


def conflict_issues(assessment: ConflictAssessment) -> list[ValidationIssue]:
    return [
        ValidationIssue(code=item.code, message=item.instruction)
        for item in assessment.revision_directives
    ]


__all__ = [
    "AssuranceDiagnoser",
    "AssuranceDiagnosis",
    "ConflictAssessment",
    "DiagnosisOwner",
    "PolicyConflict",
    "PolicyMediator",
    "RevisionDirective",
    "assess_policy_conflicts",
    "conflict_issues",
]
