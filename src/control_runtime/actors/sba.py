from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from control_runtime.context.prompts.grounding import render_grounding_system_prompt
from control_runtime.graph import EpisodeGraphState, EventEmission
from control_runtime.simulation import SBA_OBSERVATION_TOOLS, observation_query_names
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import (
    CoordinationDecision,
    HandoffPayload,
    ProposalPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.runtime.journal import EpisodeJournal

from .common import find_latest_event_payload
from .main import MainProposal
from .model import StructuredModelActor
from .prompt_contract import replace_contract_clauses
from .validation import ValidationActor


class GroundedFlowBinding(ContractModel):
    app_id: NonEmptyStr | None
    flow_id: NonEmptyStr | None
    name: NonEmptyStr
    resolution_status: Literal["resolved", "unresolved"]

    @model_validator(mode="after")
    def validate_resolution(self) -> GroundedFlowBinding:
        bound = self.app_id is not None and self.flow_id is not None
        if self.resolution_status == "resolved" and not bound:
            raise ValueError("resolved flow requires app_id and flow_id")
        if self.resolution_status == "unresolved" and (
            self.app_id is not None or self.flow_id is not None
        ):
            raise ValueError("unresolved flow cannot claim app_id or flow_id")
        return self


class MobilityBinding(ContractModel):
    association_id: NonEmptyStr
    rfsp: int = Field(ge=0)
    allowed_snssais: list[NonEmptyStr] = Field(min_length=1)
    target_snssais: list[NonEmptyStr] = Field(min_length=1)
    access_type: Literal["3GPP_ACCESS", "NON_3GPP_ACCESS"] | None
    service_area_restriction: dict[str, JsonValue] | None

    @model_validator(mode="after")
    def validate_targets(self) -> MobilityBinding:
        if not set(self.target_snssais).issubset(self.allowed_snssais):
            raise ValueError("mobility targets must be a subset of allowed_snssais")
        return self


class QosOperationConstraint(ContractModel):
    flow_id: NonEmptyStr
    app_id: NonEmptyStr
    operation_type: Literal["qos_reallocation", "slice_migration"]
    require_slice_change: bool
    target_slice_preference: str
    no_op_allowed: bool
    rationale: list[NonEmptyStr]


class SliceMigrationDecision(ContractModel):
    decision: Literal[
        "not_applicable",
        "migration_authorized",
        "blocked_by_subscription_entitlement",
        "blocked_requires_subscription_provisioning",
        "evidence_missing",
    ]
    target_snssais: list[NonEmptyStr]
    subscription_change_required: bool
    rationale: list[NonEmptyStr]


class BlockingQuestion(ContractModel):
    owner_agent: NonEmptyStr
    question: NonEmptyStr
    blocking: bool
    related_domains: list[Literal["qos", "mobility"]]


class SbaProposal(ProposalPayload):
    flows: list[GroundedFlowBinding]
    mobility_intent: MobilityBinding | None
    qos_operation_constraints: list[QosOperationConstraint]
    slice_migration_authorization: SliceMigrationDecision
    open_questions: list[BlockingQuestion]

    @model_validator(mode="after")
    def validate_unique_bindings(self) -> SbaProposal:
        flow_names = [flow.name for flow in self.flows]
        if len(flow_names) != len(set(flow_names)):
            raise ValueError("flow names must be unique")
        resolved_ids = [flow.flow_id for flow in self.flows if flow.flow_id is not None]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("resolved flow_id values must be unique")
        operation_ids = [item.flow_id for item in self.qos_operation_constraints]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("qos operation flow_id values must be unique")
        return self


class SbaArtifact(HandoffPayload):
    supi: NonEmptyStr
    flows: list[GroundedFlowBinding]
    mobility_intent: MobilityBinding | None
    qos_operation_constraints: list[QosOperationConstraint]
    slice_migration_authorization: SliceMigrationDecision
    open_questions: list[BlockingQuestion]


SBA_SYSTEM_PROMPT = replace_contract_clauses(
    render_grounding_system_prompt(),
    (
        (
            "goals. Do not repeat, revise, or output any of those fields. Ground only the active Main stage and return one raw IEA decision JSON object.",
            "goals. Do not repeat, revise, or output any of those fields. Ground only the active Main stage and return one structured CoordinationDecision[SbaProposal].",
        ),
        (
            "You generate only these decision fields: `flows`, `mobility_intent`, `qos_operation_constraints`, `slice_migration_authorization`, and `open_questions`.",
            "The `payload` contains only these SbaProposal fields: `flows`, `mobility_intent`, `qos_operation_constraints`, `slice_migration_authorization`, and `open_questions`.",
        ),
        (
            "Schema firewall: emit exactly those five root keys.",
            "Schema firewall: emit exactly `target_agent`, `message_type`, and `payload` at the root, and exactly those five fields inside `payload`.",
        ),
        (
            "Top-level shape gate: `[]`, any bare JSON list, `{}`, and an empty response are never valid final output. Even when evidence is missing or a target is unresolved, emit the one five-key IEA object with the appropriate empty collections and unresolved flow ledger row.",
            'Top-level shape gate: `[]`, any bare JSON list, `{}`, and an empty response are never valid final output. Emit `target_agent="sba_compiler"`; use `message_type="proposal"` normally and `message_type="repair_response"` when replying to a repair request. Even when evidence is missing or a target is unresolved, emit the five-key SbaProposal payload with the appropriate empty collections and unresolved flow ledger row. Runtime stamps event identity, scope, source agent, and evidence references.',
        ),
        (
            "Few-shot output shapes (IDs are illustrative; copy only evidence-backed values):",
            "Few-shot SbaProposal payload shapes (IDs are illustrative; copy only evidence-backed values). Each shown object belongs inside the CoordinationDecision `payload`:",
        ),
        (
            "Return raw JSON only: one object, no prose or markdown. Do not output runtime\nkeys, Main-owned fields, full policy payloads, or control-stage fields.",
            "Return one CoordinationDecision object, no prose or markdown. Do not output runtime-stamped\nkeys, Main-owned fields, full policy payloads, or control-stage fields inside `payload`.",
        ),
    ),
)


def build_sba_prompt(context_jsonl: str) -> str:
    return (
        "User request and structured evidence:\n"
        "Read the Main proposal, ControlRequest, and simulator observations from the exact event ancestry.\n\n"
        "Coordinator context:\n"
        f"{context_jsonl}\n\n"
        "Task:\n"
        "- Resolve only choices that remain ambiguous; use already-grounded evidence directly.\n"
        "- Do not guess identifiers. An unresolved QoS target still appears in flows.\n"
        "- Follow every finalization and tool gate in the original system prompt.\n"
        "- Return one structured CoordinationDecision with an SbaProposal payload."
    )


class SbaActor(StructuredModelActor):
    actor_id = "sba"
    validator_actor = "sba_validator"
    system_prompt = SBA_SYSTEM_PROMPT
    response_schema = CoordinationDecision[SbaProposal]
    prompt_builder = staticmethod(build_sba_prompt)
    default_tools = SBA_OBSERVATION_TOOLS
    requires_simulator_adapter = True


def validate_sba_proposal(
    *,
    proposal: SbaProposal,
    main_proposal: MainProposal,
    target_agent: str,
    message_type: str,
    trigger_message_type: str,
    observation_queries: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(code: str, message: str, path: str | None = None) -> None:
        issues.append(ValidationIssue(code=code, message=message, path=path))

    if target_agent != "sba_compiler":
        add("invalid_target", "SBA must target sba_compiler", "target_agent")
    expected = (
        "repair_response" if trigger_message_type == "repair_request" else "proposal"
    )
    if message_type != expected:
        add(
            "invalid_message_type",
            f"SBA must emit {expected} for this trigger",
            "message_type",
        )
    requested_domains = {domain.value for domain in main_proposal.requested_domains}
    if "qos" in requested_domains:
        if "get_sm_ue_flow_catalog" not in observation_queries:
            add(
                "missing_sm_catalog",
                "QoS grounding requires get_sm_ue_flow_catalog evidence",
            )
        if not proposal.flows:
            add(
                "missing_flows",
                "QoS grounding requires at least one flow",
                "payload.flows",
            )
        unresolved = [
            flow.name
            for flow in proposal.flows
            if flow.resolution_status == "unresolved"
        ]
        if unresolved:
            add(
                "unresolved_flows",
                "QoS targets remain unresolved: " + ", ".join(unresolved),
            )
        resolved_pairs = {
            (flow.flow_id, flow.app_id)
            for flow in proposal.flows
            if flow.resolution_status == "resolved"
        }
        for index, constraint in enumerate(proposal.qos_operation_constraints):
            if (constraint.flow_id, constraint.app_id) not in resolved_pairs:
                add(
                    "operation_binding_mismatch",
                    "QoS operation must reference a resolved flow/app pair",
                    f"payload.qos_operation_constraints.{index}",
                )
    elif proposal.flows or proposal.qos_operation_constraints:
        add("unexpected_qos_binding", "non-QoS requests cannot contain QoS bindings")

    if "mobility" in requested_domains:
        if "get_am_policy_context" not in observation_queries:
            add(
                "missing_am_context",
                "mobility grounding requires get_am_policy_context evidence",
            )
        if proposal.mobility_intent is None:
            add(
                "missing_mobility_binding",
                "mobility request requires a grounded mobility binding",
            )
    elif proposal.mobility_intent is not None:
        add(
            "unexpected_mobility_binding",
            "non-mobility requests cannot contain mobility binding",
        )

    if main_proposal.serving_slice_change_requested:
        if "get_ue_slice_subscription" not in observation_queries:
            add(
                "missing_subscription", "slice migration requires subscription evidence"
            )
    elif proposal.slice_migration_authorization.decision != "not_applicable":
        add(
            "unexpected_slice_migration",
            "slice migration decision must be not_applicable",
        )

    if any(question.blocking for question in proposal.open_questions):
        add("blocking_question", "SBA still has blocking grounding questions")
    return issues


class SbaValidator(ValidationActor):
    actor_id = "sba_validator"

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        pending_failure = state.get("pending_model_failure")
        if pending_failure is not None:
            return self.repair(
                current=current,
                issues=[
                    ValidationIssue(
                        code="structured_output_invalid", message=pending_failure.error
                    )
                ],
                repair_owner="sba",
                failed_trace_id=pending_failure.failed_trace_id,
                invalidated_artifact_ids=(),
            )
        if not isinstance(current.payload, SbaProposal):
            raise TypeError("SbaValidator requires SbaProposal")
        main_proposal = find_latest_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=MainProposal,
        )
        trigger = (
            self.journal.get_event(current.in_reply_to)
            if current.in_reply_to
            else current
        )
        issues = validate_sba_proposal(
            proposal=current.payload,
            main_proposal=main_proposal,
            target_agent=current.target_agent,
            message_type=current.message_type,
            trigger_message_type=trigger.message_type,
            observation_queries=observation_query_names(
                journal=self.journal,
                event_ids=current.input_artifact_ids,
            ),
        )
        if issues:
            return self.repair(
                current=current,
                issues=issues,
                repair_owner="sba",
                failed_trace_id=None,
                invalidated_artifact_ids=(current.event_id,),
            )
        return self.accept(
            current=current,
            target_agent="sba_compiler",
            notes=["SBA proposal accepted"],
            resolved_actor="sba",
        )


class SbaCompiler:
    actor_id = "sba_compiler"

    def __init__(self, *, journal: EpisodeJournal) -> None:
        self.journal = journal

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ValidationResultPayload):
            raise TypeError("SbaCompiler requires ValidationResultPayload")
        if current.in_reply_to is None:
            raise ValueError("SBA validation result must reference its proposal")
        proposal_event = self.journal.get_event(current.in_reply_to)
        if not isinstance(proposal_event.payload, SbaProposal):
            raise TypeError("SBA validation result does not reference SbaProposal")
        main_proposal = find_latest_event_payload(
            journal=self.journal,
            current_event=proposal_event,
            payload_type=MainProposal,
        )
        artifact = SbaArtifact(
            supi=main_proposal.supi,
            **proposal_event.payload.model_dump(mode="python"),
        )
        return EventEmission(
            decision=CoordinationDecision[SbaArtifact](
                target_agent="ppa",
                message_type="handoff",
                payload=artifact,
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(proposal_event.event_id, current.event_id),
            invalidated_artifact_ids=(),
        )


__all__ = [
    "BlockingQuestion",
    "GroundedFlowBinding",
    "MobilityBinding",
    "QosOperationConstraint",
    "SbaActor",
    "SbaArtifact",
    "SbaCompiler",
    "SbaProposal",
    "SbaValidator",
    "SliceMigrationDecision",
    "validate_sba_proposal",
]
