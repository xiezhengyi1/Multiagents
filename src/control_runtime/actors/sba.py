from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from control_runtime.context.prompts.grounding import render_grounding_system_prompt
from control_runtime.graph import EpisodeGraphState, EventEmission
from control_runtime.simulation import (
    SBA_OBSERVATION_TOOLS,
    EnvironmentObservation,
    observation_query_names,
)
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
    subscription: dict[str, JsonValue] | None = None
    flow_catalog: list[dict[str, JsonValue]] | None = None
    ue_policy_associations: list[dict[str, JsonValue]] | None = None
    pdu_session_pool: list[dict[str, JsonValue]] | None = None


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
        "Read the Main proposal, ControlRequest, and simulator observations from the lossless UIR tables. "
        "`entities` holds exact identifiers, `facts` holds exact payloads, and `deltas` links each event to its payload.\n\n"
        "Coordinator context:\n"
        f"{context_jsonl}\n\n"
        "Task:\n"
        "- Resolve only choices that remain ambiguous; use already-grounded evidence directly.\n"
        "- Do not guess identifiers. An unresolved QoS target still appears in flows.\n"
        "- A flow named only under an explicit no-change constraint is not a QoS target: omit it from flows and qos_operation_constraints.\n"
        "- A resolved flow has both catalog app_id and flow_id. An unresolved flow has both fields set to JSON null; never mix the two states.\n"
        "- Emit one qos_operation_constraints item for each resolved flow and never emit two items with the same flow_id. A slice migration is encoded by require_slice_change on that one item, never by a second operation.\n"
        "- Follow every finalization and tool gate in the original system prompt.\n"
        "- Return one structured CoordinationDecision with an SbaProposal payload.\n"
        "- Reply gate: inspect only the newest event. A repair_request requires message_type=repair_response; every other trigger requires message_type=proposal. Never copy a message_type from an older event."
        "\nTool argument gate:\n"
        "- For every observation tool call, copy `supi` exactly from the newest MainProposal. Never use an illustrative, inferred, or newly constructed SUPI.\n"
        "- In a repair round, reuse observations already present in the coordinator context. Query only a fact that is still missing for the current MainProposal SUPI."
        "\n- A validator issue `missing_sm_catalog` means the flow catalog is still missing. Query `get_sm_ue_flow_catalog` for the authoritative SUPI before returning the repair response."
        "\n- In a QoS repair, inspect UIR `facts` for a current `get_sm_ue_flow_catalog` result for the authoritative SUPI. If it is absent, call that tool before grounding flows."
    )


def _execution_supi(proposal: MainProposal) -> NonEmptyStr:
    if proposal.supi is not None:
        return proposal.supi
    return next(
        target.supi
        for stage in proposal.control_semantics.stages
        if stage.stage_index == proposal.control_semantics.current_stage
        for target in stage.targets
    )


class SbaActor(StructuredModelActor):
    actor_id = "sba"
    validator_actor = "sba_validator"
    system_prompt = SBA_SYSTEM_PROMPT
    response_schema = CoordinationDecision[SbaProposal]
    prompt_builder = staticmethod(build_sba_prompt)
    default_tools = SBA_OBSERVATION_TOOLS
    requires_simulator_adapter = True

    def prepare_observations(
        self,
        *,
        current: object,
        observation_sink: object,
    ) -> tuple[str, ...]:
        main_proposal = find_latest_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=MainProposal,
        )
        self.authorized_supi = _execution_supi(main_proposal)
        return ()

    def render_prompt(self, context_jsonl: str) -> str:
        return (
            build_sba_prompt(context_jsonl)
            + "\n\nAuthoritative tool parameter for this invocation:\n"
            + f"- The only allowed `supi` value is `{self.authorized_supi}`. "
            "Copy this exact literal into every tool call. "
            "`supi-12345`, examples, aliases, and any other value are invalid."
            "\n- When Main top-level supi is null, this is the primary SUPI of the first current-stage target. Ground only current-stage targets with this SUPI; later targets remain Main-owned staged semantics."
            "\n\nQoS observation gate:\n"
            "- When the active Main proposal selects QoS and UIR has no current `get_sm_ue_flow_catalog` fact for this SUPI, your next response is that tool call with this exact `supi`. Do not submit CoordinationDecision before its result.\n"
            "- After the catalog result, use only its identifiers. If it lacks the named target, call `search_sm_flow_targets`; otherwise continue the active evidence gates below.\n"
            "- When Main selects mobility only, set flows and qos_operation_constraints to []; do not create an inspection placeholder or QoS binding.\n"
            "\nEvidence-completion gate:\n"
            "- Do not submit a CoordinationDecision while an active Main domain lacks its required fact. If Main selects mobility and UIR has no current `get_am_policy_context` fact for this SUPI, call `get_am_policy_context` next.\n"
            "- If Main set `serving_slice_change_requested:true`, bind the flow from the catalog, then, when UIR has no current `get_ue_slice_subscription` fact for this SUPI, call `get_ue_slice_subscription` next.\n"
            "- When both mobility context and subscription are missing, call `get_am_policy_context` first, then call `get_ue_slice_subscription`; never repeat the catalog.\n"
            "- For QoS plus mobility with `serving_slice_change_requested:true`, the only valid sequence is catalog, AM context, subscription, then structured completion. A catalog result alone is never sufficient.\n"
            "- A repair issue `missing_am_context` means call `get_am_policy_context`; `missing_subscription` means call `get_ue_slice_subscription`; `missing_sm_catalog` means call `get_sm_ue_flow_catalog`. Act on the missing fact, not on a previously completed query.\n"
            "- Once the catalog, AM context, and subscription required by the active Main proposal are present, finalize from those facts: emit the evidence-grounded mobility_intent and migration decision, and set open_questions to []. A subscription result is an answer, not a blocking question.\n"
            "- Copy mobility_intent.target_snssais only from observed allowed_snssais; a requested or preferred slice is not evidence. When migration is blocked and no alternative is authorized, slice_migration_authorization.target_snssais must be [].\n"
            "- Never change a catalog-resolved flow into unresolved during repair. `resolution_status=resolved` requires its exact catalog app_id and flow_id; `resolution_status=unresolved` requires both JSON fields to be null."
            "\n- A blocked_by_subscription_entitlement decision is final evidence: set open_questions to [] and return it without another tool call."
        )


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
        observations = [
            event.payload
            for event_id in proposal_event.input_artifact_ids
            if isinstance(
                (event := self.journal.get_event(event_id)).payload,
                EnvironmentObservation,
            )
        ]

        def observation_result(query_name: str) -> dict[str, JsonValue] | None:
            for observation in reversed(observations):
                if observation.query.query_name == query_name and isinstance(
                    observation.result, dict
                ):
                    return observation.result
            return None

        subscription = observation_result("get_ue_slice_subscription")
        catalog = observation_result("get_sm_ue_flow_catalog")
        am_context = observation_result("get_am_policy_context")
        artifact = SbaArtifact(
            supi=_execution_supi(main_proposal),
            **proposal_event.payload.model_dump(mode="python"),
            subscription=subscription,
            flow_catalog=catalog["flow_catalog"] if catalog is not None else None,
            ue_policy_associations=(
                am_context["ue_policy_associations"]
                if am_context is not None
                else None
            ),
            pdu_session_pool=(
                am_context["pdu_session_pool"] if am_context is not None else None
            ),
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
