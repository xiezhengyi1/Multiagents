from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, model_validator

from control_runtime.context.prompts.main import (
    build_main_user_prompt,
    render_main_system_prompt,
)
from control_runtime.contracts import ControlDomain, ObjectiveProfile
from control_runtime.graph import (
    EpisodeGraphState,
    EventEmission,
)
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import (
    CoordinationDecision,
    ProposalPayload,
    ValidationIssue,
)

from .common import ControlRequest, find_event_payload
from .model import StructuredModelActor
from .prompt_contract import replace_contract_clauses
from .validation import ValidationActor


class MainInvestigationTarget(StrEnum):
    DOMAIN_BOUNDARY = "domain_boundary"
    UE_BINDING = "ue_binding"
    QOS_FLOW_BINDING = "qos_flow_binding"
    MOBILITY_TARGET_BINDING = "mobility_target_binding"
    POLICY_FEASIBILITY = "policy_feasibility"
    CROSS_DOMAIN_CONSISTENCY = "cross_domain_consistency"
    ASSURANCE_GAP = "assurance_gap"


class MainUncertaintyFlag(StrEnum):
    DOMAIN_AMBIGUOUS = "domain_ambiguous"
    IDENTIFIER_RISK = "identifier_risk"
    RUNTIME_EVIDENCE_MISSING = "runtime_evidence_missing"
    EXECUTION_FEEDBACK_INCOMPLETE = "execution_feedback_incomplete"
    CONFLICT_SIGNAL_PRESENT = "conflict_signal_present"


class MainRetryScope(StrEnum):
    FULL_REGROUND = "full_reground"
    PARTIAL_REGROUND = "partial_reground"
    TARGET_STABLE = "target_stable"
    EXECUTION_RETRY_FORBIDDEN = "execution_retry_forbidden"


class SemanticGoal(StrEnum):
    PROTECT = "protect"
    DEPRIORITIZE = "deprioritize"
    DEFER = "defer"
    OBSERVE = "observe"


class SemanticTargetType(StrEnum):
    FLOW = "flow"
    APP = "app"
    SCOPE = "scope"
    NAMED_OBJECT = "named_object"


class StageTrigger(StrEnum):
    INITIAL = "initial"
    ON_PREVIOUS_FAILURE = "on_previous_failure"
    AFTER_PREVIOUS_STAGE = "after_previous_stage"
    RETRY = "retry"


class ControlSemanticMode(StrEnum):
    SINGLE_STEP = "single_step"
    STAGED_PRIORITY = "staged_priority"
    CONDITIONAL_FALLBACK = "conditional_fallback"


class MainSemanticTarget(ContractModel):
    semantic_name: str
    target_type: SemanticTargetType
    goal: SemanticGoal
    metric_focus: str | None
    note: str
    supi: str


class MainControlStage(ContractModel):
    stage_index: int = Field(ge=1)
    name: str
    trigger: StageTrigger
    summary: str
    targets: list[MainSemanticTarget]


class MainControlSemantics(ContractModel):
    mode: ControlSemanticMode
    current_stage: int = Field(ge=1)
    stages: list[MainControlStage]
    notes: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_stage_selection(self) -> MainControlSemantics:
        if len(self.notes) != len(set(self.notes)):
            raise ValueError("control semantic notes must be unique")
        if not self.stages:
            if self.current_stage != 1:
                raise ValueError("current_stage must be 1 when stages is empty")
            return self
        indices = [stage.stage_index for stage in self.stages]
        if len(indices) != len(set(indices)):
            raise ValueError("stage_index values must be unique")
        if self.current_stage not in set(indices):
            raise ValueError("current_stage must reference an existing stage")
        return self


class MainDomainEvidence(ContractModel):
    qos: list[NonEmptyStr] = Field(default_factory=list)
    mobility: list[NonEmptyStr] = Field(default_factory=list)


class MainProposal(ProposalPayload):
    supi: str
    requested_domains: list[ControlDomain]
    serving_slice_change_requested: bool
    domain_evidence: MainDomainEvidence
    control_semantics: MainControlSemantics
    objective_profile: ObjectiveProfile
    investigation_targets: list[MainInvestigationTarget]
    uncertainty_flags: list[MainUncertaintyFlag]
    retry_scope: MainRetryScope | None
    required_evidence: list[NonEmptyStr]
    forbidden_assumptions: list[NonEmptyStr]
    intent_encoding_guidance: str

    @model_validator(mode="after")
    def validate_unique_collections(self) -> MainProposal:
        for name, values in (
            ("requested_domains", self.requested_domains),
            ("investigation_targets", self.investigation_targets),
            ("uncertainty_flags", self.uncertainty_flags),
            ("required_evidence", self.required_evidence),
            ("forbidden_assumptions", self.forbidden_assumptions),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        return self


MAIN_SYSTEM_PROMPT = replace_contract_clauses(
    render_main_system_prompt(),
    (
        (
            "then return exactly one MainControlDecision JSON object.",
            "then return exactly one structured CoordinationDecision[MainProposal].",
        ),
        (
            (
                "Output contract:\n"
                "- Return raw JSON only: one top-level object, never a list or prose.\n"
                "- Populate every decision field, using empty lists or strings where appropriate."
            ),
            (
                "Coordination data contract:\n"
                "- Return one object with exactly `target_agent`, `message_type`, and `payload`; never a list or prose.\n"
                '- Use `target_agent="sba"` for the original `next_agent="intent_encoding"` route and `target_agent="ppa"` for the original `next_agent="optimization_strategy"` route.\n'
                '- Use `message_type="proposal"` for a new decision and `message_type="repair_response"` when replying to a repair request.\n'
                "- Populate every MainProposal payload field, using empty lists or strings where appropriate. Runtime stamps event identity, scope, source agent, and evidence references."
            ),
        ),
        (
            "- Set next_agent to exactly `intent_encoding` or `optimization_strategy`.\n",
            "",
        ),
        (
            (
                "Required decision keys (use empty values where no decision is needed):\n"
                "supi, next_agent, requested_domains, serving_slice_change_requested, domain_evidence,"
            ),
            (
                "Required MainProposal payload keys (use empty values where no decision is needed):\n"
                "supi, requested_domains, serving_slice_change_requested, domain_evidence,"
            ),
        ),
        (
            "- Decide requested_domains, next_agent, retry_scope, optional",
            "- Decide requested_domains, the coordination target, retry_scope, optional",
        ),
        (
            (
                "MainControlDecision structure-only few-shot. Do not emit runtime wrapper keys like messages or structured_response.\n"
                "Return only these decision fields directly at the top level:"
            ),
            (
                "MainProposal payload structure-only few-shot. Do not copy these fields outside `payload` and do not emit runtime wrapper keys like messages or structured_response:\n"
                "The complete CoordinationDecision wraps this payload with `target_agent` and `message_type`:"
            ),
        ),
        (
            '  "next_agent": "intent_encoding",\n',
            "",
        ),
        (
            "- Any object missing next_agent.",
            "- Any object missing root `target_agent`, root `message_type`, or the MainProposal `payload`.",
        ),
    ),
)


def build_main_prompt(context_jsonl: str) -> str:
    return build_main_user_prompt(
        user_input=(
            "Read ControlRequest.user_input from the lossless UIR table below. "
            "Resolve entity and fact references before deciding."
        ),
        coordinator_context=context_jsonl,
    ) + (
        "\n\nData-contract task: return one structured CoordinationDecision "
        "with a MainProposal payload under the replacement contract in the system prompt.\nReply gate: inspect only the newest event in the UIR event table. If it is a repair_request, set message_type to repair_response; otherwise set it to proposal. Preserve every ControlRequest SUPI on its matching semantic target and follow the active single- or multi-SUPI gate exactly."
    )


def validate_main_proposal(
    *,
    proposal: MainProposal,
    target_agent: str,
    message_type: str,
    trigger_message_type: str,
    user_input: str,
    round_index: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(code: str, message: str, path: str | None = None) -> None:
        issues.append(ValidationIssue(code=code, message=message, path=path))

    if target_agent not in {"sba", "ppa"}:
        add("invalid_target", "Main target_agent must be sba or ppa", "target_agent")
    expected_message = (
        "repair_response" if trigger_message_type == "repair_request" else "proposal"
    )
    if message_type != expected_message:
        add(
            "invalid_message_type",
            f"Main must emit {expected_message} for this trigger",
            "message_type",
        )
    if round_index == 0 and target_agent != "sba":
        add("initial_route", "round 0 must route to sba", "target_agent")
    if not proposal.requested_domains:
        add(
            "missing_domains",
            "requested_domains cannot be empty",
            "payload.requested_domains",
        )
    domains = {domain.value for domain in proposal.requested_domains}
    if proposal.serving_slice_change_requested and domains != {"qos", "mobility"}:
        add(
            "slice_change_domains",
            "serving slice change requires qos and mobility",
            "payload.requested_domains",
        )
    evidence_domains = {
        domain
        for domain, evidence in proposal.domain_evidence.model_dump().items()
        if evidence
    }
    if not domains.issubset(evidence_domains):
        add(
            "missing_domain_evidence",
            "domain_evidence must cover every requested domain",
            "payload.domain_evidence",
        )
    if round_index > 0 and proposal.retry_scope is None:
        add(
            "missing_retry_scope",
            "repair rounds require retry_scope",
            "payload.retry_scope",
        )
    if (
        target_agent == "sba"
        and round_index > 0
        and not proposal.intent_encoding_guidance.strip()
    ):
        add(
            "missing_intent_encoding_guidance",
            "repair routing to sba requires intent_encoding_guidance",
            "payload.intent_encoding_guidance",
        )
    if target_agent != "sba" and proposal.intent_encoding_guidance.strip():
        add(
            "unexpected_intent_encoding_guidance",
            "intent_encoding_guidance is only valid when targeting sba",
            "payload.intent_encoding_guidance",
        )
    explicit_supis = list(dict.fromkeys(re.findall(r"(?i)(imsi-\d{5,})", user_input)))
    if len(explicit_supis) == 1:
        explicit = explicit_supis[0]
        if proposal.supi != explicit:
            add("supi_mismatch", f"supi must equal {explicit}", "payload.supi")
        for stage_index, stage in enumerate(proposal.control_semantics.stages):
            for target_index, target in enumerate(stage.targets):
                if target.supi != explicit:
                    add(
                        "semantic_target_supi",
                        f"semantic target must carry {explicit}",
                        f"payload.control_semantics.stages.{stage_index}.targets.{target_index}.supi",
                    )
    elif len(explicit_supis) > 1 and proposal.supi.strip():
        add(
            "multi_supi",
            "top-level supi must be empty for multi-SUPI requests",
            "payload.supi",
        )
    return issues


class MainActor(StructuredModelActor):
    actor_id = "main"
    validator_actor = "main_validator"
    system_prompt = MAIN_SYSTEM_PROMPT
    response_schema = CoordinationDecision[MainProposal]
    prompt_builder = staticmethod(build_main_prompt)

    def prepare_observations(
        self,
        *,
        current: object,
        observation_sink: object,
    ) -> tuple[str, ...]:
        control_request = find_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=ControlRequest,
        )
        supis = list(
            dict.fromkeys(re.findall(r"(?i)(imsi-\d{5,})", control_request.user_input))
        )
        self.request_supis = tuple(supis)
        self.authoritative_supi = supis[0] if len(supis) == 1 else None
        return ()

    def render_prompt(self, context_jsonl: str) -> str:
        prompt = build_main_prompt(context_jsonl)
        if len(self.request_supis) > 1:
            target_supis = ", ".join(f"`{supi}`" for supi in self.request_supis)
            return (
                prompt
                + "\n\nMulti-SUPI output gate:\n"
                + f"- This request has these exact target SUPIs: {target_supis}.\n"
                + "- Set payload.supi to the empty string `\"\"`. Never choose one target, invent a SUPI, or copy an illustrative SUPI into payload.supi.\n"
                + "- Each semantic target keeps its own exact SUPI from this list. Preserve that target-to-SUPI mapping in every repair response."
            )
        if self.authoritative_supi is None:
            return prompt
        return (
            prompt
            + "\n\nAuthoritative repair anchor:\n"
            + f"- This request has exactly one SUPI: `{self.authoritative_supi}`. "
            "Copy it verbatim into payload.supi and every semantic target.supi, "
            "including all repair responses. Repair evidence never replaces it."
        )


class MainValidator(ValidationActor):
    actor_id = "main_validator"

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        pending_failure = state.get("pending_model_failure")
        if pending_failure is not None:
            return self.repair(
                current=current,
                issues=[
                    ValidationIssue(
                        code="structured_output_invalid",
                        message=pending_failure.error,
                    )
                ],
                failed_trace_id=pending_failure.failed_trace_id,
                invalidated_artifact_ids=(),
                repair_owner="main",
            )
        if not isinstance(current.payload, MainProposal):
            raise TypeError("MainValidator requires MainProposal")
        control_request = find_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=ControlRequest,
        )
        trigger = (
            self.journal.get_event(current.in_reply_to)
            if current.in_reply_to
            else current
        )
        issues = validate_main_proposal(
            proposal=current.payload,
            target_agent=current.target_agent,
            message_type=current.message_type,
            trigger_message_type=trigger.message_type,
            user_input=control_request.user_input,
            round_index=current.scope.round_index,
        )
        if issues:
            return self.repair(
                current=current,
                issues=issues,
                failed_trace_id=None,
                invalidated_artifact_ids=(current.event_id,),
                repair_owner="main",
            )
        return self.accept(
            current=current,
            target_agent=current.target_agent,
            notes=["main proposal accepted"],
            resolved_actor="main",
        )


__all__ = [
    "MAIN_SYSTEM_PROMPT",
    "ControlSemanticMode",
    "MainActor",
    "MainControlSemantics",
    "MainControlStage",
    "MainInvestigationTarget",
    "MainProposal",
    "MainRetryScope",
    "MainSemanticTarget",
    "MainUncertaintyFlag",
    "MainValidator",
    "SemanticGoal",
    "SemanticTargetType",
    "StageTrigger",
    "build_main_prompt",
    "validate_main_proposal",
]
