from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from control_runtime.context.prompts.planning import (
    build_planning_user_prompt,
    render_planning_system_prompt,
)
from control_runtime.execution.policy_compiler import compile_ppa_execution_action
from control_runtime.graph import EpisodeGraphState, EventEmission
from control_runtime.simulation import (
    PPA_OBSERVATION_TOOLS,
    EnvironmentObservation,
    ExecutionAction,
    observation_query_names,
)
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import (
    CoordinationDecision,
    ProposalPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.runtime.journal import EpisodeJournal

from .common import PlanningBlocker, find_latest_event_payload
from .model import StructuredModelActor
from .prompt_contract import replace_contract_clauses
from .sba import BlockingQuestion, SbaArtifact
from .validation import ValidationActor


class SnssaiSpec(ContractModel):
    sst: int = Field(ge=0, le=255)
    sd: str | None = Field(pattern=r"^[0-9a-f]{6}$")


class SmPolicySpec(ContractModel):
    flow_id: NonEmptyStr
    app_id: NonEmptyStr
    priority: int = Field(ge=1, le=15)
    target_latency_ms: float = Field(ge=1.0)
    packet_error_rate: float = Field(ge=0.0, le=1.0)
    max_br_ul_mbps: float = Field(gt=0.0)
    max_br_dl_mbps: float = Field(gt=0.0)
    gbr_ul_mbps: float | None = Field(ge=0.0)
    gbr_dl_mbps: float | None = Field(ge=0.0)
    target_jitter_ms: float | None = Field(ge=0.0)
    flow_description: NonEmptyStr | None

    @model_validator(mode="after")
    def validate_gbr(self) -> SmPolicySpec:
        if self.gbr_ul_mbps is not None and self.gbr_ul_mbps > self.max_br_ul_mbps:
            raise ValueError("gbr_ul_mbps cannot exceed max_br_ul_mbps")
        if self.gbr_dl_mbps is not None and self.gbr_dl_mbps > self.max_br_dl_mbps:
            raise ValueError("gbr_dl_mbps cannot exceed max_br_dl_mbps")
        return self


class AmPolicySpec(ContractModel):
    triggers: list[NonEmptyStr] = Field(min_length=1)
    rfsp: int = Field(ge=0)
    allowed_snssais: list[SnssaiSpec] = Field(min_length=1)
    target_snssais: list[SnssaiSpec] = Field(min_length=1)
    ue_ambr_ul_mbps: float | None = Field(ge=0.0)
    ue_ambr_dl_mbps: float | None = Field(ge=0.0)
    notification_uri: NonEmptyStr
    supp_feat: NonEmptyStr

    @model_validator(mode="after")
    def validate_targets(self) -> AmPolicySpec:
        allowed = {(item.sst, item.sd) for item in self.allowed_snssais}
        targets = {(item.sst, item.sd) for item in self.target_snssais}
        if not targets.issubset(allowed):
            raise ValueError("target_snssais must be a subset of allowed_snssais")
        if (self.ue_ambr_ul_mbps is None) != (self.ue_ambr_dl_mbps is None):
            raise ValueError(
                "ue_ambr_ul_mbps and ue_ambr_dl_mbps must be provided together"
            )
        return self


class TrafficDescriptorSpec(ContractModel):
    flow_descs: list[NonEmptyStr]
    dnns: list[NonEmptyStr]
    app_ids: list[NonEmptyStr]
    os_id: NonEmptyStr | None

    @model_validator(mode="after")
    def validate_signal(self) -> TrafficDescriptorSpec:
        if not self.flow_descs and not self.dnns and not self.app_ids:
            raise ValueError("traffic descriptor needs flow_descs, dnns, or app_ids")
        return self


class RouteSelectionParameterSetSpec(ContractModel):
    dnn: NonEmptyStr
    precedence: int = Field(ge=1)
    snssai: SnssaiSpec | None


class UrspPolicySpec(ContractModel):
    target_type: Literal["flow", "app"]
    app_id: NonEmptyStr
    flow_id: NonEmptyStr | None
    relat_precedence: int = Field(ge=1)
    traffic_desc: TrafficDescriptorSpec | None
    route_sel_param_sets: list[RouteSelectionParameterSetSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target(self) -> UrspPolicySpec:
        if self.target_type == "flow" and self.flow_id is None:
            raise ValueError("flow-scoped URSP policy requires flow_id")
        if self.target_type == "flow" and self.traffic_desc is None:
            raise ValueError("flow-scoped URSP policy requires traffic_desc")
        return self


class PartialPolicySpec(ContractModel):
    app_id: str
    flow_id: NonEmptyStr | None
    target_type: Literal["flow", "app", "ue"]
    policy_type: NonEmptyStr
    resource_keys: list[NonEmptyStr]
    policy_details: dict[str, JsonValue]


class PpaProposal(ProposalPayload):
    planning_status: Literal[
        "executable_plan", "partial_plan", "needs_upstream_reground"
    ]
    rationale: NonEmptyStr
    missing_evidence: list[NonEmptyStr]
    blocked_targets: list[NonEmptyStr]
    upstream_requests: list[NonEmptyStr]
    planner_conflicts: list[NonEmptyStr]
    open_questions: list[BlockingQuestion]
    sm_policies: list[SmPolicySpec]
    am_policy: AmPolicySpec | None
    ursp_policies: list[UrspPolicySpec]
    partial_policies: list[PartialPolicySpec]

    @model_validator(mode="after")
    def validate_status(self) -> PpaProposal:
        for name, values in (
            ("missing_evidence", self.missing_evidence),
            ("blocked_targets", self.blocked_targets),
            ("upstream_requests", self.upstream_requests),
            ("planner_conflicts", self.planner_conflicts),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        flow_ids = [policy.flow_id for policy in self.sm_policies]
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("SM policy flow_id values must be unique")
        if self.planning_status == "needs_upstream_reground":
            if self.sm_policies or self.am_policy is not None or self.ursp_policies:
                raise ValueError(
                    "needs_upstream_reground cannot contain executable policies"
                )
            if not self.missing_evidence and not self.upstream_requests:
                raise ValueError(
                    "needs_upstream_reground must state missing evidence or requests"
                )
        if self.planning_status == "partial_plan":
            if (
                not self.partial_policies
                and not self.sm_policies
                and self.am_policy is None
                and not self.ursp_policies
            ):
                raise ValueError(
                    "partial_plan must contain a partial or executable fragment"
                )
            if (
                not self.missing_evidence
                and not self.blocked_targets
                and not self.planner_conflicts
            ):
                raise ValueError("partial_plan must explain what blocks full execution")
        if self.planning_status == "executable_plan":
            if self.missing_evidence or self.upstream_requests:
                raise ValueError(
                    "executable_plan cannot report unresolved upstream evidence"
                )
            if (
                not self.sm_policies
                and self.am_policy is None
                and not self.ursp_policies
            ):
                raise ValueError("executable_plan must contain at least one policy")
        return self


PPA_SYSTEM_PROMPT = replace_contract_clauses(
    render_planning_system_prompt(),
    (
        (
            "3. output one complete OsaAdvisorOutput JSON object.",
            "3. output one complete structured CoordinationDecision[PpaProposal].",
        ),
        (
            (
                "- Return raw JSON: exactly one OsaAdvisorOutput object; never markdown, `planning_metadata`, a bare policy object, or an array.\n"
                "- Required root fields: planning_status, rationale, missing_evidence, blocked_targets, upstream_requests, planner_conflicts, open_questions, sm_policies, am_policy, ursp_policies, partial_policies."
            ),
            (
                "- Return exactly one object with root fields `target_agent`, `message_type`, and `payload`; never markdown, `planning_metadata`, a bare policy object, or an array.\n"
                '- Use `target_agent="ppa_compiler"` for `executable_plan`, `target_agent="sba"` for `needs_upstream_reground`, and `target_agent="episode_controller"` for `partial_plan`. Use `message_type="proposal"` normally and `message_type="repair_response"` when replying to a repair request.\n'
                "- Required PpaProposal payload fields: planning_status, rationale, missing_evidence, blocked_targets, upstream_requests, planner_conflicts, open_questions, sm_policies, am_policy, ursp_policies, partial_policies. Runtime stamps event identity, scope, source agent, and evidence references."
            ),
        ),
        (
            "- Return only OsaAdvisorOutput fields.",
            "- Return only PpaProposal fields inside `payload`.",
        ),
        (
            "Few-shot OSA output shapes — structure only.",
            "Few-shot PpaProposal payload shapes — structure only. Each shown object belongs inside the CoordinationDecision `payload`.",
        ),
        (
            "Never return bare policy objects or bare arrays. Always wrap policy items inside the top-level OsaAdvisorOutput object.",
            "Never return bare policy objects or bare arrays. Always wrap policy items inside the PpaProposal `payload` of the top-level CoordinationDecision object.",
        ),
    ),
)


def build_ppa_prompt(
    context_jsonl: str,
    *,
    available_tool_names: list[str],
) -> str:
    return build_planning_user_prompt(
        normalized_user_intent={"source": "MainProposal in events_jsonl"},
        coordination_context={"events_jsonl": context_jsonl},
        planning_evidence={
            "source": "SbaArtifact and tool observations in events_jsonl"
        },
        available_tool_names=available_tool_names,
    ) + (
        "\nData-contract task: return one structured CoordinationDecision "
        "with a PpaProposal payload under the replacement contract in the system prompt."
    )


class PpaActor(StructuredModelActor):
    actor_id = "ppa"
    validator_actor = "ppa_validator"
    system_prompt = PPA_SYSTEM_PROMPT
    response_schema = CoordinationDecision[PpaProposal]
    prompt_builder = staticmethod(
        lambda context_jsonl: build_ppa_prompt(
            context_jsonl,
            available_tool_names=[tool.name for tool in PPA_OBSERVATION_TOOLS],
        )
    )
    default_tools = PPA_OBSERVATION_TOOLS
    requires_simulator_adapter = True

    def render_prompt(self, context_jsonl: str) -> str:
        return build_ppa_prompt(
            context_jsonl,
            available_tool_names=list(self.tool_names),
        )


def _latest_observation(
    *,
    journal: EpisodeJournal,
    event_ids: list[str],
    query_name: str,
) -> EnvironmentObservation:
    matches: list[tuple[int, EnvironmentObservation]] = []
    for event_id in event_ids:
        event = journal.get_event(event_id)
        if (
            isinstance(event.payload, EnvironmentObservation)
            and event.payload.query.query_name == query_name
        ):
            matches.append((event.event_index, event.payload))
    if not matches:
        raise ValueError(f"missing observation {query_name}")
    return max(matches, key=lambda item: item[0])[1]


def _validate_optimizer_values(
    *,
    journal: EpisodeJournal,
    event_ids: list[str],
    policies: list[SmPolicySpec],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        observation = _latest_observation(
            journal=journal,
            event_ids=event_ids,
            query_name="preview_qos_optimizer",
        )
    except ValueError:
        return [
            ValidationIssue(
                code="missing_optimizer_preview",
                message="SM policies require preview_qos_optimizer evidence",
            )
        ]
    if not isinstance(observation.result, dict):
        return [
            ValidationIssue(
                code="invalid_optimizer_preview",
                message="optimizer preview must be an object",
            )
        ]
    assignments = observation.result.get("qos_flow_assignments")
    if not isinstance(assignments, list):
        return [
            ValidationIssue(
                code="invalid_optimizer_preview",
                message="optimizer preview must contain qos_flow_assignments",
            )
        ]
    by_flow = {
        str(item["flow_id"]): item
        for item in assignments
        if isinstance(item, dict) and "flow_id" in item
    }
    for index, policy in enumerate(policies):
        assignment = by_flow.get(policy.flow_id)
        if assignment is None:
            issues.append(
                ValidationIssue(
                    code="optimizer_flow_missing",
                    message=f"optimizer preview has no assignment for {policy.flow_id}",
                    path=f"payload.sm_policies.{index}",
                )
            )
            continue
        try:
            bw_ul = float(assignment["bw_ul"])
            bw_dl = float(assignment["bw_dl"])
        except (KeyError, TypeError, ValueError):
            issues.append(
                ValidationIssue(
                    code="optimizer_bandwidth_invalid",
                    message=f"optimizer assignment for {policy.flow_id} has invalid bandwidths",
                )
            )
            continue
        if not math.isclose(policy.max_br_ul_mbps, bw_ul) or not math.isclose(
            policy.max_br_dl_mbps, bw_dl
        ):
            issues.append(
                ValidationIssue(
                    code="optimizer_value_mismatch",
                    message=f"SM bandwidths for {policy.flow_id} must equal optimizer output",
                    path=f"payload.sm_policies.{index}",
                )
            )
    return issues


def validate_ppa_proposal(
    *,
    proposal: PpaProposal,
    sba_artifact: SbaArtifact,
    target_agent: str,
    message_type: str,
    trigger_message_type: str,
    observation_queries: set[str],
    journal: EpisodeJournal,
    input_artifact_ids: list[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(code: str, message: str, path: str | None = None) -> None:
        issues.append(ValidationIssue(code=code, message=message, path=path))

    expected = (
        "repair_response" if trigger_message_type == "repair_request" else "proposal"
    )
    if message_type != expected:
        add(
            "invalid_message_type",
            f"PPA must emit {expected} for this trigger",
            "message_type",
        )
    if proposal.planning_status == "needs_upstream_reground":
        if target_agent != "sba":
            add(
                "invalid_upstream_target",
                "upstream reground must target sba",
                "target_agent",
            )
        return issues
    if proposal.planning_status == "partial_plan":
        if target_agent != "episode_controller":
            add(
                "invalid_partial_target",
                "partial_plan must target episode_controller",
                "target_agent",
            )
        return issues
    if target_agent != "ppa_compiler":
        add(
            "invalid_target",
            "executable PPA plan must target ppa_compiler",
            "target_agent",
        )

    resolved_pairs = {
        (flow.flow_id, flow.app_id)
        for flow in sba_artifact.flows
        if flow.resolution_status == "resolved"
    }
    for index, policy in enumerate(proposal.sm_policies):
        if (policy.flow_id, policy.app_id) not in resolved_pairs:
            add(
                "unknown_flow_binding",
                "SM policy must reference an SBA-resolved flow/app pair",
                f"payload.sm_policies.{index}",
            )
    if proposal.sm_policies:
        issues.extend(
            _validate_optimizer_values(
                journal=journal,
                event_ids=input_artifact_ids,
                policies=proposal.sm_policies,
            )
        )
    if proposal.am_policy is not None:
        if "inspect_mobility_ue_policies" not in observation_queries:
            add(
                "missing_mobility_observation",
                "AM policy requires live mobility-policy evidence",
            )
        if sba_artifact.mobility_intent is None:
            add("missing_mobility_binding", "AM policy requires SBA mobility binding")
        else:
            if proposal.am_policy.rfsp != sba_artifact.mobility_intent.rfsp:
                add("rfsp_mismatch", "AM RFSP must match SBA binding")
    if not issues:
        try:
            compile_ppa_execution_action(
                proposal=proposal,
                sba_artifact=sba_artifact,
            )
        except (TypeError, ValueError) as exc:
            add("policy_output_contract_invalid", str(exc), "payload")
    return issues


class PpaValidator(ValidationActor):
    actor_id = "ppa_validator"

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
                repair_owner="ppa",
                failed_trace_id=pending_failure.failed_trace_id,
                invalidated_artifact_ids=(),
            )
        if not isinstance(current.payload, PpaProposal):
            raise TypeError("PpaValidator requires PpaProposal")
        sba_artifact = find_latest_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=SbaArtifact,
        )
        trigger = (
            self.journal.get_event(current.in_reply_to)
            if current.in_reply_to
            else current
        )
        issues = validate_ppa_proposal(
            proposal=current.payload,
            sba_artifact=sba_artifact,
            target_agent=current.target_agent,
            message_type=current.message_type,
            trigger_message_type=trigger.message_type,
            observation_queries=observation_query_names(
                journal=self.journal,
                event_ids=current.input_artifact_ids,
            ),
            journal=self.journal,
            input_artifact_ids=current.input_artifact_ids,
        )
        if current.payload.planning_status == "needs_upstream_reground" and not issues:
            semantic_issues = [
                ValidationIssue(code="upstream_evidence_required", message=item)
                for item in [
                    *current.payload.missing_evidence,
                    *current.payload.upstream_requests,
                ]
            ]
            return self.repair(
                current=current,
                issues=semantic_issues,
                repair_owner="sba",
                failed_trace_id=None,
                invalidated_artifact_ids=(),
            )
        if current.payload.planning_status == "partial_plan" and not issues:
            return EventEmission(
                decision=CoordinationDecision[PlanningBlocker](
                    target_agent="episode_controller",
                    message_type="handoff",
                    payload=PlanningBlocker(
                        rationale=current.payload.rationale,
                        missing_evidence=current.payload.missing_evidence,
                        blocked_targets=current.payload.blocked_targets,
                        upstream_requests=current.payload.upstream_requests,
                        planner_conflicts=current.payload.planner_conflicts,
                        repair_owner=(
                            "sba"
                            if current.payload.missing_evidence
                            or current.payload.upstream_requests
                            else "ppa"
                        ),
                    ),
                ),
                in_reply_to=current.event_id,
                input_artifact_ids=(current.event_id,),
                invalidated_artifact_ids=(),
            )
        if issues:
            return self.repair(
                current=current,
                issues=issues,
                repair_owner="ppa",
                failed_trace_id=None,
                invalidated_artifact_ids=(current.event_id,),
            )
        return self.accept(
            current=current,
            target_agent="ppa_compiler",
            notes=["PPA proposal accepted"],
            resolved_actor="ppa",
        )


class PpaCompiler:
    actor_id = "ppa_compiler"

    def __init__(self, *, journal: EpisodeJournal) -> None:
        self.journal = journal

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ValidationResultPayload):
            raise TypeError("PpaCompiler requires ValidationResultPayload")
        if current.in_reply_to is None:
            raise ValueError("PPA validation result must reference its proposal")
        proposal_event = self.journal.get_event(current.in_reply_to)
        if not isinstance(proposal_event.payload, PpaProposal):
            raise TypeError("PPA validation result does not reference PpaProposal")
        proposal = proposal_event.payload
        sba_artifact = find_latest_event_payload(
            journal=self.journal,
            current_event=proposal_event,
            payload_type=SbaArtifact,
        )
        action = compile_ppa_execution_action(
            proposal=proposal,
            sba_artifact=sba_artifact,
        )
        return EventEmission(
            decision=CoordinationDecision[ExecutionAction](
                target_agent="policy_mediator",
                message_type="handoff",
                payload=action,
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(proposal_event.event_id, current.event_id),
            invalidated_artifact_ids=(),
        )


__all__ = [
    "AmPolicySpec",
    "PartialPolicySpec",
    "PpaActor",
    "PpaCompiler",
    "PpaProposal",
    "PpaValidator",
    "RouteSelectionParameterSetSpec",
    "SmPolicySpec",
    "SnssaiSpec",
    "TrafficDescriptorSpec",
    "UrspPolicySpec",
    "compile_ppa_execution_action",
    "validate_ppa_proposal",
]
