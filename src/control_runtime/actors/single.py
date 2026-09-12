from __future__ import annotations

from control_runtime.context.prompts.single import render_single_system_prompt
from control_runtime.graph import EpisodeGraphState, EventEmission
from control_runtime.simulation import (
    PPA_OBSERVATION_TOOLS,
    SBA_OBSERVATION_TOOLS,
    ExecutionAction,
    observation_query_names,
)
from shared.contracts.coordination import (
    CoordinationDecision,
    ProposalPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.runtime.journal import EpisodeJournal

from .common import ControlRequest, find_event_payload
from .main import MainProposal, validate_main_proposal
from .model import StructuredModelActor
from .ppa import PpaProposal, compile_ppa_execution_action, validate_ppa_proposal
from .sba import SbaArtifact, SbaProposal, validate_sba_proposal
from .validation import ValidationActor


class SingleControlProposal(ProposalPayload):
    """One model call containing the same three semantic stages as the graph baseline."""

    main: MainProposal
    sba: SbaProposal
    ppa: PpaProposal


SINGLE_SYSTEM_PROMPT = (
    render_single_system_prompt()
    + """

Coordination data contract for the single-agent baseline:
- Preserve every domain, grounding, tool-use, optimizer, entitlement, and policy rule above.
- Return one `CoordinationDecision[SingleControlProposal]` with root fields `target_agent`, `message_type`, and `payload`.
- Set `target_agent="single_compiler"`. Use `message_type="proposal"` normally and `message_type="repair_response"` when replying to a repair request.
- Put the Main-equivalent fields in `payload.main`, the IEA/SBA-equivalent fields in `payload.sba`, and the OSA/PPA-equivalent fields in `payload.ppa`.
- `payload.ppa.planning_status` must be `executable_plan`; incomplete evidence must be repaired inside this single actor rather than routed to another model actor.
- Runtime stamps event identity, scope, source agent, and evidence references.
"""
)


def build_single_prompt(context_jsonl: str) -> str:
    return (
        "User request, simulator snapshot, repair feedback, and tool observations are "
        "provided as lossless UIR tables. Complete the Main, SBA, and PPA semantic "
        "stages in one model invocation.\n\nCoordinator context:\n"
        f"{context_jsonl}"
    )


class SingleControlActor(StructuredModelActor):
    actor_id = "single_control"
    validator_actor = "single_validator"
    system_prompt = SINGLE_SYSTEM_PROMPT
    response_schema = CoordinationDecision[SingleControlProposal]
    prompt_builder = staticmethod(build_single_prompt)
    default_tools = (*SBA_OBSERVATION_TOOLS, *PPA_OBSERVATION_TOOLS)
    requires_simulator_adapter = True


class SingleControlValidator(ValidationActor):
    actor_id = "single_validator"

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
                repair_owner="single_control",
                failed_trace_id=pending_failure.failed_trace_id,
                invalidated_artifact_ids=(),
            )
        if not isinstance(current.payload, SingleControlProposal):
            raise TypeError("SingleControlValidator requires SingleControlProposal")

        proposal = current.payload
        request = find_event_payload(
            journal=self.journal,
            current_event=current,
            payload_type=ControlRequest,
        )
        trigger = (
            self.journal.get_event(current.in_reply_to)
            if current.in_reply_to
            else current
        )
        queries = observation_query_names(
            journal=self.journal,
            event_ids=current.input_artifact_ids,
        )
        expected_message = (
            "repair_response"
            if trigger.message_type == "repair_request"
            else "proposal"
        )
        issues: list[ValidationIssue] = []
        if current.target_agent != "single_compiler":
            issues.append(
                ValidationIssue(
                    code="invalid_target",
                    message="single control output must target single_compiler",
                    path="target_agent",
                )
            )
        if current.message_type != expected_message:
            issues.append(
                ValidationIssue(
                    code="invalid_message_type",
                    message=f"single control must emit {expected_message} for this trigger",
                    path="message_type",
                )
            )
        if not proposal.main.supi.strip():
            issues.append(
                ValidationIssue(
                    code="single_supi_required",
                    message="single-agent execution requires one explicit SUPI",
                    path="payload.main.supi",
                )
            )

        issues.extend(
            validate_main_proposal(
                proposal=proposal.main,
                target_agent="sba",
                message_type=current.message_type,
                trigger_message_type=trigger.message_type,
                user_input=request.user_input,
                round_index=current.scope.round_index,
            )
        )
        issues.extend(
            validate_sba_proposal(
                proposal=proposal.sba,
                main_proposal=proposal.main,
                target_agent="sba_compiler",
                message_type=current.message_type,
                trigger_message_type=trigger.message_type,
                observation_queries=queries,
            )
        )
        sba_artifact = SbaArtifact(
            supi=proposal.main.supi,
            **proposal.sba.model_dump(mode="python"),
        )
        ppa_target = {
            "executable_plan": "ppa_compiler",
            "needs_upstream_reground": "sba",
            "partial_plan": "episode_controller",
        }[proposal.ppa.planning_status]
        issues.extend(
            validate_ppa_proposal(
                proposal=proposal.ppa,
                sba_artifact=sba_artifact,
                target_agent=ppa_target,
                message_type=current.message_type,
                trigger_message_type=trigger.message_type,
                observation_queries=queries,
                journal=self.journal,
                input_artifact_ids=current.input_artifact_ids,
            )
        )
        if proposal.ppa.planning_status != "executable_plan":
            issues.append(
                ValidationIssue(
                    code="single_plan_not_executable",
                    message="single-agent baseline must repair to an executable plan",
                    path="payload.ppa.planning_status",
                )
            )
        if issues:
            return self.repair(
                current=current,
                issues=issues,
                repair_owner="single_control",
                failed_trace_id=None,
                invalidated_artifact_ids=(current.event_id,),
            )
        return self.accept(
            current=current,
            target_agent="single_compiler",
            notes=["single-agent semantic proposal accepted"],
            resolved_actor="single_control",
        )


class SingleControlCompiler:
    actor_id = "single_compiler"

    def __init__(self, *, journal: EpisodeJournal) -> None:
        self.journal = journal

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ValidationResultPayload):
            raise TypeError("SingleControlCompiler requires ValidationResultPayload")
        if current.in_reply_to is None:
            raise ValueError("single validation result must reference its proposal")
        proposal_event = self.journal.get_event(current.in_reply_to)
        if not isinstance(proposal_event.payload, SingleControlProposal):
            raise TypeError(
                "single validation result does not reference SingleControlProposal"
            )
        proposal = proposal_event.payload
        sba_artifact = SbaArtifact(
            supi=proposal.main.supi,
            **proposal.sba.model_dump(mode="python"),
        )
        action = compile_ppa_execution_action(
            proposal=proposal.ppa,
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
    "SINGLE_SYSTEM_PROMPT",
    "SingleControlActor",
    "SingleControlCompiler",
    "SingleControlProposal",
    "SingleControlValidator",
    "build_single_prompt",
]
