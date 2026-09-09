from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from control_runtime.actors.model import StructuredModelActor
from control_runtime.actors.validation import ValidationActor
from control_runtime.graph import (
    AllowedEdge,
    ContextBuilder,
    EpisodeGraph,
    EpisodeGraphState,
    EventEmission,
    EventRouter,
    RepairBudget,
    RepairLimits,
)
from shared.agents import transport_retry_policy
from shared.contracts.coordination import (
    CoordinationDecision,
    CoordinationEvent,
    ProposalPayload,
    RepairRequestPayload,
    TerminalPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.runtime.journal import EpisodeJournal

from .compiler import S3_QUALITY_PROFILE, EnvironmentCandidateValidator
from .contracts import EnvironmentArtifact, EnvironmentGenerationRequest
from .design import (
    EnvironmentCatalog,
    EnvironmentDesign,
    EnvironmentDesignCompiler,
    EnvironmentDesignError,
)
from .prompts import ENVIRONMENT_DESIGN_AGENT_SYSTEM_PROMPT


class EnvironmentProposal(ProposalPayload):
    catalog: EnvironmentCatalog
    design: EnvironmentDesign


def build_environment_prompt(context_jsonl: str) -> str:
    return (
        "Generate one complete semantic environment from the request in the lossless "
        "UIR tables. On repair, use the validation issues in those tables and "
        "return a complete replacement.\n\nCoordinator context:\n"
        f"{context_jsonl}"
    )


class EnvironmentGenerationActor(StructuredModelActor):
    actor_id = "environment_generator"
    validator_actor = "environment_validator"
    system_prompt = ENVIRONMENT_DESIGN_AGENT_SYSTEM_PROMPT
    response_schema = CoordinationDecision[EnvironmentProposal]
    prompt_builder = staticmethod(build_environment_prompt)


class EnvironmentValidator(ValidationActor):
    actor_id = "environment_validator"

    def __init__(
        self,
        *,
        journal: EpisodeJournal,
        repair_budget: RepairBudget,
        quality_profile: str | None,
    ) -> None:
        super().__init__(journal=journal, repair_budget=repair_budget)
        self.design_compiler = EnvironmentDesignCompiler()
        self.candidate_validator = EnvironmentCandidateValidator()
        self.quality_profile = quality_profile

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
                repair_owner="environment_generator",
                failed_trace_id=pending_failure.failed_trace_id,
                invalidated_artifact_ids=(),
            )
        if not isinstance(current.payload, EnvironmentProposal):
            raise TypeError("EnvironmentValidator requires EnvironmentProposal")
        trigger = (
            self.journal.get_event(current.in_reply_to)
            if current.in_reply_to
            else current
        )
        expected_message = (
            "repair_response"
            if trigger.message_type == "repair_request"
            else "proposal"
        )
        issues: list[ValidationIssue] = []
        if current.target_agent != "environment_compiler":
            issues.append(
                ValidationIssue(
                    code="invalid_target",
                    message="environment proposal must target environment_compiler",
                    path="target_agent",
                )
            )
        if current.message_type != expected_message:
            issues.append(
                ValidationIssue(
                    code="invalid_message_type",
                    message=f"environment generator must emit {expected_message}",
                    path="message_type",
                )
            )
        try:
            compiled = self.design_compiler.compile(
                current.payload.catalog,
                current.payload.design,
            )
        except EnvironmentDesignError as exc:
            issues.extend(
                ValidationIssue(code="environment_design_invalid", message=error)
                for error in exc.errors
            )
        else:
            report = self.candidate_validator.validate_candidate(
                compiled.candidate,
                quality_profile=self.quality_profile,
            )
            issues.extend(
                ValidationIssue(code="environment_candidate_invalid", message=error)
                for error in report.errors
            )
        if issues:
            return self.repair(
                current=current,
                issues=issues,
                repair_owner="environment_generator",
                failed_trace_id=None,
                invalidated_artifact_ids=(current.event_id,),
            )
        return self.accept(
            current=current,
            target_agent="environment_compiler",
            notes=["environment proposal accepted"],
            resolved_actor="environment_generator",
        )


class EnvironmentCompiler:
    actor_id = "environment_compiler"

    def __init__(self, *, journal: EpisodeJournal) -> None:
        self.journal = journal
        self.compiler = EnvironmentDesignCompiler()

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if not isinstance(current.payload, ValidationResultPayload):
            raise TypeError("EnvironmentCompiler requires ValidationResultPayload")
        if current.in_reply_to is None:
            raise ValueError(
                "environment validation result must reference its proposal"
            )
        proposal_event = self.journal.get_event(current.in_reply_to)
        if not isinstance(proposal_event.payload, EnvironmentProposal):
            raise TypeError("validation result does not reference EnvironmentProposal")
        compiled = self.compiler.compile(
            proposal_event.payload.catalog,
            proposal_event.payload.design,
        )
        return EventEmission(
            decision=CoordinationDecision[EnvironmentArtifact](
                target_agent="episode_controller",
                message_type="handoff",
                payload=EnvironmentArtifact(
                    candidate=compiled.candidate,
                    compilation_report=compiled.report,
                ),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(proposal_event.event_id, current.event_id),
            invalidated_artifact_ids=(),
        )


class EnvironmentController:
    actor_id = "episode_controller"

    def __init__(self, *, success_reward: float, failure_reward: float) -> None:
        self.success_reward = success_reward
        self.failure_reward = failure_reward

    def __call__(self, state: EpisodeGraphState) -> EventEmission:
        current = state["current_event"]
        if isinstance(current.payload, EnvironmentArtifact):
            outcome = "success"
            reason = "environment compiled and validated"
            components = {"valid_environment": self.success_reward}
        elif isinstance(current.payload, RepairRequestPayload):
            outcome = "failure"
            reason = "; ".join(issue.message for issue in current.payload.issues)
            components = {"invalid_environment": self.failure_reward}
        else:
            raise TypeError(
                "EnvironmentController requires EnvironmentArtifact or RepairRequestPayload"
            )
        return EventEmission(
            decision=CoordinationDecision[TerminalPayload](
                target_agent="episode_controller",
                message_type="terminal",
                payload=TerminalPayload(
                    outcome=outcome,
                    reason=reason,
                    reward_total=sum(components.values()),
                    reward_components=components,
                ),
            ),
            in_reply_to=current.event_id,
            input_artifact_ids=(current.event_id,),
            invalidated_artifact_ids=(),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentModel:
    model: BaseChatModel
    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("environment model name is required")


class EnvironmentWorkflow:
    """Environment generation graph with its own trace dataset partition."""

    def __init__(self, *, journal: EpisodeJournal, graph: EpisodeGraph) -> None:
        self.journal = journal
        self.graph = graph

    @classmethod
    def create(
        cls,
        *,
        dataset_root: Path,
        episode_id: str,
        model: EnvironmentModel,
        repair_limits: RepairLimits,
        context_token_budget: int,
        count_tokens: Callable[[str], int],
        transport_max_attempts: int,
        quality_profile: str | None = S3_QUALITY_PROFILE,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
    ) -> EnvironmentWorkflow:
        if set(repair_limits.local_limits) != {"environment_generator"}:
            raise ValueError(
                "environment repair limits must define environment_generator"
            )
        if repair_limits.upstream != {"environment_generator": "episode_controller"}:
            raise ValueError(
                "environment repair escalation must terminate at episode_controller"
            )
        if repair_limits.controller_actor != "episode_controller":
            raise ValueError("repair controller must be episode_controller")

        journal = EpisodeJournal(
            root=Path(dataset_root) / "environment",
            episode_id=episode_id,
        )
        context_builder = ContextBuilder(
            journal=journal,
            token_budget=context_token_budget,
            count_tokens=count_tokens,
        )
        repair_budget = RepairBudget(repair_limits)
        generator = EnvironmentGenerationActor.from_model(
            model=model.model,
            model_name=model.name,
            journal=journal,
            context_builder=context_builder,
        )
        handlers = {
            "environment_generator": generator,
            "environment_validator": EnvironmentValidator(
                journal=journal,
                repair_budget=repair_budget,
                quality_profile=quality_profile,
            ),
            "environment_compiler": EnvironmentCompiler(journal=journal),
            "episode_controller": EnvironmentController(
                success_reward=success_reward,
                failure_reward=failure_reward,
            ),
        }
        router = EventRouter(
            _environment_edges(),
            controller_actor="episode_controller",
            interceptors={
                ("environment_generator", message_type): "environment_validator"
                for message_type in ("proposal", "repair_response")
            },
        )
        graph = EpisodeGraph(
            journal=journal,
            router=router,
            handlers=handlers,
            retry_policies={
                actor_id: (
                    transport_retry_policy(max_attempts=transport_max_attempts)
                    if actor_id == "environment_generator"
                    else None
                )
                for actor_id in handlers
            },
        )
        return cls(journal=journal, graph=graph)

    def run(
        self,
        *,
        request: EnvironmentGenerationRequest,
    ) -> CoordinationEvent[object]:
        request_event = self.journal.append_event(
            round_index=0,
            source_agent="episode_controller",
            target_agent="environment_generator",
            message_type="handoff",
            payload=request,
        )
        return self.graph.invoke(request_event)


def _environment_edges() -> list[AllowedEdge]:
    edge_specs = [
        ("episode_controller", "environment_generator", "handoff"),
        ("environment_validator", "environment_compiler", "validation_result"),
        ("environment_validator", "environment_generator", "repair_request"),
        ("environment_validator", "episode_controller", "repair_request"),
        ("environment_compiler", "episode_controller", "handoff"),
        ("episode_controller", "episode_controller", "terminal"),
    ]
    return [
        AllowedEdge(source_agent=source, target_agent=target, message_type=message_type)
        for source, target, message_type in edge_specs
    ]


__all__ = [
    "EnvironmentCompiler",
    "EnvironmentController",
    "EnvironmentGenerationActor",
    "EnvironmentModel",
    "EnvironmentProposal",
    "EnvironmentValidator",
    "EnvironmentWorkflow",
]
