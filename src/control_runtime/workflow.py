from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from control_runtime.actors import (
    AssuranceDiagnoser,
    ControlRequest,
    EpisodeController,
    ExecutionDispatcher,
    MainActor,
    MainValidator,
    OutcomeEvaluator,
    PolicyMediator,
    PpaActor,
    PpaCompiler,
    PpaValidator,
    SbaActor,
    SbaCompiler,
    SbaValidator,
    SingleControlActor,
    SingleControlCompiler,
    SingleControlValidator,
)
from control_runtime.graph import (
    AllowedEdge,
    ContextBuilder,
    EpisodeGraph,
    EventRouter,
    RepairBudget,
    RepairLimits,
)
from control_runtime.simulation import SimulatorAdapter
from shared.agents import transport_retry_policy
from shared.contracts.coordination import CoordinationEvent
from shared.runtime.journal import EpisodeJournal


@dataclass(frozen=True, slots=True)
class ModelBinding:
    model: BaseChatModel
    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model binding name is required")


@dataclass(frozen=True, slots=True)
class ControlModels:
    main: ModelBinding
    sba: ModelBinding
    ppa: ModelBinding


@dataclass(frozen=True, slots=True)
class ControlTools:
    main: tuple[BaseTool, ...]
    sba: tuple[BaseTool, ...]
    ppa: tuple[BaseTool, ...]


@dataclass(frozen=True, slots=True)
class SingleControlTools:
    extra: tuple[BaseTool, ...]


class ControlWorkflow:
    """Canonical multi-agent control workflow and dataset-generation boundary."""

    def __init__(
        self,
        *,
        journal: EpisodeJournal,
        simulator_adapter: SimulatorAdapter,
        graph: EpisodeGraph,
    ) -> None:
        self.journal = journal
        self.simulator_adapter = simulator_adapter
        self.graph = graph

    @classmethod
    def create(
        cls,
        *,
        dataset_root: Path,
        episode_id: str,
        simulator_adapter: SimulatorAdapter,
        models: ControlModels,
        tools: ControlTools,
        repair_limits: RepairLimits,
        outcome_evaluator: OutcomeEvaluator,
        budget_exhausted_reward_components: dict[str, float],
        context_token_budget: int,
        count_tokens: Callable[[str], int],
        transport_max_attempts: int,
    ) -> ControlWorkflow:
        expected_limits = {"main", "sba", "ppa"}
        if set(repair_limits.local_limits) != expected_limits:
            raise ValueError("repair limits must define main, sba, and ppa")
        if repair_limits.upstream != {
            "main": "episode_controller",
            "sba": "main",
            "ppa": "sba",
        }:
            raise ValueError(
                "repair escalation must be ppa -> sba -> main -> episode_controller"
            )
        if repair_limits.controller_actor != "episode_controller":
            raise ValueError("repair controller must be episode_controller")

        journal = EpisodeJournal(
            root=Path(dataset_root) / "multi_agent_control",
            episode_id=episode_id,
        )
        context_builder = ContextBuilder(
            journal=journal,
            token_budget=context_token_budget,
            count_tokens=count_tokens,
        )
        repair_budget = RepairBudget(repair_limits)
        main = MainActor.from_model(
            model=models.main.model,
            model_name=models.main.name,
            journal=journal,
            context_builder=context_builder,
            tools=tools.main,
        )
        sba = SbaActor.from_model(
            model=models.sba.model,
            model_name=models.sba.name,
            journal=journal,
            context_builder=context_builder,
            tools=(*SbaActor.default_tools, *tools.sba),
            simulator_adapter=simulator_adapter,
        )
        ppa = PpaActor.from_model(
            model=models.ppa.model,
            model_name=models.ppa.name,
            journal=journal,
            context_builder=context_builder,
            tools=(*PpaActor.default_tools, *tools.ppa),
            simulator_adapter=simulator_adapter,
        )
        handlers = {
            "main": main,
            "main_validator": MainValidator(
                journal=journal, repair_budget=repair_budget
            ),
            "sba": sba,
            "sba_validator": SbaValidator(journal=journal, repair_budget=repair_budget),
            "sba_compiler": SbaCompiler(journal=journal),
            "ppa": ppa,
            "ppa_validator": PpaValidator(journal=journal, repair_budget=repair_budget),
            "ppa_compiler": PpaCompiler(journal=journal),
            "policy_mediator": PolicyMediator(),
            "execution_dispatcher": ExecutionDispatcher(
                journal=journal, simulator_adapter=simulator_adapter
            ),
            "assurance_diagnoser": AssuranceDiagnoser(
                journal=journal,
                outcome_evaluator=outcome_evaluator,
                repair_owner="ppa",
            ),
            "episode_controller": EpisodeController(
                repair_budget=repair_budget,
                budget_exhausted_reward_components=budget_exhausted_reward_components,
            ),
        }
        model_retry = transport_retry_policy(max_attempts=transport_max_attempts)
        retry_policies = {
            actor_id: model_retry if actor_id in {"main", "sba", "ppa"} else None
            for actor_id in handlers
        }
        router = EventRouter(
            _control_edges(),
            controller_actor="episode_controller",
            interceptors={
                (actor, message_type): f"{actor}_validator"
                for actor in ("main", "sba", "ppa")
                for message_type in ("proposal", "repair_response")
            },
        )
        graph = EpisodeGraph(
            journal=journal,
            router=router,
            handlers=handlers,
            retry_policies=retry_policies,
        )
        return cls(
            journal=journal,
            simulator_adapter=simulator_adapter,
            graph=graph,
        )

    def run(self, *, user_input: str) -> CoordinationEvent[object]:
        snapshot = self.simulator_adapter.capture_snapshot()
        self.journal.append_event(
            round_index=0,
            source_agent="simulator_adapter",
            target_agent="episode_controller",
            message_type="observation",
            payload=snapshot,
        )
        request_event = self.journal.append_event(
            round_index=0,
            source_agent="episode_controller",
            target_agent="main",
            message_type="handoff",
            payload=ControlRequest(user_input=user_input),
        )
        return self.graph.invoke(request_event)


class SingleControlWorkflow(ControlWorkflow):
    """One model actor with the same semantic contracts, tools, compiler, and executor."""

    @classmethod
    def create(
        cls,
        *,
        dataset_root: Path,
        episode_id: str,
        simulator_adapter: SimulatorAdapter,
        model: ModelBinding,
        tools: SingleControlTools,
        repair_limits: RepairLimits,
        outcome_evaluator: OutcomeEvaluator,
        budget_exhausted_reward_components: dict[str, float],
        context_token_budget: int,
        count_tokens: Callable[[str], int],
        transport_max_attempts: int,
    ) -> SingleControlWorkflow:
        if set(repair_limits.local_limits) != {"single_control"}:
            raise ValueError("single-agent repair limits must define single_control")
        if repair_limits.upstream != {"single_control": "episode_controller"}:
            raise ValueError(
                "single-agent repair escalation must terminate at episode_controller"
            )
        if repair_limits.controller_actor != "episode_controller":
            raise ValueError("repair controller must be episode_controller")

        journal = EpisodeJournal(
            root=Path(dataset_root) / "single_agent_control",
            episode_id=episode_id,
        )
        context_builder = ContextBuilder(
            journal=journal,
            token_budget=context_token_budget,
            count_tokens=count_tokens,
        )
        repair_budget = RepairBudget(repair_limits)
        single = SingleControlActor.from_model(
            model=model.model,
            model_name=model.name,
            journal=journal,
            context_builder=context_builder,
            tools=(*SingleControlActor.default_tools, *tools.extra),
            simulator_adapter=simulator_adapter,
        )
        handlers = {
            "single_control": single,
            "single_validator": SingleControlValidator(
                journal=journal,
                repair_budget=repair_budget,
            ),
            "single_compiler": SingleControlCompiler(journal=journal),
            "policy_mediator": PolicyMediator(),
            "execution_dispatcher": ExecutionDispatcher(
                journal=journal, simulator_adapter=simulator_adapter
            ),
            "assurance_diagnoser": AssuranceDiagnoser(
                journal=journal,
                outcome_evaluator=outcome_evaluator,
                repair_owner="single_control",
            ),
            "episode_controller": EpisodeController(
                repair_budget=repair_budget,
                budget_exhausted_reward_components=budget_exhausted_reward_components,
            ),
        }
        router = EventRouter(
            _single_control_edges(),
            controller_actor="episode_controller",
            interceptors={
                ("single_control", message_type): "single_validator"
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
                    if actor_id == "single_control"
                    else None
                )
                for actor_id in handlers
            },
        )
        return cls(
            journal=journal,
            simulator_adapter=simulator_adapter,
            graph=graph,
        )

    def run(self, *, user_input: str) -> CoordinationEvent[object]:
        snapshot = self.simulator_adapter.capture_snapshot()
        self.journal.append_event(
            round_index=0,
            source_agent="simulator_adapter",
            target_agent="episode_controller",
            message_type="observation",
            payload=snapshot,
        )
        request_event = self.journal.append_event(
            round_index=0,
            source_agent="episode_controller",
            target_agent="single_control",
            message_type="handoff",
            payload=ControlRequest(user_input=user_input),
        )
        return self.graph.invoke(request_event)


def _control_edges() -> list[AllowedEdge]:
    edge_specs = [
        ("episode_controller", "main", "handoff"),
        ("main_validator", "sba", "validation_result"),
        ("main_validator", "ppa", "validation_result"),
        ("main_validator", "main", "repair_request"),
        ("main_validator", "episode_controller", "repair_request"),
        ("sba_validator", "sba_compiler", "validation_result"),
        ("sba_validator", "sba", "repair_request"),
        ("sba_validator", "main", "repair_request"),
        ("sba_validator", "episode_controller", "repair_request"),
        ("sba_compiler", "ppa", "handoff"),
        ("ppa_validator", "ppa_compiler", "validation_result"),
        ("ppa_validator", "ppa", "repair_request"),
        ("ppa_validator", "sba", "repair_request"),
        ("ppa_validator", "main", "repair_request"),
        ("ppa_validator", "episode_controller", "repair_request"),
        ("ppa_validator", "episode_controller", "handoff"),
        ("ppa_compiler", "policy_mediator", "handoff"),
        ("policy_mediator", "execution_dispatcher", "diagnosis"),
        ("policy_mediator", "episode_controller", "diagnosis"),
        ("execution_dispatcher", "assurance_diagnoser", "execution_feedback"),
        ("assurance_diagnoser", "episode_controller", "diagnosis"),
        ("episode_controller", "ppa", "repair_request"),
        ("episode_controller", "sba", "repair_request"),
        ("episode_controller", "episode_controller", "terminal"),
    ]
    return [
        AllowedEdge(source_agent=source, target_agent=target, message_type=message_type)
        for source, target, message_type in edge_specs
    ]


def _single_control_edges() -> list[AllowedEdge]:
    edge_specs = [
        ("episode_controller", "single_control", "handoff"),
        ("single_validator", "single_compiler", "validation_result"),
        ("single_validator", "single_control", "repair_request"),
        ("single_validator", "episode_controller", "repair_request"),
        ("single_compiler", "policy_mediator", "handoff"),
        ("policy_mediator", "execution_dispatcher", "diagnosis"),
        ("policy_mediator", "episode_controller", "diagnosis"),
        ("execution_dispatcher", "assurance_diagnoser", "execution_feedback"),
        ("assurance_diagnoser", "episode_controller", "diagnosis"),
        ("episode_controller", "single_control", "repair_request"),
        ("episode_controller", "episode_controller", "terminal"),
    ]
    return [
        AllowedEdge(source_agent=source, target_agent=target, message_type=message_type)
        for source, target, message_type in edge_specs
    ]


__all__ = [
    "ControlModels",
    "ControlTools",
    "ControlWorkflow",
    "ModelBinding",
    "SingleControlTools",
    "SingleControlWorkflow",
]
