from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from langchain.agents.structured_output import StructuredOutputError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from control_runtime.graph import (
    ContextBuilder,
    ContextView,
    EpisodeGraphState,
    EventEmission,
    ModelFailure,
)
from control_runtime.simulation import SimulatorAdapter, SimulatorObservationSink
from shared.agents.factory import AgentSpec, create_structured_agent
from shared.contracts.coordination import CoordinationDecision, ProposalPayload
from shared.runtime.journal import EpisodeJournal
from shared.runtime.langchain_trace import AgentRunContext


class StructuredModelActor:
    """Shared LangChain execution boundary for model-authored graph actors."""

    actor_id: ClassVar[str]
    validator_actor: ClassVar[str]
    system_prompt: ClassVar[str]
    response_schema: ClassVar[type[Any]]
    prompt_builder: ClassVar[Callable[[str], str]]
    default_tools: ClassVar[Sequence[BaseTool]] = ()
    requires_simulator_adapter: ClassVar[bool] = False

    def __init__(
        self,
        *,
        agent: Any,
        model_name: str,
        journal: EpisodeJournal,
        context_builder: ContextBuilder,
        simulator_adapter: SimulatorAdapter | None = None,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        self.agent = agent
        self.model_name = model_name
        self.journal = journal
        self.context_builder = context_builder
        self.simulator_adapter = simulator_adapter
        self.tool_names = tuple(tool.name for tool in tools)

    @classmethod
    def from_model(
        cls,
        *,
        model: BaseChatModel,
        model_name: str,
        journal: EpisodeJournal,
        context_builder: ContextBuilder,
        tools: Sequence[BaseTool] | None = None,
        simulator_adapter: SimulatorAdapter | None = None,
    ) -> StructuredModelActor:
        selected_tools = cls.default_tools if tools is None else tools
        if cls.requires_simulator_adapter and simulator_adapter is None:
            raise ValueError(f"{cls.actor_id} requires simulator_adapter")
        agent = create_structured_agent(
            model=model,
            spec=AgentSpec(
                name=cls.actor_id,
                system_prompt=cls.system_prompt,
                response_schema=cls.response_schema,
                tools=selected_tools,
            ),
        )
        return cls(
            agent=agent,
            model_name=model_name,
            journal=journal,
            context_builder=context_builder,
            simulator_adapter=simulator_adapter,
            tools=selected_tools,
        )

    def render_prompt(self, context_jsonl: str) -> str:
        return self.prompt_builder(context_jsonl)

    def prepare_observations(
        self,
        *,
        current: Any,
        observation_sink: SimulatorObservationSink | None,
    ) -> tuple[str, ...]:
        return ()

    def __call__(self, state: EpisodeGraphState) -> EventEmission | ModelFailure:
        current = state["current_event"]
        observation_sink = (
            SimulatorObservationSink(
                adapter=self.simulator_adapter,
                journal=self.journal,
                trigger_event=current,
                target_agent=self.actor_id,
            )
            if self.simulator_adapter is not None
            else None
        )
        prepared_evidence_ids = self.prepare_observations(
            current=current, observation_sink=observation_sink
        )
        view, context_jsonl = self.context_builder.build(current)
        if prepared_evidence_ids:
            source_event_ids = [*view.source_event_ids, *prepared_evidence_ids]
            view = ContextView(
                source_event_ids=list(dict.fromkeys(source_event_ids))
            )
            context_jsonl = self.context_builder.render(
                [self.journal.get_event(event_id) for event_id in view.source_event_ids]
            )
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=self.render_prompt(context_jsonl))]},
                context=AgentRunContext(
                    journal=self.journal,
                    trigger_event_id=current.event_id,
                    round_index=current.scope.round_index,
                    agent_name=self.actor_id,
                    model_name=self.model_name,
                    observation_sink=observation_sink,
                ),
            )
        except StructuredOutputError as exc:
            trace = self.journal.latest_trace(
                trigger_event_id=current.event_id,
                agent_name=self.actor_id,
            )
            return ModelFailure(
                validator_actor=self.validator_actor,
                failed_trace_id=trace.trace_id,
                error=str(exc),
            )
        decision = self.response_schema.model_validate(result["structured_response"])
        if not isinstance(decision, CoordinationDecision):
            raise TypeError(
                "structured model actor response must be CoordinationDecision"
            )
        if not isinstance(decision.payload, ProposalPayload):
            raise TypeError("structured model actor must produce a ProposalPayload")
        expected_message_type = (
            "repair_response"
            if current.message_type == "repair_request"
            else "proposal"
        )
        decision = decision.model_copy(
            update={"message_type": expected_message_type}
        )
        evidence_ids = (
            observation_sink.event_ids if observation_sink is not None else ()
        )
        return EventEmission(
            decision=decision,
            in_reply_to=current.event_id,
            input_artifact_ids=tuple(
                dict.fromkeys((*view.source_event_ids, *evidence_ids))
            ),
            invalidated_artifact_ids=(),
        )


__all__ = ["StructuredModelActor"]
