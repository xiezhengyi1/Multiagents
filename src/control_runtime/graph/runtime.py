from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from shared.contracts.coordination import (
    CoordinationDecision,
    CoordinationEvent,
    CoordinationPayload,
    MessageType,
)
from shared.runtime.journal import EpisodeJournal


class EpisodeGraphState(TypedDict):
    current_event: CoordinationEvent[Any]
    pending_model_failure: ModelFailure | None


@dataclass(frozen=True, slots=True)
class AllowedEdge:
    source_agent: str
    target_agent: str
    message_type: MessageType


@dataclass(frozen=True, slots=True)
class EventEmission:
    decision: CoordinationDecision[Any]
    in_reply_to: str | None
    input_artifact_ids: tuple[str, ...]
    invalidated_artifact_ids: tuple[str, ...]
    round_delta: Literal[0, 1] = 0


@dataclass(frozen=True, slots=True)
class ModelFailure:
    validator_actor: str
    failed_trace_id: str
    error: str


ActorResult = EventEmission | ModelFailure
ActorHandler = Callable[[EpisodeGraphState], ActorResult]


class EventRouter:
    def __init__(
        self,
        edges: Iterable[AllowedEdge],
        *,
        controller_actor: str,
        interceptors: dict[tuple[str, MessageType], str] | None = None,
    ) -> None:
        self.controller_actor = str(controller_actor).strip()
        if not self.controller_actor:
            raise ValueError("controller_actor is required")
        self._edges = {
            (edge.source_agent, edge.target_agent, edge.message_type) for edge in edges
        }
        if not self._edges:
            raise ValueError("at least one allowed edge is required")
        self._interceptors = dict(interceptors or {})

    def validate(self, event: CoordinationEvent[Any]) -> None:
        if (
            event.message_type == "terminal"
            and event.source_agent != self.controller_actor
        ):
            raise ValueError("only episode_controller may emit terminal")
        if (event.source_agent, event.message_type) in self._interceptors:
            return
        edge = (event.source_agent, event.target_agent, event.message_type)
        if edge not in self._edges:
            raise ValueError(
                "coordination edge is not allowed: "
                f"{event.source_agent} -> {event.target_agent} ({event.message_type})"
            )

    def validate_decision(
        self,
        *,
        source_agent: str,
        decision: CoordinationDecision[CoordinationPayload],
    ) -> None:
        if (
            decision.message_type == "terminal"
            and source_agent != self.controller_actor
        ):
            raise ValueError("only episode_controller may emit terminal")
        if (source_agent, decision.message_type) in self._interceptors:
            return
        edge = (source_agent, decision.target_agent, decision.message_type)
        if edge not in self._edges:
            raise ValueError(
                "coordination edge is not allowed: "
                f"{source_agent} -> {decision.target_agent} ({decision.message_type})"
            )

    def is_allowed(
        self,
        *,
        source_agent: str,
        target_agent: str,
        message_type: MessageType,
    ) -> bool:
        return (source_agent, target_agent, message_type) in self._edges

    def route_actor(self, event: CoordinationEvent[Any]) -> str:
        return self._interceptors.get(
            (event.source_agent, event.message_type),
            event.target_agent,
        )


class EpisodeGraph:
    """A serial, bidirectional LangGraph whose durable boundary is CoordinationEvent."""

    def __init__(
        self,
        *,
        journal: EpisodeJournal,
        router: EventRouter,
        handlers: dict[str, ActorHandler],
        retry_policies: dict[str, RetryPolicy | None],
    ) -> None:
        if not handlers:
            raise ValueError("at least one actor handler is required")
        self.journal = journal
        self.router = router
        self.handlers = dict(handlers)
        if set(retry_policies) != set(self.handlers):
            raise ValueError(
                "retry_policies must name every actor handler exactly once"
            )
        self.retry_policies = dict(retry_policies)
        self.graph = self._compile()

    def _compile(self) -> Any:
        builder = StateGraph(EpisodeGraphState)
        builder.add_node("route", self._route_node)
        for actor_id in self.handlers:
            builder.add_node(
                actor_id,
                self._actor_node(actor_id),
                retry_policy=self.retry_policies[actor_id],
            )
        builder.add_edge(START, "route")
        destinations: dict[str, str] = {
            actor_id: actor_id for actor_id in self.handlers
        }
        destinations[END] = END
        builder.add_conditional_edges("route", self._next_actor, destinations)
        for actor_id in self.handlers:
            builder.add_edge(actor_id, "route")
        return builder.compile()

    def _route_node(self, state: EpisodeGraphState) -> dict[str, Any]:
        self.router.validate(state["current_event"])
        return {}

    def _next_actor(self, state: EpisodeGraphState) -> str:
        pending_failure = state.get("pending_model_failure")
        if pending_failure is not None:
            if pending_failure.validator_actor not in self.handlers:
                raise ValueError(
                    f"no handler registered for validator {pending_failure.validator_actor}"
                )
            return pending_failure.validator_actor
        event = state["current_event"]
        if event.message_type == "terminal":
            return END
        actor_id = self.router.route_actor(event)
        if actor_id not in self.handlers:
            raise ValueError(f"no handler registered for actor {actor_id}")
        return actor_id

    def _actor_node(
        self, actor_id: str
    ) -> Callable[[EpisodeGraphState], dict[str, Any]]:
        handler = self.handlers[actor_id]

        def run(state: EpisodeGraphState) -> dict[str, Any]:
            current = state["current_event"]
            pending_failure = state.get("pending_model_failure")
            routed_actor = (
                pending_failure.validator_actor
                if pending_failure is not None
                else self.router.route_actor(current)
            )
            if routed_actor != actor_id:
                raise ValueError(
                    f"actor {actor_id} received event routed to {routed_actor}"
                )
            result = handler(state)
            if isinstance(result, ModelFailure):
                if pending_failure is not None:
                    raise ValueError(
                        "validator cannot replace an unresolved model failure"
                    )
                return {"pending_model_failure": result}
            emission = result
            self.router.validate_decision(
                source_agent=actor_id,
                decision=emission.decision,
            )
            event = self.journal.append_event(
                round_index=current.scope.round_index + emission.round_delta,
                source_agent=actor_id,
                target_agent=emission.decision.target_agent,
                message_type=emission.decision.message_type,
                payload=emission.decision.payload,
                in_reply_to=emission.in_reply_to,
                input_artifact_ids=emission.input_artifact_ids,
                invalidated_artifact_ids=emission.invalidated_artifact_ids,
            )
            return {"current_event": event, "pending_model_failure": None}

        return run

    def invoke(self, initial_event: CoordinationEvent[Any]) -> CoordinationEvent[Any]:
        result = self.graph.invoke(
            {"current_event": initial_event, "pending_model_failure": None}
        )
        return result["current_event"]


__all__ = [
    "ActorHandler",
    "ActorResult",
    "AllowedEdge",
    "EpisodeGraph",
    "EpisodeGraphState",
    "EventEmission",
    "EventRouter",
    "ModelFailure",
]
