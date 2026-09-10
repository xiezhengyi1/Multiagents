from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import Field, model_validator

from control_runtime.simulation.contracts import (
    EnvironmentObservation,
    EnvironmentSnapshot,
    ExecutionFeedback,
)
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import CoordinationEvent
from shared.runtime.journal import EpisodeJournal


class ContextView(ContractModel):
    source_event_ids: list[NonEmptyStr] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> ContextView:
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("ContextView source_event_ids must be unique")
        return self


class ContextWindowExceeded(RuntimeError):
    pass


_ENTITY_FIELDS = {
    "supi",
    "app_id",
    "flow_id",
    "association_id",
    "snapshot_id",
    "session_id",
    "operation_id",
    "source_snapshot_id",
    "base_snapshot_id",
}
_PLURAL_ENTITY_FIELDS = {
    "app_ids": "app_id",
    "flow_ids": "flow_id",
    "affected_operation_ids": "operation_id",
}


def render_unified_representation(
    events: Sequence[CoordinationEvent[object]],
) -> str:
    """Render exact event data as a shared entity, fact, event, delta table."""

    entities: list[list[str]] = []
    entity_ids: dict[tuple[str, str], str] = {}
    facts: list[list[Any]] = []
    fact_ids: dict[str, str] = {}
    agents: list[str] = []
    messages: list[str] = []

    def entity_id(kind: str, value: str) -> str:
        key = (kind, value)
        if key not in entity_ids:
            identifier = f"e{len(entity_ids) + 1}"
            entity_ids[key] = identifier
            entities.append([identifier, kind, value])
        return entity_ids[key]

    def encode_value(value: Any, field: str | None = None) -> Any:
        if isinstance(value, dict):
            return {key: encode_value(item, key) for key, item in value.items()}
        if isinstance(value, list):
            item_field = _PLURAL_ENTITY_FIELDS.get(field)
            return [encode_value(item, item_field) for item in value]
        if isinstance(value, str) and value and field in _ENTITY_FIELDS:
            return {"@": entity_id(field, value)}
        return value

    def index(values: list[str], value: str) -> int:
        if value not in values:
            values.append(value)
        return values.index(value)

    def fact_id(payload: Any) -> str:
        encoded = encode_value(payload)
        fingerprint = json.dumps(
            encoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if fingerprint not in fact_ids:
            identifier = f"f{len(fact_ids) + 1}"
            fact_ids[fingerprint] = identifier
            facts.append([identifier, encoded])
        return fact_ids[fingerprint]

    rows: list[list[Any]] = []
    deltas: list[list[str]] = []
    episode_id = ""
    for event in events:
        raw = event.model_dump(mode="json")
        episode_id = raw["scope"]["episode_id"]
        event_ref = entity_id("event", raw["event_id"])
        payload_ref = fact_id(raw["payload"])
        rows.append(
            [
                event_ref,
                raw["event_index"],
                raw["scope"]["round_index"],
                index(agents, raw["source_agent"]),
                index(agents, raw["target_agent"]),
                index(messages, raw["message_type"]),
                (
                    entity_id("event", raw["in_reply_to"])
                    if raw["in_reply_to"] is not None
                    else None
                ),
                [entity_id("event", value) for value in raw["input_artifact_ids"]],
                [
                    entity_id("event", value)
                    for value in raw["invalidated_artifact_ids"]
                ],
            ]
        )
        deltas.append([event_ref, payload_ref])
    return json.dumps(
        {
            "format": "uir/v1",
            "scope": episode_id,
            "entities": entities,
            "facts": facts,
            "agents": agents,
            "messages": messages,
            "event_columns": [
                "event",
                "index",
                "round",
                "source",
                "target",
                "message",
                "reply",
                "inputs",
                "invalidated",
            ],
            "events": rows,
            "deltas": deltas,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class ContextBuilder:
    """Render exact event dependencies as a lossless shared representation."""

    def __init__(
        self,
        *,
        journal: EpisodeJournal,
        token_budget: int,
        count_tokens: Callable[[str], int],
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.journal = journal
        self.token_budget = token_budget
        self.count_tokens = count_tokens

    def build(
        self, current_event: CoordinationEvent[object]
    ) -> tuple[ContextView, str]:
        events = self._collect_dependencies(current_event)
        rendered = self.render(events)
        return (
            ContextView(source_event_ids=[event.event_id for event in events]),
            rendered,
        )

    def render(self, events: Sequence[CoordinationEvent[object]]) -> str:
        rendered = render_unified_representation(events)
        token_count = self.count_tokens(rendered)
        if token_count > self.token_budget:
            raise ContextWindowExceeded(
                f"context requires {token_count} tokens but budget is {self.token_budget}"
            )
        return rendered

    def _collect_dependencies(
        self,
        current_event: CoordinationEvent[object],
    ) -> list[CoordinationEvent[object]]:
        collected: dict[str, CoordinationEvent[object]] = {}

        def collect(event: CoordinationEvent[object]) -> None:
            if event.event_id in collected:
                return
            collected[event.event_id] = event
            reference_ids = [
                *event.input_artifact_ids,
                *event.invalidated_artifact_ids,
                *([event.in_reply_to] if event.in_reply_to is not None else []),
            ]
            for reference_id in reference_ids:
                collect(self.journal.get_event(reference_id))

        collect(current_event)
        events = sorted(collected.values(), key=lambda event: event.event_index)
        if current_event.target_agent == "main":
            return [
                event
                for event in events
                if not isinstance(event.payload, ExecutionFeedback)
            ]
        if current_event.target_agent != "ppa":
            return [
                event
                for event in events
                if not isinstance(
                    event.payload, (EnvironmentSnapshot, ExecutionFeedback)
                )
            ]
        latest_observations: dict[str, str] = {}
        for event in events:
            if (
                event.target_agent == "ppa"
                and isinstance(event.payload, EnvironmentObservation)
            ):
                latest_observations[event.payload.query.model_dump_json()] = event.event_id
        return [
            event
            for event in events
            if not (
                isinstance(event.payload, (EnvironmentSnapshot, ExecutionFeedback))
                or (
                    event.target_agent == "ppa"
                    and isinstance(event.payload, EnvironmentObservation)
                    and latest_observations[event.payload.query.model_dump_json()]
                    != event.event_id
                )
            )
        ]


__all__ = [
    "ContextBuilder",
    "ContextView",
    "ContextWindowExceeded",
    "render_unified_representation",
]
