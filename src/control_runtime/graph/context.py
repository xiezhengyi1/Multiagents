from __future__ import annotations

from collections.abc import Callable

from pydantic import Field, model_validator

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


class ContextBuilder:
    """Render exact event dependencies without summaries or hidden state."""

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
        rendered = "\n".join(event.model_dump_json() for event in events)
        token_count = self.count_tokens(rendered)
        if token_count > self.token_budget:
            raise ContextWindowExceeded(
                f"context requires {token_count} tokens but budget is {self.token_budget}"
            )
        return (
            ContextView(source_event_ids=[event.event_id for event in events]),
            rendered,
        )

    def _collect_dependencies(
        self,
        current_event: CoordinationEvent[object],
    ) -> list[CoordinationEvent[object]]:
        collected: dict[str, CoordinationEvent[object]] = {}

        def visit(event: CoordinationEvent[object]) -> None:
            if event.event_id in collected:
                return
            reference_ids = [
                *event.input_artifact_ids,
                *event.invalidated_artifact_ids,
                *([event.in_reply_to] if event.in_reply_to is not None else []),
            ]
            for reference_id in reference_ids:
                visit(self.journal.get_event(reference_id))
            collected[event.event_id] = event

        visit(current_event)
        return sorted(collected.values(), key=lambda event: event.event_index)


__all__ = ["ContextBuilder", "ContextView", "ContextWindowExceeded"]
