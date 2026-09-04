from __future__ import annotations

from typing import Literal, TypeVar

from shared.contracts import NonEmptyStr
from shared.contracts.coordination import (
    CoordinationEvent,
    CoordinationPayload,
    HandoffPayload,
)
from shared.runtime.journal import EpisodeJournal


class ControlRequest(HandoffPayload):
    user_input: NonEmptyStr


class PlanningBlocker(HandoffPayload):
    rationale: NonEmptyStr
    missing_evidence: list[NonEmptyStr]
    blocked_targets: list[NonEmptyStr]
    upstream_requests: list[NonEmptyStr]
    planner_conflicts: list[NonEmptyStr]
    repair_owner: Literal["sba", "ppa"]


PayloadT = TypeVar("PayloadT", bound=CoordinationPayload)


def find_event_payload(
    *,
    journal: EpisodeJournal,
    current_event: CoordinationEvent[object],
    payload_type: type[PayloadT],
) -> PayloadT:
    found: list[PayloadT] = []
    visited: set[str] = set()

    def visit(event: CoordinationEvent[object]) -> None:
        if event.event_id in visited:
            return
        visited.add(event.event_id)
        if isinstance(event.payload, payload_type):
            found.append(event.payload)
        references = [
            *event.input_artifact_ids,
            *event.invalidated_artifact_ids,
            *([event.in_reply_to] if event.in_reply_to is not None else []),
        ]
        for event_id in references:
            visit(journal.get_event(event_id))

    visit(current_event)
    unique = list(dict.fromkeys(id(payload) for payload in found))
    if len(unique) != 1:
        raise ValueError(
            f"expected exactly one {payload_type.__name__} in event ancestry, found {len(unique)}"
        )
    return found[0]


def find_latest_event_payload(
    *,
    journal: EpisodeJournal,
    current_event: CoordinationEvent[object],
    payload_type: type[PayloadT],
) -> PayloadT:
    candidates: list[tuple[int, PayloadT]] = []
    visited: set[str] = set()

    def visit(event: CoordinationEvent[object]) -> None:
        if event.event_id in visited:
            return
        visited.add(event.event_id)
        if isinstance(event.payload, payload_type):
            candidates.append((event.event_index, event.payload))
        references = [
            *event.input_artifact_ids,
            *event.invalidated_artifact_ids,
            *([event.in_reply_to] if event.in_reply_to is not None else []),
        ]
        for event_id in references:
            visit(journal.get_event(event_id))

    visit(current_event)
    if not candidates:
        raise ValueError(f"no {payload_type.__name__} exists in event ancestry")
    return max(candidates, key=lambda item: item[0])[1]


__all__ = [
    "ControlRequest",
    "PlanningBlocker",
    "find_event_payload",
    "find_latest_event_payload",
]
