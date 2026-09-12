from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TypeVar

from shared.contracts.common import EventScope, NonEmptyStr, utc_now
from shared.contracts.coordination import (
    CoordinationEvent,
    CoordinationPayload,
    MessageType,
)
from shared.contracts.model_trace import (
    AgentTrace,
    ModelMessage,
    ModelToolSpec,
    TokenUsage,
)

PayloadT = TypeVar("PayloadT", bound=CoordinationPayload)


class EpisodeJournal:
    """Single-process append-only event and model-call journal for one episode."""

    def __init__(self, *, root: Path, episode_id: NonEmptyStr) -> None:
        self.episode_id = str(episode_id).strip()
        if not self.episode_id:
            raise ValueError("episode_id is required")
        self.directory = Path(root) / self.episode_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.directory / "events.jsonl"
        self.traces_path = self.directory / "traces.jsonl"
        self.events_path.touch()
        self.traces_path.touch()
        self._next_event_index = 0
        self._next_trace_index = 0
        self._event_ids: set[str] = set()
        self._events: dict[str, CoordinationEvent[CoordinationPayload]] = {}
        self._traces: list[AgentTrace] = []
        self._call_indices: dict[tuple[str, str], int] = {}
        self._lock = Lock()

    def append_event(
        self,
        *,
        round_index: int,
        source_agent: str,
        target_agent: str,
        message_type: MessageType,
        payload: PayloadT,
        in_reply_to: str | None = None,
        input_artifact_ids: Iterable[str] = (),
        invalidated_artifact_ids: Iterable[str] = (),
    ) -> CoordinationEvent[PayloadT]:
        with self._lock:
            event_index = self._next_event_index
            event = CoordinationEvent[PayloadT](
                event_id=f"{self.episode_id}:event:{event_index:06d}",
                event_index=event_index,
                scope=EventScope(episode_id=self.episode_id, round_index=round_index),
                source_agent=source_agent,
                target_agent=target_agent,
                message_type=message_type,
                in_reply_to=in_reply_to,
                input_artifact_ids=list(input_artifact_ids),
                invalidated_artifact_ids=list(invalidated_artifact_ids),
                payload=payload,
            )
            self._append_json(self.events_path, event.model_dump_json())
            self._event_ids.add(event.event_id)
            self._events[event.event_id] = event
            self._next_event_index += 1
            return event

    def append_trace(
        self,
        *,
        trigger_event_id: str,
        round_index: int,
        agent_name: str,
        model_name: str,
        input_messages: list[ModelMessage],
        output_message: ModelMessage | None,
        tools: list[ModelToolSpec],
        status: str,
        error: str | None = None,
        usage: TokenUsage | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> AgentTrace:
        with self._lock:
            if trigger_event_id not in self._event_ids:
                raise ValueError(f"unknown trigger_event_id: {trigger_event_id}")
            key = (trigger_event_id, str(agent_name).strip())
            call_index = self._call_indices.get(key, 0)
            trace_index = self._next_trace_index
            started = started_at or utc_now()
            ended = ended_at or utc_now()
            trace = AgentTrace(
                trace_id=f"{self.episode_id}:trace:{trace_index:06d}",
                trigger_event_id=trigger_event_id,
                call_index=call_index,
                scope=EventScope(episode_id=self.episode_id, round_index=round_index),
                agent_name=agent_name,
                model_name=model_name,
                started_at=started,
                ended_at=ended,
                input_messages=input_messages,
                output_message=output_message,
                tools=tools,
                status=status,
                error=error,
                usage=usage,
            )
            self._append_json(self.traces_path, trace.model_dump_json())
            self._call_indices[key] = call_index + 1
            self._next_trace_index += 1
            self._traces.append(trace)
            return trace

    def get_event(self, event_id: str) -> CoordinationEvent[CoordinationPayload]:
        try:
            return self._events[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown event_id: {event_id}") from exc

    def latest_trace(self, *, trigger_event_id: str, agent_name: str) -> AgentTrace:
        for trace in reversed(self._traces):
            if (
                trace.trigger_event_id == trigger_event_id
                and trace.agent_name == agent_name
            ):
                return trace
        raise KeyError(
            f"no trace for trigger_event_id={trigger_event_id} agent_name={agent_name}"
        )

    @staticmethod
    def _append_json(path: Path, payload: str) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")


__all__ = ["EpisodeJournal"]
