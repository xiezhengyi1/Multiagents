from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import TerminalPayload
from shared.contracts.model_trace import AgentTrace

ModelActor = Literal["main", "sba", "ppa", "single_control", "environment_generator"]
ValidationDisposition = Literal["accepted", "accepted_upstream_repair", "rejected"]

MODEL_ACTORS: frozenset[ModelActor] = frozenset(
    {"main", "sba", "ppa", "single_control", "environment_generator"}
)
VALIDATOR_ACTORS = frozenset(
    {
        "main_validator",
        "sba_validator",
        "ppa_validator",
        "single_validator",
        "environment_validator",
    }
)


class SftCandidate(ContractModel):
    trace: AgentTrace
    output_event_id: NonEmptyStr
    validation_event_id: NonEmptyStr
    disposition: Literal["accepted", "accepted_upstream_repair"]


class EpisodeRecords(ContractModel):
    """Native episode data; deliberately contains no training projection."""

    events: list[dict[str, JsonValue]] = Field(min_length=1)
    traces: list[AgentTrace]

    @classmethod
    def load(cls, episode_directory: Path) -> EpisodeRecords:
        files = sorted(
            path.name for path in episode_directory.iterdir() if path.is_file()
        )
        if files != ["events.jsonl", "traces.jsonl"]:
            raise ValueError(
                "episode directory must contain only events.jsonl and traces.jsonl"
            )
        events = [
            json.loads(line)
            for line in (episode_directory / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        traces = [
            AgentTrace.model_validate_json(line)
            for line in (episode_directory / "traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        return cls(events=events, traces=traces)

    @model_validator(mode="after")
    def validate_lineage(self) -> EpisodeRecords:
        event_ids: set[str] = set()
        for expected_index, event in enumerate(self.events):
            if event.get("event_index") != expected_index:
                raise ValueError("event_index values must be contiguous and ordered")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("every event requires event_id")
            if event_id in event_ids:
                raise ValueError("event_id values must be unique")
            references = [
                *event.get("input_artifact_ids", []),
                *event.get("invalidated_artifact_ids", []),
                *(
                    [event["in_reply_to"]]
                    if event.get("in_reply_to") is not None
                    else []
                ),
            ]
            unknown = [
                reference for reference in references if reference not in event_ids
            ]
            if unknown:
                raise ValueError(
                    f"event {event_id} references non-prior artifacts: {', '.join(unknown)}"
                )
            event_ids.add(event_id)
        terminal_events = [
            event for event in self.events if event.get("message_type") == "terminal"
        ]
        if len(terminal_events) != 1 or terminal_events[0] is not self.events[-1]:
            raise ValueError(
                "completed episode must end with exactly one terminal event"
            )
        TerminalPayload.model_validate(terminal_events[0]["payload"])

        call_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for trace in self.traces:
            if trace.trigger_event_id not in event_ids:
                raise ValueError(
                    f"trace {trace.trace_id} references unknown trigger event"
                )
            call_indices[(trace.trigger_event_id, trace.agent_name)].append(
                trace.call_index
            )
        for key, values in call_indices.items():
            if values != list(range(len(values))):
                raise ValueError(
                    f"trace call_index values are not contiguous for {key}"
                )
        return self

    def sft_candidates(self) -> list[SftCandidate]:
        traces_by_invocation: dict[tuple[str, str], list[AgentTrace]] = defaultdict(
            list
        )
        for trace in self.traces:
            traces_by_invocation[(trace.trigger_event_id, trace.agent_name)].append(
                trace
            )

        resolutions_by_output: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
        for event in self.events:
            source = event.get("source_agent")
            if source in VALIDATOR_ACTORS:
                in_reply_to = event.get("in_reply_to")
                if isinstance(in_reply_to, str):
                    resolutions_by_output[in_reply_to].append(event)

        candidates: list[SftCandidate] = []
        for event in self.events:
            actor = event.get("source_agent")
            message_type = event.get("message_type")
            trigger_event_id = event.get("in_reply_to")
            if actor not in MODEL_ACTORS:
                continue
            if message_type not in {"proposal", "repair_response"}:
                continue
            if not isinstance(trigger_event_id, str):
                raise TypeError(
                    "model-authored output event requires string in_reply_to"
                )
            successful = [
                trace
                for trace in traces_by_invocation[(trigger_event_id, actor)]
                if trace.status == "success"
            ]
            if not successful:
                raise ValueError(
                    f"model output event {event['event_id']} has no successful atomic trace"
                )
            trace = max(successful, key=lambda item: item.call_index)
            resolutions = resolutions_by_output[str(event["event_id"])]
            if len(resolutions) != 1:
                raise ValueError(
                    f"model output event {event['event_id']} must have exactly one validator resolution"
                )
            resolution = resolutions[0]
            if resolution["message_type"] == "validation_result":
                disposition: ValidationDisposition = "accepted"
            elif resolution["message_type"] == "repair_request":
                invalidated = resolution.get("invalidated_artifact_ids", [])
                disposition = (
                    "rejected"
                    if event["event_id"] in invalidated
                    else "accepted_upstream_repair"
                )
            else:
                raise ValueError(
                    "validator resolution must be validation_result or repair_request"
                )
            if disposition == "rejected":
                continue
            candidates.append(
                SftCandidate(
                    trace=trace,
                    output_event_id=str(event["event_id"]),
                    validation_event_id=str(resolution["event_id"]),
                    disposition=disposition,
                )
            )
        return candidates


__all__ = [
    "MODEL_ACTORS",
    "VALIDATOR_ACTORS",
    "EpisodeRecords",
    "ModelActor",
    "SftCandidate",
    "ValidationDisposition",
]
