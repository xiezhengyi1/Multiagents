from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from pydantic import BaseModel

from sft_data.common import ArtifactPair, load_artifact_pairs, load_trace_records


RecordT = TypeVar("RecordT", bound=BaseModel)
TraceT = TypeVar("TraceT", bound=BaseModel)
EpisodeT = TypeVar("EpisodeT", bound=BaseModel)
TransitionT = TypeVar("TransitionT", bound=BaseModel)


@dataclass(frozen=True)
class CanonicalBuilderConfig(Generic[RecordT]):
    project_root: Path
    request_relative: str
    response_relative: str
    build_record: Callable[[ArtifactPair], RecordT]
    synthetic_records: Optional[Callable[[], Sequence[RecordT]]] = None


@dataclass(frozen=True)
class AgenticBuilderConfig(Generic[RecordT, TraceT]):
    trace_file: Path
    trace_model: type[TraceT]
    build_record: Callable[[TraceT], Optional[RecordT]]
    reject_trace: Optional[Callable[[TraceT], Optional[Dict[str, Any]]]] = None
    is_missing_trace: Optional[Callable[[TraceT], bool]] = None


@dataclass(frozen=True)
class TrajectoryBuilderConfig(Generic[EpisodeT, TransitionT, TraceT]):
    project_root: Path
    request_relative: str
    response_relative: str
    trace_file: Path
    trace_model: type[TraceT]
    build_rows: Callable[[ArtifactPair, Sequence[TraceT]], Tuple[EpisodeT, TransitionT, Optional[Dict[str, Any]]]]


def build_canonical_dataset(config: CanonicalBuilderConfig[RecordT]) -> Tuple[List[RecordT], List[Dict[str, Any]], int]:
    pairs, rejects, artifact_total = load_artifact_pairs(
        config.project_root,
        request_relative=config.request_relative,
        response_relative=config.response_relative,
    )

    records: List[RecordT] = []
    for pair in pairs:
        try:
            records.append(config.build_record(pair))
        except Exception as exc:
            rejects.append(
                {
                    "kind": "canonical_build_failed",
                    "artifact_id": pair.response.get("artifact_id"),
                    "reason": str(exc),
                }
            )

    if config.synthetic_records is not None:
        records.extend(config.synthetic_records())

    return records, rejects, artifact_total


def build_agentic_dataset(config: AgenticBuilderConfig[RecordT, TraceT]) -> Tuple[List[RecordT], List[Dict[str, Any]], int]:
    traces = load_trace_records(config.trace_file, config.trace_model)
    records: List[RecordT] = []
    rejects: List[Dict[str, Any]] = []
    missing_trace_total = 0

    for trace in traces:
        if config.is_missing_trace is not None and config.is_missing_trace(trace):
            missing_trace_total += 1
            continue

        if config.reject_trace is not None:
            reject_row = config.reject_trace(trace)
            if reject_row is not None:
                rejects.append(reject_row)
                continue

        try:
            record = config.build_record(trace)
            if record is None:
                continue
            records.append(record)
        except Exception as exc:
            trace_id = getattr(trace, "trace_id", "")
            rejects.append({"kind": "agentic_build_failed", "trace_id": trace_id, "reason": str(exc)})

    return records, rejects, missing_trace_total


def build_trajectory_dataset(
    config: TrajectoryBuilderConfig[EpisodeT, TransitionT, TraceT],
) -> Tuple[List[EpisodeT], List[TransitionT], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    pairs, rejects, artifact_total = load_artifact_pairs(
        config.project_root,
        request_relative=config.request_relative,
        response_relative=config.response_relative,
    )
    traces = load_trace_records(config.trace_file, config.trace_model)

    episodes: List[EpisodeT] = []
    transitions: List[TransitionT] = []
    hard_failures: List[Dict[str, Any]] = []

    for pair in pairs:
        try:
            episode, transition, hard_failure = config.build_rows(pair, traces)
            episodes.append(episode)
            transitions.append(transition)
            if hard_failure is not None:
                hard_failures.append(hard_failure)
        except Exception as exc:
            rejects.append(
                {
                    "kind": "trajectory_build_failed",
                    "artifact_id": pair.response.get("artifact_id"),
                    "reason": str(exc),
                }
            )

    return episodes, transitions, hard_failures, rejects, artifact_total
