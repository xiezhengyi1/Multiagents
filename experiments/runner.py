from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shared.contracts.coordination import CoordinationEvent, TerminalPayload
from training import EpisodeRecords, project_episode_metrics

from .contracts import ExperimentCase, ExperimentResult


@dataclass(frozen=True, slots=True)
class PreparedEpisode:
    episode_directory: Path
    execute: Callable[[], CoordinationEvent[object]]


class ExperimentBackend(Protocol):
    """Boundary implemented by the selected simulator and model environment."""

    def prepare(self, case: ExperimentCase) -> PreparedEpisode: ...


def run_case(case: ExperimentCase, *, backend: ExperimentBackend) -> ExperimentResult:
    prepared = backend.prepare(case)
    terminal_event = prepared.execute()
    if terminal_event.message_type != "terminal":
        raise ValueError(f"experiment case {case.case_id} did not terminate")
    if not isinstance(terminal_event.payload, TerminalPayload):
        raise TypeError("terminal event payload must be TerminalPayload")
    records = EpisodeRecords.load(prepared.episode_directory)
    recorded_terminal = records.events[-1]
    if recorded_terminal["event_id"] != terminal_event.event_id:
        raise ValueError("returned terminal event does not match the episode journal")
    return ExperimentResult(
        case_id=case.case_id,
        episode_id=case.episode_id,
        method=case.method,
        episode_directory=str(prepared.episode_directory.resolve()),
        terminal=terminal_event.payload,
        metrics=project_episode_metrics(records),
    )


def run_cases(
    cases: Iterable[ExperimentCase],
    *,
    backend: ExperimentBackend,
) -> list[ExperimentResult]:
    return [run_case(case, backend=backend) for case in cases]


__all__ = [
    "ExperimentBackend",
    "PreparedEpisode",
    "run_case",
    "run_cases",
]
