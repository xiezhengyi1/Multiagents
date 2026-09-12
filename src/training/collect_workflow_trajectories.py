from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from shared.contracts.coordination import CoordinationEvent

from .artifacts import read_jsonl
from .records import EpisodeRecords
from .schemas import CollectionResult, TrainingInputRecord

EpisodeRunner = Callable[[TrainingInputRecord], tuple[CoordinationEvent[object], Path]]


def load_user_input_records(path: Path) -> list[TrainingInputRecord]:
    source = Path(path)
    if source.suffix != ".jsonl":
        raise ValueError("training input records must use JSONL")
    return [TrainingInputRecord.model_validate(row) for row in read_jsonl(source)]


def collect_workflow_trajectories(
    records: Iterable[TrainingInputRecord],
    *,
    run_episode: EpisodeRunner,
) -> CollectionResult:
    directories: list[str] = []
    for record in records:
        terminal, episode_directory = run_episode(record)
        if terminal.message_type != "terminal":
            raise ValueError(f"episode {record.episode_id} did not terminate")
        loaded = EpisodeRecords.load(episode_directory)
        scope = loaded.events[0].get("scope")
        if not isinstance(scope, dict) or scope.get("episode_id") != record.episode_id:
            raise ValueError("runner returned a different episode_id")
        directories.append(str(Path(episode_directory).resolve()))
    return CollectionResult(episode_directories=directories)


__all__ = [
    "EpisodeRunner",
    "collect_workflow_trajectories",
    "load_user_input_records",
]
