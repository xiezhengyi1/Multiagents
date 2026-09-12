from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .records import EpisodeRecords
from .schemas import EvaluationSummary


def evaluate_workflow_trajectories(
    episode_directories: Iterable[Path],
) -> EvaluationSummary:
    outcomes: list[str] = []
    rewards: list[float] = []
    for directory in episode_directories:
        records = EpisodeRecords.load(directory)
        terminal = records.events[-1]["payload"]
        if not isinstance(terminal, dict):
            raise TypeError("terminal payload must be an object")
        outcome = terminal.get("outcome")
        reward = terminal.get("reward_total")
        if outcome not in {"success", "failure"}:
            raise ValueError("terminal outcome must be success or failure")
        if not isinstance(reward, int | float) or isinstance(reward, bool):
            raise TypeError("terminal reward_total must be numeric")
        outcomes.append(outcome)
        rewards.append(float(reward))
    if not outcomes:
        raise ValueError("at least one episode is required")
    success_count = outcomes.count("success")
    return EvaluationSummary(
        episode_count=len(outcomes),
        success_count=success_count,
        failure_count=len(outcomes) - success_count,
        mean_reward=sum(rewards) / len(rewards),
    )


__all__ = ["evaluate_workflow_trajectories"]
