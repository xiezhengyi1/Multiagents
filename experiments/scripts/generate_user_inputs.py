from __future__ import annotations

from collections.abc import Iterable

from control_runtime.simulation import EnvironmentSnapshot
from training.schemas import TrainingInputRecord


def generate_user_inputs(
    *,
    experiment_id: str,
    snapshot: EnvironmentSnapshot,
    user_inputs: Iterable[str],
) -> list[TrainingInputRecord]:
    normalized_experiment_id = experiment_id.strip()
    if not normalized_experiment_id:
        raise ValueError("experiment_id is required")
    records = [
        TrainingInputRecord(
            record_id=f"{normalized_experiment_id}:{index:06d}",
            episode_id=f"{normalized_experiment_id}-{index:06d}",
            user_input=user_input,
            metadata={"snapshot_id": snapshot.snapshot_id},
        )
        for index, user_input in enumerate(user_inputs)
    ]
    if not records:
        raise ValueError("at least one user input is required")
    return records


__all__ = ["generate_user_inputs"]
