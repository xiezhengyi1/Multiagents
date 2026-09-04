from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import DatasetPartition, episode_directories, write_jsonl
from .projection import project_rl_trajectory, project_sft_examples
from .records import EpisodeRecords


def project_dataset(
    *,
    dataset_root: Path,
    partition: DatasetPartition,
    sft_output: Path,
    rl_output: Path,
) -> None:
    records = [
        EpisodeRecords.load(directory)
        for directory in episode_directories(dataset_root, partition)
    ]
    write_jsonl(
        sft_output,
        (example for episode in records for example in project_sft_examples(episode)),
    )
    write_jsonl(rl_output, (project_rl_trajectory(episode) for episode in records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("multi_agent_control", "single_agent_control", "environment"),
        required=True,
    )
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--rl-output", type=Path, required=True)
    args = parser.parse_args()
    project_dataset(
        dataset_root=args.dataset_root,
        partition=args.partition,
        sft_output=args.sft_output,
        rl_output=args.rl_output,
    )


if __name__ == "__main__":
    main()


__all__ = ["project_dataset"]
