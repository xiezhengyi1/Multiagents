from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, JsonValue

DatasetPartition = Literal[
    "multi_agent_control",
    "single_agent_control",
    "environment",
]


def partition_directory(dataset_root: Path, partition: DatasetPartition) -> Path:
    return Path(dataset_root) / partition


def episode_directories(
    dataset_root: Path,
    partition: DatasetPartition,
) -> list[Path]:
    root = partition_directory(dataset_root, partition)
    return sorted(path for path in root.iterdir() if path.is_dir())


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, JsonValue]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "DatasetPartition",
    "episode_directories",
    "partition_directory",
    "read_jsonl",
    "write_jsonl",
]
