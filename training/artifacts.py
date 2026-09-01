from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel

from shared.runtime.io.paths import current_trace_segment_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"


def processed_dir(agent_name: str) -> Path:
    return TRAINING_ROOT / agent_name / "processed"


def raw_trace_path(agent_name: str) -> Path:
    return current_trace_segment_path(agent_name)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not (text := line.strip()):
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(
    path: Path,
    records: Iterable[BaseModel | Mapping[str, Any]],
    *,
    mode: str = "w",
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else dict(record)
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count
