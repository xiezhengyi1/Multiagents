from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.contracts import ExperimentCase
from training.artifacts import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "experiments" / "cases" / "s2_original_30.json"
EXPECTED_TASK_IDS = (
    "T001",
    "T003",
    "T004",
    "T005",
    "T006",
    "T007",
    "T008",
    "T009",
    "T011",
    "T012",
    "T013",
    "T016",
    "T017",
    "T018",
    "T019",
    "T020",
    "T021",
    "T022",
    "T023",
    "T024",
    "T025",
    "T026",
    "T027",
    "T028",
    "T029",
    "T030",
    "T031",
    "T032",
    "T033",
    "T034",
)


def load_s2_records(path: Path = SOURCE_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    task_ids = tuple(record["task_metadata"]["task_id"] for record in records)
    if payload["meta"]["scenario_id"] != "S2" or task_ids != EXPECTED_TASK_IDS:
        raise ValueError("S2 source records do not match the original 30-case inventory")
    return records


def build_s2_cases(*, run_id: str, path: Path = SOURCE_PATH) -> list[ExperimentCase]:
    return [
        ExperimentCase(
            case_id=f"S2-{record['task_metadata']['task_id']}",
            episode_id=f"{run_id}-{record['task_metadata']['task_id'].lower()}",
            method="multi_agent_control",
            user_input=record["user_input"],
            metadata={
                "scenario_id": "S2",
                "record_index": record["record_index"],
                "original_commit": "9649f526eb47c32c26db86ae214074981eea8f3e",
                "task": record["task_metadata"],
            },
        )
        for record in load_s2_records(path)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = build_s2_cases(run_id=args.run_id)
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} S2 cases to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["EXPECTED_TASK_IDS", "SOURCE_PATH", "build_s2_cases", "load_s2_records"]
