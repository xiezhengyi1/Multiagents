from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"
TASK_CATALOG_PATH = EXPERIMENT_ROOT / "tasks" / "task_catalog.json"
MATRIX_PATH = EXPERIMENT_ROOT / "configs" / "experiment_matrix.json"
OUTPUT_DIR = EXPERIMENT_ROOT / "generated_inputs"


def _load_catalog(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_id_list(values: List[str]) -> List[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _resolve_allowed_scenarios(experiment_id: str) -> List[str]:
    if not experiment_id:
        return []
    payload = _load_matrix(MATRIX_PATH)
    for item in payload.get("experiments", []):
        if str(item.get("id") or "").strip() == experiment_id:
            return _normalize_id_list(list(item.get("scenarios") or []))
    raise ValueError(f"Unknown experiment id: {experiment_id}")


def _build_record(index: int, task: Dict[str, Any]) -> Dict[str, Any]:
    user_input = str(task["user_input"]).strip()
    scenario_ids = [str(item).strip() for item in task.get("scenario_ids", []) if str(item).strip()]
    category = str(task.get("category") or "").strip()
    task_id = str(task.get("task_id") or f"T{index:03d}").strip()
    return {
        "record_index": index,
        "user_input": user_input,
        "messages": [{"role": "user", "content": user_input}],
        "context": "",
        "scenario_id": task_id,
        "scenario_tags": ["experiment", category, *scenario_ids],
        "task_metadata": {
            "task_id": task_id,
            "category": category,
            "expected_objects": task.get("expected_objects", {}),
            "expected_direction": task.get("expected_direction", ""),
            "success_criteria": task.get("success_criteria", ""),
            "scenario_ids": scenario_ids,
        },
    }


def _filter_tasks(
    tasks: List[Dict[str, Any]],
    *,
    scenario_id: str,
    experiment_id: str,
) -> List[Dict[str, Any]]:
    allowed_scenarios = set(_resolve_allowed_scenarios(experiment_id)) if experiment_id else set()
    explicit_scenario = str(scenario_id or "").strip()
    filtered: List[Dict[str, Any]] = []
    for task in tasks:
        task_scenarios = {
            str(item).strip()
            for item in (task.get("scenario_ids") or [])
            if str(item).strip()
        }
        if explicit_scenario and explicit_scenario not in task_scenarios:
            continue
        if allowed_scenarios and not (task_scenarios & allowed_scenarios):
            continue
        filtered.append(task)
    return filtered


def _build_output_path(*, experiment_id: str, scenario_id: str) -> Path:
    parts = ["user_inputs"]
    if experiment_id:
        parts.append(experiment_id)
    if scenario_id:
        parts.append(scenario_id)
    filename = "_".join(parts) + ".json"
    return OUTPUT_DIR / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment user inputs from the task catalog.")
    parser.add_argument("--scenario", default="", help="Filter tasks by scenario id, e.g. S2")
    parser.add_argument("--experiment", default="", help="Filter tasks by experiment id, e.g. E1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = _load_catalog(TASK_CATALOG_PATH)
    tasks: List[Dict[str, Any]] = list(catalog.get("tasks", []))
    if not tasks:
        raise RuntimeError(f"No tasks found in {TASK_CATALOG_PATH}")

    filtered_tasks = _filter_tasks(
        tasks,
        scenario_id=str(args.scenario or "").strip(),
        experiment_id=str(args.experiment or "").strip(),
    )
    if not filtered_tasks:
        raise RuntimeError("No tasks matched the requested experiment/scenario filters.")

    records = [_build_record(index, task) for index, task in enumerate(filtered_tasks, start=1)]
    payload = {
        "meta": {
            "count": len(records),
            "source": str(TASK_CATALOG_PATH),
            "experiment_id": str(args.experiment or "").strip(),
            "scenario_id": str(args.scenario or "").strip(),
        },
        "records": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _build_output_path(
        experiment_id=str(args.experiment or "").strip(),
        scenario_id=str(args.scenario or "").strip(),
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    default_output = EXPERIMENT_ROOT / "generated_user_inputs.json"
    default_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} experiment records -> {output_path}")
    print(f"Updated default input file -> {default_output}")


if __name__ == "__main__":
    main()
