from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"
TASK_CATALOG_PATH = EXPERIMENT_ROOT / "tasks" / "task_catalog.json"
MATRIX_PATH = EXPERIMENT_ROOT / "configs" / "experiment_matrix.json"
SCENARIO_CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "scenarios.json"
OUTPUT_DIR = EXPERIMENT_ROOT / "generated_inputs"


def _load_catalog(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Scenario file must decode to a mapping: {path}")
    return payload


def _normalize_id_list(values: List[str]) -> List[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _load_scenario_registry() -> Dict[str, Dict[str, Any]]:
    payload = json.loads(SCENARIO_CONFIG_PATH.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise TypeError("scenarios.json scenarios must be a list")
    return {
        str(item.get("id") or "").strip(): item
        for item in scenarios
        if str(item.get("id") or "").strip()
    }


def _build_scenario_inventory(scenario_id: str) -> Dict[str, Set[str]]:
    registry = _load_scenario_registry()
    scenario_meta = registry.get(scenario_id)
    if scenario_meta is None:
        raise ValueError(f"Unknown scenario id referenced by task catalog: {scenario_id}")
    source = str(scenario_meta.get("source") or "").strip()
    if not source:
        raise ValueError(f"Scenario {scenario_id} does not define a source file")

    payload = _load_yaml((PROJECT_ROOT / source).resolve())
    apps = payload.get("apps")
    flows = payload.get("flows")
    ues = payload.get("ues")
    if not isinstance(apps, list) or not isinstance(flows, list) or not isinstance(ues, list):
        raise ValueError(f"Scenario {scenario_id} must define apps/flows/ues lists")

    return {
        "apps": {str(item.get("name") or "").strip() for item in apps if str(item.get("name") or "").strip()},
        "flows": {str(item.get("name") or "").strip() for item in flows if str(item.get("name") or "").strip()},
        "supis": {str(item.get("supi") or "").strip() for item in ues if str(item.get("supi") or "").strip()},
    }


def _collect_object_references(expected_objects: Dict[str, Any]) -> Dict[str, Set[str]]:
    refs = {"apps": set(), "flows": set(), "supis": set()}
    if not isinstance(expected_objects, dict):
        raise TypeError("expected_objects must be a mapping")

    for key, value in expected_objects.items():
        normalized_key = str(key).strip()
        values: List[str]
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            values = [value.strip()] if value.strip() else []
        else:
            continue

        if normalized_key == "supi":
            refs["supis"].update(values)
        elif normalized_key == "app":
            refs["apps"].update(values)
        elif normalized_key in {"flow", "primary_flow", "secondary_flow", "target_flow", "excluded_flow", "protected_flow"}:
            refs["flows"].update(values)
        elif normalized_key in {"flows", "priority_flows", "deprioritized_flows"}:
            refs["flows"].update(values)
        elif normalized_key == "deprioritized_app":
            refs["apps"].update(values)
    return refs


def _validate_tasks_against_scenarios(tasks: List[Dict[str, Any]]) -> None:
    inventory_cache: Dict[str, Dict[str, Set[str]]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("Each task must define task_id")
        scenario_ids = _normalize_id_list(list(task.get("scenario_ids") or []))
        if not scenario_ids:
            raise ValueError(f"Task {task_id} must declare at least one scenario_id")
        expected_objects = task.get("expected_objects", {})
        refs = _collect_object_references(expected_objects)
        for scenario_id in scenario_ids:
            if scenario_id not in inventory_cache:
                inventory_cache[scenario_id] = _build_scenario_inventory(scenario_id)
            inventory = inventory_cache[scenario_id]
            for obj_type in ("apps", "flows", "supis"):
                missing = sorted(refs[obj_type] - inventory[obj_type])
                if missing:
                    raise ValueError(
                        f"Task {task_id} references missing {obj_type[:-1]} values in {scenario_id}: {missing}"
                    )


def _resolve_allowed_scenarios(experiment_id: str) -> List[str]:
    if not experiment_id:
        return []
    payload = _load_matrix(MATRIX_PATH)
    for item in payload.get("experiments", []):
        if str(item.get("id") or "").strip() == experiment_id:
            return _normalize_id_list(list(item.get("scenarios") or []))
    raise ValueError(f"Unknown experiment id: {experiment_id}")


def _build_record(index: int, task: Dict[str, Any], *, resolved_scenario_id: str) -> Dict[str, Any]:
    user_input = str(task["user_input"]).strip()
    scenario_ids = [str(item).strip() for item in task.get("scenario_ids", []) if str(item).strip()]
    category = str(task.get("category") or "").strip()
    task_id = str(task.get("task_id") or f"T{index:03d}").strip()
    if not resolved_scenario_id:
        raise ValueError(f"Task {task_id} is missing a resolved scenario_id")
    return {
        "record_index": index,
        "user_input": user_input,
        "messages": [{"role": "user", "content": user_input}],
        "context": "",
        "scenario_id": resolved_scenario_id,
        "scenario_tags": _dedupe_preserve_order(["experiment", category, resolved_scenario_id, *scenario_ids]),
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


def _resolve_record_scenario_id(task: Dict[str, Any], *, explicit_scenario_id: str) -> str:
    task_id = str(task.get("task_id") or "").strip()
    scenario_ids = _normalize_id_list(list(task.get("scenario_ids") or []))
    if explicit_scenario_id:
        if explicit_scenario_id not in scenario_ids:
            raise ValueError(f"Task {task_id} does not belong to scenario {explicit_scenario_id}")
        return explicit_scenario_id
    if len(scenario_ids) == 1:
        return scenario_ids[0]
    raise ValueError(
        f"Task {task_id} maps to multiple scenarios {scenario_ids}. "
        "Run build_user_inputs.py with --scenario to generate unambiguous records."
    )


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
    _validate_tasks_against_scenarios(tasks)

    filtered_tasks = _filter_tasks(
        tasks,
        scenario_id=str(args.scenario or "").strip(),
        experiment_id=str(args.experiment or "").strip(),
    )
    if not filtered_tasks:
        raise RuntimeError("No tasks matched the requested experiment/scenario filters.")

    explicit_scenario_id = str(args.scenario or "").strip()
    records = [
        _build_record(
            index,
            task,
            resolved_scenario_id=_resolve_record_scenario_id(task, explicit_scenario_id=explicit_scenario_id),
        )
        for index, task in enumerate(filtered_tasks, start=1)
    ]
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
