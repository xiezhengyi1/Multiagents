from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
GENERATED_INPUT_ROOT = PACKAGE_ROOT / "generated_inputs"
SCENARIO_ROOT = PACKAGE_ROOT / "scenarios"
TASK_ROOT = PACKAGE_ROOT / "tasks"
RESULTS_ROOT = PACKAGE_ROOT / "results"
LEDGER_ROOT = RESULTS_ROOT / "ledgers"
RAW_RUN_ROOT = RESULTS_ROOT / "raw_runs"
SUMMARY_ROOT = RESULTS_ROOT / "summaries"


def workflow_experiment_input_path() -> Path:
    return GENERATED_INPUT_ROOT / "workflow_experiment_user_inputs.json"


def default_catalog_input_path() -> Path:
    return GENERATED_INPUT_ROOT / "user_inputs.json"


def scoped_catalog_input_path(*, experiment_id: str = "", scenario_id: str = "") -> Path:
    parts = ["user_inputs"]
    if str(experiment_id or "").strip():
        parts.append(str(experiment_id).strip())
    if str(scenario_id or "").strip():
        parts.append(str(scenario_id).strip())
    return GENERATED_INPUT_ROOT / ("_".join(parts) + ".json")


def load_scenario_registry() -> Dict[str, Dict[str, Any]]:
    payload = json.loads((CONFIG_ROOT / "scenarios.json").read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise TypeError("experiments/configs/scenarios.json field 'scenarios' must be a list")
    return {
        str(item.get("id") or "").strip(): item
        for item in scenarios
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }


def resolve_scenario_source_path(scenario_id: str) -> Path:
    normalized_id = str(scenario_id or "").strip()
    if not normalized_id:
        raise ValueError("scenario_id is required to resolve the scenario source path")
    registry = load_scenario_registry()
    metadata = registry.get(normalized_id)
    if metadata is None:
        raise KeyError(f"unknown scenario_id: {normalized_id}")
    source = str(metadata.get("source") or "").strip()
    if not source:
        raise ValueError(f"scenario {normalized_id} does not define a source path")
    return (PACKAGE_ROOT.parent / source).resolve()


__all__ = [
    "CONFIG_ROOT",
    "GENERATED_INPUT_ROOT",
    "LEDGER_ROOT",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "RAW_RUN_ROOT",
    "RESULTS_ROOT",
    "SCENARIO_ROOT",
    "SUMMARY_ROOT",
    "TASK_ROOT",
    "WORKSPACE_ROOT",
    "default_catalog_input_path",
    "load_scenario_registry",
    "resolve_scenario_source_path",
    "scoped_catalog_input_path",
    "workflow_experiment_input_path",
]
