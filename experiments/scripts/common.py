from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
SCENARIO_ROOT = EXPERIMENT_ROOT / "scenarios"
TASK_ROOT = EXPERIMENT_ROOT / "tasks"
GENERATED_INPUT_ROOT = EXPERIMENT_ROOT / "generated_inputs"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
LEDGER_ROOT = RESULTS_ROOT / "ledgers"
RAW_RUN_ROOT = RESULTS_ROOT / "raw_runs"
SUMMARY_ROOT = RESULTS_ROOT / "summaries"


def resolve_python_executable(project_root: Path) -> Path:
    candidates = [
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "bin" / "python3",
        project_root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


__all__ = [
    "CONFIG_ROOT",
    "EXPERIMENT_ROOT",
    "GENERATED_INPUT_ROOT",
    "LEDGER_ROOT",
    "PROJECT_ROOT",
    "RAW_RUN_ROOT",
    "RESULTS_ROOT",
    "SCENARIO_ROOT",
    "SUMMARY_ROOT",
    "TASK_ROOT",
    "WORKSPACE_ROOT",
    "load_json",
    "load_yaml_mapping",
    "resolve_python_executable",
]
