from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml
import requests

from experiments.paths import (
    CONFIG_ROOT,
    GENERATED_INPUT_ROOT,
    LEDGER_ROOT,
    PACKAGE_ROOT as EXPERIMENT_ROOT,
    PROJECT_ROOT,
    RAW_RUN_ROOT,
    RESULTS_ROOT,
    SCENARIO_ROOT,
    SUMMARY_ROOT,
    TASK_ROOT,
    WORKSPACE_ROOT,
)


SIMULATOR_RESET_URL = "http://127.0.0.1:18081/v1/reset"
SIMULATOR_RESET_TIMEOUT_SECONDS = 30.0


def reset_simulator_state(
    *,
    url: str = SIMULATOR_RESET_URL,
    timeout_seconds: float = SIMULATOR_RESET_TIMEOUT_SECONDS,
) -> None:
    response = requests.post(url, timeout=timeout_seconds)
    if response.status_code != 200:
        body = str(response.text or "").strip()[:500]
        raise RuntimeError(
            f"Simulator reset expected HTTP 200 but received {response.status_code}"
            + (f": {body}" if body else "")
        )
    print(f"[simulator] reset complete: HTTP {response.status_code}", flush=True)



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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_project_python_env() -> Dict[str, str]:
    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT / "src")
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = pythonpath + ((os.pathsep + existing) if existing else "")
    return env


def run_project_python(script: Path, *args: str, cwd: Path = WORKSPACE_ROOT) -> None:
    subprocess.run(
        [str(resolve_python_executable(PROJECT_ROOT)), str(script), *args],
        cwd=cwd,
        env=build_project_python_env(),
        check=True,
    )


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
    "append_jsonl",
    "build_project_python_env",
    "load_json",
    "load_yaml_mapping",
    "read_jsonl",
    "reset_simulator_state",
    "resolve_python_executable",
    "run_project_python",
    "write_csv",
    "write_json",
]
