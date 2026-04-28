from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "methods.json"
MATRIX_PATH = EXPERIMENT_ROOT / "configs" / "experiment_matrix.json"
DEFAULT_USER_INPUTS_PATH = EXPERIMENT_ROOT / "generated_user_inputs.json"
GENERATED_INPUT_DIR = EXPERIMENT_ROOT / "generated_inputs"
RAW_RUN_DIR = EXPERIMENT_ROOT / "results" / "raw_runs"
SUMMARY_DIR = EXPERIMENT_ROOT / "results" / "summaries"
LEDGER_PATH = EXPERIMENT_ROOT / "results" / "ledgers" / "run_ledger.csv"


def _load_methods() -> Dict[str, Dict[str, Any]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    methods = payload.get("methods", [])
    return {str(item["id"]).strip(): item for item in methods}


def _load_matrix() -> Dict[str, Dict[str, Any]]:
    payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    items = payload.get("experiments", [])
    return {str(item["id"]).strip(): item for item in items}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _append_ledger(row: Dict[str, Any]) -> None:
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "experiment_id",
                "method_id",
                "scenario_id",
                "task_id",
                "run_index",
                "status",
                "completed",
                "isr",
                "ga",
                "sv",
                "esr",
                "crr",
                "ssi",
                "round_count",
                "retry_count",
                "avg_latency_ms",
                "execution_status",
                "status_code",
                "failure_type",
                "snapshot_before",
                "snapshot_after",
                "result_path",
                "summary_path",
                "notes",
            ],
        )
        writer.writerow(row)


def _build_user_inputs_path(*, experiment_id: str, scenario_id: str) -> Path:
    parts = ["user_inputs"]
    if experiment_id:
        parts.append(experiment_id)
    if scenario_id:
        parts.append(scenario_id)
    candidate = GENERATED_INPUT_DIR / ("_".join(parts) + ".json")
    return candidate if candidate.exists() else DEFAULT_USER_INPUTS_PATH


def _resolve_case_count(user_inputs_path: Path) -> int:
    payload = json.loads(user_inputs_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No records found in {user_inputs_path}")
    return len(records)


def _run_ours(*, experiment_id: str, scenario_id: str) -> tuple[Path, Path]:
    user_inputs_path = _build_user_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not user_inputs_path.exists():
        raise FileNotFoundError(
            f"Missing {user_inputs_path}. Run experiment/scripts/build_user_inputs.py first."
        )
    case_count = _resolve_case_count(user_inputs_path)

    stamp = _timestamp()
    result_output = RAW_RUN_DIR / f"ours_runs_{stamp}.jsonl"
    workflow_output = RAW_RUN_DIR / f"ours_trajectories_{stamp}.jsonl"
    spec_output = RAW_RUN_DIR / f"ours_specs_{stamp}.jsonl"
    summary_output = SUMMARY_DIR / f"ours_summary_{stamp}.json"

    command = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(PROJECT_ROOT / "run_workflow_experiment.py"),
        "--user-inputs",
        str(user_inputs_path),
        "--count",
        str(case_count),
        "--max-rounds",
        "3",
        "--scenario-prefix",
        "experiment-ours",
        "--scenario-tag",
        "experiment",
        "--scenario-tag",
        "Ours",
        "--scenario-tag",
        experiment_id or "unscoped",
        "--scenario-tag",
        scenario_id or "mixed",
        "--result-output",
        str(result_output),
        "--workflow-output",
        str(workflow_output),
        "--spec-output",
        str(spec_output),
        "--summary-output",
        str(summary_output),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return result_output, summary_output


def _run_single_agent(*, method_id: str, experiment_id: str, scenario_id: str) -> tuple[Path, Path]:
    user_inputs_path = _build_user_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not user_inputs_path.exists():
        raise FileNotFoundError(
            f"Missing {user_inputs_path}. Run experiment/scripts/build_user_inputs.py first."
        )
    case_count = _resolve_case_count(user_inputs_path)
    stamp = _timestamp()
    result_output = RAW_RUN_DIR / f"{method_id.lower()}_runs_{stamp}.jsonl"
    summary_output = SUMMARY_DIR / f"{method_id.lower()}_summary_{stamp}.json"
    command = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(PROJECT_ROOT / "run_single_agent_experiment.py"),
        "--user-inputs",
        str(user_inputs_path),
        "--count",
        str(case_count),
        "--max-rounds",
        "3" if method_id == "B3" else "1",
        "--scenario-prefix",
        f"experiment-{method_id.lower()}",
        "--scenario-tag",
        "experiment",
        "--scenario-tag",
        method_id,
        "--scenario-tag",
        experiment_id or "unscoped",
        "--scenario-tag",
        scenario_id or "mixed",
        "--result-output",
        str(result_output),
        "--summary-output",
        str(summary_output),
    ]
    if method_id == "B1":
        command.append("--disable-rag")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return result_output, summary_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one method in the experiment matrix.")
    parser.add_argument("--method", required=True, help="Method id: B1 / B2 / B3 / Ours")
    parser.add_argument("--experiment", default="", help="Experiment id, e.g. E1")
    parser.add_argument("--scenario", default="", help="Scenario id, e.g. S2")
    args = parser.parse_args()

    methods = _load_methods()
    experiments = _load_matrix()
    method_id = str(args.method).strip()
    if method_id not in methods:
        raise ValueError(f"Unknown method id: {method_id}")
    experiment_id = str(args.experiment or "").strip()
    scenario_id = str(args.scenario or "").strip()
    if experiment_id and experiment_id not in experiments:
        raise ValueError(f"Unknown experiment id: {experiment_id}")
    if experiment_id:
        allowed_methods = {
            str(item).strip()
            for item in (experiments[experiment_id].get("methods") or [])
            if str(item).strip()
        }
        if method_id not in allowed_methods:
            raise ValueError(f"Method {method_id} is not part of experiment {experiment_id}")

    method = methods[method_id]
    status = str(method.get("implementation_status") or "").strip().lower()
    if status != "ready":
        raise NotImplementedError(
            f"Method {method_id} is not runnable in the current repository. "
            f"Reason: {method.get('notes', '')}"
        )

    if method_id == "Ours":
        result_path, summary_path = _run_ours(experiment_id=experiment_id, scenario_id=scenario_id)
    else:
        result_path, summary_path = _run_single_agent(
            method_id=method_id,
            experiment_id=experiment_id,
            scenario_id=scenario_id,
        )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = f"{method_id}-{_timestamp()}"
    _append_ledger(
        {
            "run_id": run_id,
            "experiment_id": experiment_id or "unspecified",
            "method_id": method_id,
            "scenario_id": scenario_id or "mixed",
            "task_id": "batch",
            "run_index": 1,
            "status": "success",
            "completed": summary_payload.get("correct_case_count", summary_payload.get("completed_case_count", "")),
            "isr": "",
            "ga": "",
            "sv": "",
            "esr": summary_payload.get("correct_case_rate", summary_payload.get("completed_case_rate", "")),
            "crr": "",
            "ssi": "",
            "round_count": "",
            "retry_count": "",
            "avg_latency_ms": "",
            "execution_status": "batch_completed",
            "status_code": "",
            "failure_type": "",
            "snapshot_before": "",
            "snapshot_after": "",
            "result_path": str(result_path),
            "summary_path": str(summary_path),
            "notes": "Run aggregate_results.py for per-task ledger rows.",
        }
    )
    print(json.dumps({"method": method_id, "result_path": str(result_path), "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
