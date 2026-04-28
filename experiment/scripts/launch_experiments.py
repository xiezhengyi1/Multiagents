from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"
MATRIX_PATH = EXPERIMENT_ROOT / "configs" / "experiment_matrix.json"
METHODS_PATH = EXPERIMENT_ROOT / "configs" / "methods.json"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
BUILD_INPUTS_SCRIPT = EXPERIMENT_ROOT / "scripts" / "build_user_inputs.py"
RUN_METHOD_SCRIPT = EXPERIMENT_ROOT / "scripts" / "run_method.py"
AGGREGATE_SCRIPT = EXPERIMENT_ROOT / "scripts" / "aggregate_results.py"
FAILED_RUNS_PATH = EXPERIMENT_ROOT / "results" / "ledgers" / "failed_experiments.jsonl"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix() -> List[Dict[str, Any]]:
    payload = _load_json(MATRIX_PATH)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise TypeError("experiment_matrix.json experiments must be a list")
    return experiments


def _load_methods() -> Dict[str, Dict[str, Any]]:
    payload = _load_json(METHODS_PATH)
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise TypeError("methods.json methods must be a list")
    return {
        str(item.get("id") or "").strip(): item
        for item in methods
        if str(item.get("id") or "").strip()
    }


def _run_command(command: List[str], *, label: str) -> str:
    print(f"[launch] {label}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout, flush=True)
    stderr = completed.stderr.strip()
    if stderr:
        print(stderr, file=sys.stderr, flush=True)
    return stdout


def _append_failed_run(record: Dict[str, Any]) -> None:
    FAILED_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_RUNS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def _parse_run_method_output(stdout: str) -> Dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        raise RuntimeError("run_method.py returned empty stdout")
    lines = [line for line in text.splitlines() if line.strip()]
    for start_index in range(len(lines)):
        candidate = "\n".join(lines[start_index:])
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("result_path") and payload.get("summary_path"):
            return payload
    raise RuntimeError("Failed to parse run_method.py JSON output")


def _iter_selected_experiments(
    experiments: List[Dict[str, Any]],
    *,
    experiment_filter: str,
) -> List[Dict[str, Any]]:
    if not experiment_filter:
        return experiments
    selected = [item for item in experiments if str(item.get("id") or "").strip() == experiment_filter]
    if not selected:
        raise ValueError(f"Unknown experiment id: {experiment_filter}")
    return selected


def _filter_values(values: List[str], allowed_filter: str) -> List[str]:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if not allowed_filter:
        return normalized
    selected = [item for item in normalized if item == allowed_filter]
    if not selected:
        raise ValueError(f"Requested filter value '{allowed_filter}' not found in {normalized}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch experiment runs strictly according to experiment_matrix.json.",
    )
    parser.add_argument("--experiment", default="", help="Only launch one experiment id, e.g. E1")
    parser.add_argument("--scenario", default="", help="Only launch one scenario id, e.g. S2")
    parser.add_argument("--method", default="", help="Only launch one method id, e.g. Ours")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run build_user_inputs.py and run_method.py only, without ledger aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Missing Python executable: {PYTHON_EXE}")

    args = parse_args()
    experiments = _iter_selected_experiments(
        _load_matrix(),
        experiment_filter=str(args.experiment or "").strip(),
    )
    methods_registry = _load_methods()

    launched_runs: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    for experiment in experiments:
        experiment_id = str(experiment.get("id") or "").strip()
        scenarios = _filter_values(
            list(experiment.get("scenarios") or []),
            str(args.scenario or "").strip(),
        )
        methods = _filter_values(
            list(experiment.get("methods") or []),
            str(args.method or "").strip(),
        )
        if not scenarios:
            raise RuntimeError(f"Experiment {experiment_id} has no scenarios")
        if not methods:
            raise RuntimeError(f"Experiment {experiment_id} has no methods")

        for scenario_id in scenarios:
            try:
                _run_command(
                    [
                        str(PYTHON_EXE),
                        str(BUILD_INPUTS_SCRIPT),
                        "--experiment",
                        experiment_id,
                        "--scenario",
                        scenario_id,
                    ],
                    label=f"build inputs for {experiment_id}/{scenario_id}",
                )
            except Exception as exc:
                failure = {
                    "stage": "build_inputs",
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "method_id": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_runs.append(failure)
                _append_failed_run(failure)
                print(
                    f"[launch][skip] build inputs failed for {experiment_id}/{scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            for method_id in methods:
                try:
                    if method_id not in methods_registry:
                        raise ValueError(f"Method {method_id} referenced by {experiment_id} is not defined in methods.json")
                    method_meta = methods_registry[method_id]
                    if str(method_meta.get("implementation_status") or "").strip().lower() != "ready":
                        raise RuntimeError(
                            f"Method {method_id} is not runnable. Reason: {method_meta.get('notes', '')}"
                        )
                    stdout = _run_command(
                        [
                            str(PYTHON_EXE),
                            str(RUN_METHOD_SCRIPT),
                            "--method",
                            method_id,
                            "--experiment",
                            experiment_id,
                            "--scenario",
                            scenario_id,
                        ],
                        label=f"run {experiment_id}/{scenario_id}/{method_id}",
                    )
                    payload = _parse_run_method_output(stdout)
                    run_record = {
                        "experiment_id": experiment_id,
                        "scenario_id": scenario_id,
                        "method_id": method_id,
                        "result_path": str(payload["result_path"]),
                        "summary_path": str(payload["summary_path"]),
                    }
                    launched_runs.append(run_record)

                    if args.skip_aggregate:
                        continue

                    _run_command(
                        [
                            str(PYTHON_EXE),
                            str(AGGREGATE_SCRIPT),
                            "--result-jsonl",
                            str(payload["result_path"]),
                            "--method",
                            method_id,
                            "--experiment",
                            experiment_id,
                            "--scenario",
                            scenario_id,
                        ],
                        label=f"aggregate {experiment_id}/{scenario_id}/{method_id}",
                    )
                except Exception as exc:
                    failure = {
                        "stage": "run_or_aggregate",
                        "experiment_id": experiment_id,
                        "scenario_id": scenario_id,
                        "method_id": method_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failed_runs.append(failure)
                    _append_failed_run(failure)
                    print(
                        f"[launch][skip] failed {experiment_id}/{scenario_id}/{method_id}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

    print(
        json.dumps(
            {
                "launched_run_count": len(launched_runs),
                "launched_runs": launched_runs,
                "failed_run_count": len(failed_runs),
                "failed_runs_path": str(FAILED_RUNS_PATH),
                "aggregation_skipped": bool(args.skip_aggregate),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
