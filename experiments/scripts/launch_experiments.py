from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import (
    CONFIG_ROOT,
    EXPERIMENT_ROOT,
    LEDGER_ROOT,
    PROJECT_ROOT,
    append_jsonl,
    load_json,
    resolve_python_executable,
)
from experiments.paths import scoped_catalog_input_path
from experiments.scripts.simulator import (
    ensure_simulator_running as _ensure_simulator_running,
    run_command as _run_command,
    start_simulator_for_scenario as _start_simulator_for_scenario,
    stop_simulator as _stop_simulator,
)


MATRIX_PATH = CONFIG_ROOT / "experiment_matrix.json"
METHODS_PATH = CONFIG_ROOT / "methods.json"
PYTHON_EXE = resolve_python_executable(PROJECT_ROOT)
BUILD_INPUTS_SCRIPT = EXPERIMENT_ROOT / "scripts" / "build_user_inputs.py"
RUN_METHOD_SCRIPT = EXPERIMENT_ROOT / "scripts" / "run_method.py"
AGGREGATE_SCRIPT = EXPERIMENT_ROOT / "scripts" / "aggregate_results.py"
FAILED_RUNS_PATH = LEDGER_ROOT / "failed_experiments.jsonl"


def _append_failed_run(record: Dict[str, Any]) -> None:
    append_jsonl(FAILED_RUNS_PATH, record)

def _load_matrix() -> List[Dict[str, Any]]:
    payload = load_json(MATRIX_PATH)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise TypeError("experiment_matrix.json experiments must be a list")
    return experiments


def _load_methods() -> Dict[str, Dict[str, Any]]:
    payload = load_json(METHODS_PATH)
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise TypeError("methods.json methods must be a list")
    return {
        str(item.get("id") or "").strip(): item
        for item in methods
        if str(item.get("id") or "").strip()
    }


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


def _generated_inputs_path(*, experiment_id: str, scenario_id: str) -> Path:
    return scoped_catalog_input_path(experiment_id=experiment_id, scenario_id=scenario_id)


def _load_generated_task_count(*, experiment_id: str, scenario_id: str, start_index: int = 1) -> int:
    path = _generated_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not path.exists():
        raise FileNotFoundError(f"Generated input file not found: {path}")
    payload = load_json(path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError(f"Generated input records must be a list: {path}")
    if start_index <= 0:
        raise ValueError("--start-index must be positive")
    remaining_count = len(records) - start_index + 1
    if remaining_count <= 0:
        raise RuntimeError(f"No generated input records found in {path} from start_index={start_index}")
    return remaining_count


def _init_experiment_summary(experiment_id: str, planned_scenarios: List[str], planned_methods: List[str]) -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "planned_scenario_count": len(planned_scenarios),
        "planned_method_count": len(planned_methods),
        "planned_run_count": len(planned_scenarios) * len(planned_methods),
        "built_scenarios": {},
        "successful_runs": [],
        "failed_runs": [],
    }


def _print_experiment_progress(
    *,
    experiment_id: str,
    experiment_index: int,
    experiment_total: int,
    scenario_id: str,
    scenario_index: int,
    scenario_total: int,
    method_id: str = "",
    method_index: int = 0,
    method_total: int = 0,
    stage: str,
    task_count: int | None = None,
) -> None:
    parts = [
        f"experiment={experiment_id} ({experiment_index}/{experiment_total})",
        f"scenario={scenario_id} ({scenario_index}/{scenario_total})",
        f"stage={stage}",
    ]
    if method_id:
        parts.insert(2, f"agent={method_id} ({method_index}/{method_total})")
    if task_count is not None:
        parts.append(f"tasks={task_count}")
    print(f"[launch][progress] {' | '.join(parts)}", flush=True)


def _build_coverage_summary(experiment_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    coverage: List[Dict[str, Any]] = []
    for experiment_id, summary in experiment_summaries.items():
        built_scenarios = summary["built_scenarios"]
        successful_runs = summary["successful_runs"]
        failed_runs = summary["failed_runs"]
        successful_scenarios = sorted({item["scenario_id"] for item in successful_runs})
        successful_methods = sorted({item["method_id"] for item in successful_runs})
        successful_task_invocations = sum(
            int(built_scenarios[item["scenario_id"]]["task_count"])
            for item in successful_runs
            if item["scenario_id"] in built_scenarios
        )
        coverage.append(
            {
                "experiment_id": experiment_id,
                "planned_scenario_count": summary["planned_scenario_count"],
                "planned_method_count": summary["planned_method_count"],
                "planned_run_count": summary["planned_run_count"],
                "built_scenario_count": len(built_scenarios),
                "built_scenarios": [
                    {
                        "scenario_id": scenario_id,
                        "task_count": scenario_meta["task_count"],
                        "generated_input_path": scenario_meta["generated_input_path"],
                    }
                    for scenario_id, scenario_meta in sorted(built_scenarios.items())
                ],
                "successful_run_count": len(successful_runs),
                "failed_run_count": len(failed_runs),
                "successful_scenario_count": len(successful_scenarios),
                "successful_method_count": len(successful_methods),
                "successful_scenarios": successful_scenarios,
                "successful_methods": successful_methods,
                "successful_task_invocations": successful_task_invocations,
            }
        )
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch experiment runs strictly according to experiment_matrix.json.",
    )
    parser.add_argument("--experiment", default="", help="Only launch one experiment id, e.g. E1")
    parser.add_argument("--scenario", default="", help="Only launch one scenario id, e.g. S2")
    parser.add_argument("--method", default="", help="Only launch one method id, e.g. Ours")
    parser.add_argument("--start-index", type=int, default=1, help="1-based user_input record index to start from.")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run build_user_inputs.py and run_method.py only, without ledger aggregation.",
    )
    parser.add_argument(
        "--deepseek",
        action="store_true",
        dest="use_deepseek",
        help="Use deepseek-v4-flash for all agents (Ours variants only).",
    )
    parser.add_argument(
        "--qwen",
        action="store_true",
        dest="use_qwen",
        help="Use qwen3-30b-a3b-instruct for single-agent methods.",
    )
    return parser.parse_args()


def main() -> None:
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Missing Python executable: {PYTHON_EXE}")

    args = parse_args()
    start_index = int(args.start_index)
    if start_index <= 0:
        raise ValueError("--start-index must be positive")
    experiments = _iter_selected_experiments(
        _load_matrix(),
        experiment_filter=str(args.experiment or "").strip(),
    )
    methods_registry = _load_methods()

    launched_runs: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    experiment_summaries: Dict[str, Dict[str, Any]] = {}
    experiment_total = len(experiments)
    for experiment_index, experiment in enumerate(experiments, start=1):
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
        experiment_summaries[experiment_id] = _init_experiment_summary(experiment_id, scenarios, methods)
        scenario_total = len(scenarios)
        method_total = len(methods)

        for scenario_index, scenario_id in enumerate(scenarios, start=1):
            _print_experiment_progress(
                experiment_id=experiment_id,
                experiment_index=experiment_index,
                experiment_total=experiment_total,
                scenario_id=scenario_id,
                scenario_index=scenario_index,
                scenario_total=scenario_total,
                stage="build-inputs",
            )
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
                task_count = _load_generated_task_count(
                    experiment_id=experiment_id,
                    scenario_id=scenario_id,
                    start_index=start_index,
                )
                experiment_summaries[experiment_id]["built_scenarios"][scenario_id] = {
                    "task_count": task_count,
                    "generated_input_path": str(
                        _generated_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
                    ),
                }
                _print_experiment_progress(
                    experiment_id=experiment_id,
                    experiment_index=experiment_index,
                    experiment_total=experiment_total,
                    scenario_id=scenario_id,
                    scenario_index=scenario_index,
                    scenario_total=scenario_total,
                    stage="start-simulator",
                    task_count=task_count,
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
                experiment_summaries[experiment_id]["failed_runs"].append(failure)
                _append_failed_run(failure)
                print(
                    f"[launch][skip] build inputs failed for {experiment_id}/{scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            simulator_state: Dict[str, Any] | None = None
            try:
                simulator_state = _start_simulator_for_scenario(experiment_id, scenario_id)
            except Exception as exc:
                failure = {
                    "stage": "start_simulator",
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "method_id": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_runs.append(failure)
                experiment_summaries[experiment_id]["failed_runs"].append(failure)
                _append_failed_run(failure)
                print(
                    f"[launch][skip] simulator startup failed for {experiment_id}/{scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            try:
                for method_index, method_id in enumerate(methods, start=1):
                    try:
                        simulator_state = _ensure_simulator_running(simulator_state)
                        _print_experiment_progress(
                            experiment_id=experiment_id,
                            experiment_index=experiment_index,
                            experiment_total=experiment_total,
                            scenario_id=scenario_id,
                            scenario_index=scenario_index,
                            scenario_total=scenario_total,
                            method_id=method_id,
                            method_index=method_index,
                            method_total=method_total,
                            stage="run-method",
                            task_count=experiment_summaries[experiment_id]["built_scenarios"][scenario_id]["task_count"],
                        )
                        if method_id not in methods_registry:
                            raise ValueError(f"Method {method_id} referenced by {experiment_id} is not defined in methods.json")
                        method_meta = methods_registry[method_id]
                        if str(method_meta.get("implementation_status") or "").strip().lower() != "ready":
                            raise RuntimeError(
                                f"Method {method_id} is not runnable. Reason: {method_meta.get('notes', '')}"
                            )
                        run_method_cmd = [
                                str(PYTHON_EXE),
                                str(RUN_METHOD_SCRIPT),
                                "--method",
                                method_id,
                                "--experiment",
                                experiment_id,
                                "--scenario",
                                scenario_id,
                                "--snapshot-id",
                                str(simulator_state["live_graph_snapshot_id"]),
                                "--start-index",
                                str(start_index),
                            ]
                        if args.use_deepseek:
                            run_method_cmd.append("--deepseek")
                        if args.use_qwen:
                            run_method_cmd.append("--qwen")
                        stdout = _run_command(
                            run_method_cmd,
                            label=f"run {experiment_id}/{scenario_id}/{method_id}",
                            stream_output=True,
                        )
                        payload = _parse_run_method_output(stdout)
                        run_record = {
                            "experiment_id": experiment_id,
                            "scenario_id": scenario_id,
                            "method_id": method_id,
                            "task_count": experiment_summaries[experiment_id]["built_scenarios"][scenario_id]["task_count"],
                            "result_path": str(payload["result_path"]),
                            "summary_path": str(payload["summary_path"]),
                        }
                        launched_runs.append(run_record)
                        experiment_summaries[experiment_id]["successful_runs"].append(run_record)

                        if args.skip_aggregate:
                            continue

                        _print_experiment_progress(
                            experiment_id=experiment_id,
                            experiment_index=experiment_index,
                            experiment_total=experiment_total,
                            scenario_id=scenario_id,
                            scenario_index=scenario_index,
                            scenario_total=scenario_total,
                            method_id=method_id,
                            method_index=method_index,
                            method_total=method_total,
                            stage="aggregate",
                            task_count=run_record["task_count"],
                        )
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
                        experiment_summaries[experiment_id]["failed_runs"].append(failure)
                        _append_failed_run(failure)
                        print(
                            f"[launch][skip] failed {experiment_id}/{scenario_id}/{method_id}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
            finally:
                _stop_simulator(simulator_state, reason=f"{experiment_id}/{scenario_id} complete")

    print(
        json.dumps(
            {
                "launched_run_count": len(launched_runs),
                "launched_runs": launched_runs,
                "failed_run_count": len(failed_runs),
                "failed_runs_path": str(FAILED_RUNS_PATH),
                "aggregation_skipped": bool(args.skip_aggregate),
                "coverage_summary": _build_coverage_summary(experiment_summaries),
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
