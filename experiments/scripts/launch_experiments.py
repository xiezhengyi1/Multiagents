from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import subprocess
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
    parse_task_ids,
    read_jsonl,
    resolve_python_executable,
    write_json,
)
from experiments.paths import scoped_catalog_input_path
from experiments.scripts.simulator import (
    allocate_simulator_instance as _allocate_simulator_instance,
    ensure_simulator_running as _ensure_simulator_running,
    restart_simulator_for_current_scenario as _restart_simulator_for_current_scenario,
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
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SIMULATORS_PER_COMBINATION = 3
MAX_WRITER_RESET_FULL_RESTARTS_PER_WORKER = 1
DEFAULT_WORKFLOW_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_SINGLE_AGENT_MODEL = "deepseek-v4-flash"


def _append_failed_run(record: Dict[str, Any]) -> None:
    append_jsonl(FAILED_RUNS_PATH, record)


def _is_exhausted_reset_writer_failure(exc: BaseException) -> bool:
    """Recognize only the explicit fast-reset writer terminal condition.

    `run_method.py` runs in a child process, so the typed RuntimeError becomes
    `CalledProcessError`.  Keep the recovery gate deliberately narrow: a full
    simulator restart must never mask model, policy, or ordinary reset errors.
    """
    output = str(getattr(exc, "output", "") or "")
    text = f"{exc}\n{output}".lower()
    return (
        "simulator reset writer recovery exhausted" in text
        and "a full simulator restart is required" in text
        and "writer" in text
    )

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


def _load_model_name_from_env() -> str:
    env_value = str(os.getenv("MODEL_NAME") or "").strip()
    if env_value:
        return env_value
    if not ENV_PATH.exists():
        return ""
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "MODEL_NAME":
            continue
        return value.strip().strip('"').strip("'")
    return ""


def _filename_slug(value: str, *, default: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(value or "").strip()
    ).strip("-")
    return normalized or default


def _resolve_model_name_for_run(
    *,
    args: argparse.Namespace,
    method_id: str,
    model_flags: Dict[str, bool],
) -> str:
    """Return the model identity used in artifact names, not a display label."""

    if bool(getattr(args, "use_local_model", False)):
        return str(os.getenv("vLLM_MODEL_NAME") or "local-vllm").strip() or "local-vllm"
    configured = _load_model_name_from_env()
    if configured:
        return configured
    if bool(model_flags.get("use_deepseek")):
        return DEFAULT_SINGLE_AGENT_MODEL
    if bool(model_flags.get("use_qwen")):
        return DEFAULT_WORKFLOW_MODEL
    is_single_agent = str(method_id or "").strip() in {"B1", "B2", "B3"}
    return DEFAULT_SINGLE_AGENT_MODEL if is_single_agent else DEFAULT_WORKFLOW_MODEL


def _resolve_model_flags(args: argparse.Namespace, *, method_id: str) -> Dict[str, bool]:
    model_name = _load_model_name_from_env().lower()
    if model_name:
        is_single_agent = str(method_id or "").strip() in {"B1", "B2", "B3"}
        return {
            "use_qwen": is_single_agent and "qwen" in model_name,
            "use_deepseek": (not is_single_agent) and "deepseek" in model_name,
        }

    explicit_qwen = bool(getattr(args, "use_qwen", False))
    explicit_deepseek = bool(getattr(args, "use_deepseek", False))
    if explicit_qwen or explicit_deepseek:
        return {"use_qwen": explicit_qwen, "use_deepseek": explicit_deepseek}

    return {"use_qwen": False, "use_deepseek": False}


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


def _load_generated_task_count(
    *,
    experiment_id: str,
    scenario_id: str,
    start_index: int = 1,
    task_ids: List[str] | None = None,
) -> int:
    path = _generated_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not path.exists():
        raise FileNotFoundError(f"Generated input file not found: {path}")
    payload = load_json(path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError(f"Generated input records must be a list: {path}")
    selected_task_ids = parse_task_ids(task_ids)
    if selected_task_ids:
        if start_index != 1:
            raise ValueError("--task-ids cannot be combined with --start-index")
        available_ids = {
            str(
                ((record.get("task_metadata") or {}).get("task_id") if isinstance(record, dict) else "")
                or (record.get("task_id") if isinstance(record, dict) else "")
                or ""
            ).strip()
            for record in records
            if isinstance(record, dict)
        }
        missing_ids = [task_id for task_id in selected_task_ids if task_id not in available_ids]
        if missing_ids:
            raise ValueError(
                "--task-ids contains task ids absent from "
                f"{path}: {', '.join(missing_ids)}"
            )
        return len(selected_task_ids)
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


def _partition_case_ranges(*, start_index: int, case_count: int, worker_count: int) -> List[Dict[str, int]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    base, remainder = divmod(case_count, worker_count)
    next_index = start_index
    partitions: List[Dict[str, int]] = []
    for worker_index in range(1, worker_count + 1):
        count = base + (1 if worker_index <= remainder else 0)
        partitions.append({"worker_index": worker_index, "start_index": next_index, "count": count})
        next_index += count
    return partitions


def _partition_task_ids(*, task_ids: List[str], worker_count: int) -> List[Dict[str, Any]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    base, remainder = divmod(len(task_ids), worker_count)
    next_offset = 0
    partitions: List[Dict[str, Any]] = []
    for worker_index in range(1, worker_count + 1):
        count = base + (1 if worker_index <= remainder else 0)
        assigned_ids = list(task_ids[next_offset : next_offset + count])
        partitions.append(
            {
                "worker_index": worker_index,
                "start_index": 1,
                "count": count,
                "task_ids": assigned_ids,
            }
        )
        next_offset += count
    return partitions


def _worker_environment(simulator_state: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    gateway_port = int(simulator_state["gateway_port"])
    reset_port = int(simulator_state["reset_port"])
    env["PCF_BASE_URL"] = f"http://127.0.0.1:{gateway_port}"
    env["POLICY_GATEWAY_INSTANCE_ID"] = str(simulator_state.get("gateway_instance_id") or "")
    env["SIMULATOR_RESET_URL"] = f"http://127.0.0.1:{reset_port}/v1/reset"
    instance = simulator_state.get("instance")
    if instance is not None:
        env["EXPERIMENT_WORKER_SCHEMA"] = str(instance.database_schema)
    database_url = str(simulator_state.get("runtime_database_url") or "").strip()
    if database_url:
        env["DATABASE_URL"] = database_url
    return env


def _merge_worker_results(
    *,
    experiment_id: str,
    scenario_id: str,
    method_id: str,
    model_name: str,
    worker_runs: List[Dict[str, Any]],
) -> Dict[str, str]:
    if not worker_runs:
        raise RuntimeError("Cannot merge an empty worker result set")
    first_result_path = Path(worker_runs[0]["result_path"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    method_slug = _filename_slug(method_id, default="method")
    model_slug = _filename_slug(model_name, default="default-model")
    slug = "-".join(_filename_slug(value, default="unspecified") for value in (experiment_id, scenario_id, method_id))
    merge_suffix = f"method-{method_slug}_model-{model_slug}"
    result_path = first_result_path.parent / f"{method_slug}_runs_{slug}_{stamp}_merged_{merge_suffix}.jsonl"
    summary_path = first_result_path.parent.parent / "summaries" / f"{method_slug}_summary_{slug}_{stamp}_merged_{merge_suffix}.json"

    result_rows: List[Dict[str, Any]] = []
    summary_payloads: List[Dict[str, Any]] = []
    for worker_run in worker_runs:
        result_rows.extend(read_jsonl(Path(worker_run["result_path"])))
        summary_payloads.append(load_json(Path(worker_run["summary_path"])))
    result_rows.sort(key=lambda item: int(item.get("record_index") or 0))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in result_rows),
        encoding="utf-8",
    )

    count_keys = {
        key
        for payload in summary_payloads
        for key, value in payload.items()
        if key.endswith("_count") and isinstance(value, int)
    }
    merged_summary: Dict[str, Any] = {
        key: sum(int(payload.get(key) or 0) for payload in summary_payloads)
        for key in count_keys
    }
    total_cases = int(merged_summary.get("total_cases") or len(result_rows))
    merged_summary["total_cases"] = total_cases
    for numerator_key, rate_key in (
        ("correct_case_count", "correct_case_rate"),
        ("completed_case_count", "completed_case_rate"),
        ("error_case_count", "error_case_rate"),
        ("exception_case_count", "exception_case_rate"),
        ("successful_invocation_count", "successful_invocation_rate"),
    ):
        if numerator_key in merged_summary:
            merged_summary[rate_key] = round(int(merged_summary[numerator_key]) / total_cases, 4) if total_cases else 0.0
    for elapsed_key in ("avg_elapsed_ms", "avg_wall_elapsed_ms", "avg_round_count", "avg_retry_count"):
        elapsed_weight = sum(
            float(payload.get(elapsed_key) or 0.0) * int(payload.get("total_cases") or 0)
            for payload in summary_payloads
        )
        merged_summary[elapsed_key] = round(elapsed_weight / total_cases, 4) if total_cases else 0.0
    for histogram_key in ("round_count_histogram", "retry_count_histogram"):
        merged_histogram: Dict[int, int] = {}
        for payload in summary_payloads:
            histogram = payload.get(histogram_key) or {}
            if not isinstance(histogram, dict):
                continue
            for key, value in histogram.items():
                try:
                    numeric_key = int(key)
                    numeric_value = int(value)
                except (TypeError, ValueError):
                    continue
                merged_histogram[numeric_key] = merged_histogram.get(numeric_key, 0) + numeric_value
        merged_summary[histogram_key] = dict(sorted(merged_histogram.items()))
    for key in ("correct_cases", "error_cases", "main_judgement_wrong_cases"):
        values = [item for payload in summary_payloads for item in (payload.get(key) or []) if isinstance(item, dict)]
        if values:
            merged_summary[key] = values
    merged_summary["artifacts"] = {
        "run_records": str(result_path),
        "method_id": method_id,
        "model_name": model_name,
        "worker_results": [
            {
                "worker_index": worker_run["worker_index"],
                "result_path": worker_run["result_path"],
                "summary_path": worker_run["summary_path"],
                "snapshot_id": worker_run["snapshot_id"],
            }
            for worker_run in worker_runs
        ],
    }
    write_json(summary_path, merged_summary)
    return {"result_path": str(result_path), "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch experiment runs strictly according to experiment_matrix.json.",
    )
    parser.add_argument("--experiment", default="", help="Only launch one experiment id, e.g. E1")
    parser.add_argument("--scenario", default="", help="Only launch one scenario id, e.g. S2")
    parser.add_argument("--method", default="", help="Only launch one method id, e.g. Ours")
    parser.add_argument("--start-index", type=int, default=1, help="1-based user_input record index to start from.")
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated stable task ids to run. If omitted, run the default start-index range.",
    )
    parser.add_argument(
        "--simulators-per-combination",
        type=int,
        default=DEFAULT_SIMULATORS_PER_COMBINATION,
        help="Independent simulator workers to start for every experiment/scenario/method combination (default: 3).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run build_user_inputs.py and run_method.py only, without ledger aggregation.",
    )
    parser.add_argument(
        "--local-model",
        action="store_true",
        dest="use_local_model",
        help="Use the OpenAI-compatible local vLLM endpoint configured by vLLM_URL and vLLM_MODEL_NAME.",
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
    task_ids = parse_task_ids(args.task_ids)
    if task_ids and start_index != 1:
        raise ValueError("--task-ids cannot be combined with --start-index")
    if int(args.simulators_per_combination) <= 0:
        raise ValueError("--simulators-per-combination must be positive")
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
                    task_ids=task_ids,
                )
                experiment_summaries[experiment_id]["built_scenarios"][scenario_id] = {
                    "task_count": task_count,
                    "generated_input_path": str(
                        _generated_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
                    ),
                    "task_ids": list(task_ids),
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

            for method_index, method_id in enumerate(methods, start=1):
                task_count = experiment_summaries[experiment_id]["built_scenarios"][scenario_id]["task_count"]
                simulator_states: List[Dict[str, Any]] = []
                worker_runs: List[Dict[str, Any]] = []
                worker_failures: List[Dict[str, Any]] = []
                worker_specs: List[Dict[str, Any]] = []
                try:
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
                        stage="start-worker-simulators",
                        task_count=task_count,
                    )
                    if method_id not in methods_registry:
                        raise ValueError(f"Method {method_id} referenced by {experiment_id} is not defined in methods.json")
                    method_meta = methods_registry[method_id]
                    if str(method_meta.get("implementation_status") or "").strip().lower() != "ready":
                        raise RuntimeError(f"Method {method_id} is not runnable. Reason: {method_meta.get('notes', '')}")

                    partitions = (
                        _partition_task_ids(
                            task_ids=task_ids,
                            worker_count=int(args.simulators_per_combination),
                        )
                        if task_ids
                        else _partition_case_ranges(
                            start_index=start_index,
                            case_count=task_count,
                            worker_count=int(args.simulators_per_combination),
                        )
                    )
                    for partition in partitions:
                        worker_index = partition["worker_index"]
                        worker_label = f"{experiment_id}-{scenario_id}-{method_id}-worker-{worker_index}"
                        allocation = _allocate_simulator_instance(label=worker_label)
                        try:
                            simulator_state = _start_simulator_for_scenario(
                                experiment_id,
                                scenario_id,
                                instance=allocation,
                            )
                        except Exception:
                            allocation.release()
                            raise
                        simulator_states.append(simulator_state)
                        print(
                            "[launch] worker allocation "
                            f"logical_worker={worker_index} physical_slot={allocation.slot} "
                            f"schema={allocation.database_schema}",
                            flush=True,
                        )
                        worker_specs.append({**partition, "label": worker_label, "simulator_state": simulator_state})

                    model_flags = _resolve_model_flags(args, method_id=method_id)
                    model_name = _resolve_model_name_for_run(
                        args=args,
                        method_id=method_id,
                        model_flags=model_flags,
                    )

                    def run_worker(worker: Dict[str, Any]) -> Dict[str, Any]:
                        if int(worker["count"]) <= 0:
                            return {"worker_index": worker["worker_index"], "skipped": True}
                        worker_task_ids = list(worker.get("task_ids") or [])
                        case_scope = (
                            ",".join(worker_task_ids)
                            if worker_task_ids
                            else f"{worker['start_index']}+{worker['count']}"
                        )
                        for run_attempt in range(MAX_WRITER_RESET_FULL_RESTARTS_PER_WORKER + 1):
                            simulator_state = _ensure_simulator_running(worker["simulator_state"])
                            worker["simulator_state"] = simulator_state
                            run_label = str(worker["label"])
                            if run_attempt:
                                run_label = f"{run_label}-writer-recovery-{run_attempt}"
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
                                "--count",
                                str(worker["count"]),
                                "--run-label",
                                run_label,
                                "--skip-ledger",
                            ]
                            if worker_task_ids:
                                run_method_cmd.extend(["--task-ids", ",".join(worker_task_ids)])
                            else:
                                run_method_cmd.extend(["--start-index", str(worker["start_index"])])
                            if args.use_local_model:
                                run_method_cmd.append("--local-model")
                            if model_flags["use_deepseek"]:
                                run_method_cmd.append("--deepseek")
                            if model_flags["use_qwen"]:
                                run_method_cmd.append("--qwen")
                            try:
                                stdout = _run_command(
                                    run_method_cmd,
                                    label=(
                                        f"run {experiment_id}/{scenario_id}/{method_id} "
                                        f"logical_worker={worker['worker_index']} "
                                        f"physical_slot={simulator_state['instance'].slot} "
                                        f"cases={case_scope}"
                                    ),
                                    stream_output=True,
                                    env=_worker_environment(simulator_state),
                                )
                            except subprocess.CalledProcessError as exc:
                                if (
                                    run_attempt >= MAX_WRITER_RESET_FULL_RESTARTS_PER_WORKER
                                    or not _is_exhausted_reset_writer_failure(exc)
                                ):
                                    raise
                                print(
                                    "[launch] reset writer failed permanently; restarting the full "
                                    f"simulator once for logical_worker={worker['worker_index']}",
                                    flush=True,
                                )
                                worker["simulator_state"] = _restart_simulator_for_current_scenario(
                                    simulator_state,
                                    reason="fast-reset writer recovery exhausted",
                                )
                                continue
                            payload = _parse_run_method_output(stdout)
                            return {
                                "worker_index": worker["worker_index"],
                                "snapshot_id": simulator_state["live_graph_snapshot_id"],
                                "result_path": str(payload["result_path"]),
                                "summary_path": str(payload["summary_path"]),
                            }
                        raise RuntimeError("unreachable writer-recovery loop exit")

                    with ThreadPoolExecutor(max_workers=len(worker_specs)) as executor:
                        future_map = {executor.submit(run_worker, worker): worker for worker in worker_specs}
                        for future in as_completed(future_map):
                            worker = future_map[future]
                            try:
                                worker_result = future.result()
                                if not worker_result.get("skipped"):
                                    worker_runs.append(worker_result)
                            except Exception as exc:
                                worker_failures.append(
                                    {
                                        "worker_index": worker["worker_index"],
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                    }
                                )
                    if worker_failures:
                        raise RuntimeError(f"{len(worker_failures)} parallel worker(s) failed: {worker_failures}")
                    merged = _merge_worker_results(
                        experiment_id=experiment_id,
                        scenario_id=scenario_id,
                        method_id=method_id,
                        model_name=model_name,
                        worker_runs=worker_runs,
                    )
                    run_record = {
                        "experiment_id": experiment_id,
                        "scenario_id": scenario_id,
                        "method_id": method_id,
                        "task_count": task_count,
                        "worker_count": len(worker_specs),
                        "result_path": merged["result_path"],
                        "summary_path": merged["summary_path"],
                    }
                    launched_runs.append(run_record)
                    experiment_summaries[experiment_id]["successful_runs"].append(run_record)

                    if not args.skip_aggregate:
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
                            task_count=task_count,
                        )
                        _run_command(
                            [
                                str(PYTHON_EXE),
                                str(AGGREGATE_SCRIPT),
                                "--result-jsonl",
                                merged["result_path"],
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
                        "stage": "parallel_run_or_aggregate",
                        "experiment_id": experiment_id,
                        "scenario_id": scenario_id,
                        "method_id": method_id,
                        "worker_failures": worker_failures,
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
                finally:
                    for worker in reversed(worker_specs):
                        _stop_simulator(
                            worker["simulator_state"],
                            reason=f"{experiment_id}/{scenario_id}/{method_id} worker complete",
                        )

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
