from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import (
    CONFIG_ROOT,
    LEDGER_ROOT,
    PROJECT_ROOT,
    RAW_RUN_ROOT,
    SUMMARY_ROOT,
    WORKSPACE_ROOT,
    build_project_python_env,
    load_json,
    parse_task_ids,
    resolve_python_executable,
)
from experiments.paths import default_catalog_input_path, resolve_scenario_source_path, scoped_catalog_input_path


CONFIG_PATH = CONFIG_ROOT / "methods.json"
MATRIX_PATH = CONFIG_ROOT / "experiment_matrix.json"
DEFAULT_USER_INPUTS_PATH = default_catalog_input_path()
RAW_RUN_DIR = RAW_RUN_ROOT
SUMMARY_DIR = SUMMARY_ROOT
LEDGER_PATH = LEDGER_ROOT / "run_ledger.csv"
PYTHON_EXE = resolve_python_executable(PROJECT_ROOT)


def _load_methods() -> Dict[str, Dict[str, Any]]:
    payload = load_json(CONFIG_PATH)
    methods = payload.get("methods", [])
    return {str(item["id"]).strip(): item for item in methods}


def _load_matrix() -> Dict[str, Dict[str, Any]]:
    payload = load_json(MATRIX_PATH)
    items = payload.get("experiments", [])
    return {str(item["id"]).strip(): item for item in items}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _summary_model_suffix(
    *,
    is_workflow_method: bool,
    use_local_model: bool = False,
    use_deepseek: bool = False,
    use_qwen: bool = False,
) -> str:
    if use_local_model:
        return "local"
    if is_workflow_method:
        return "deepseek" if use_deepseek else "qwen"
    return "qwen" if use_qwen else "deepseek"


def _summary_output_path(
    *,
    method_slug: str,
    stamp: str,
    scenario_id: str,
    model_suffix: str,
) -> Path:
    scenario_suffix = str(scenario_id or "").strip() or "mixed"
    normalized_model_suffix = str(model_suffix or "").strip() or "default"
    return SUMMARY_DIR / f"{method_slug}_summary_{stamp}_{scenario_suffix}_{normalized_model_suffix}.json"


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
    candidate = scoped_catalog_input_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if candidate.exists():
        return candidate
    if experiment_id or scenario_id:
        raise FileNotFoundError(
            f"Missing scoped experiment input file: {candidate}. "
            "Run Multiagents/experiments/scripts/build_user_inputs.py with matching --experiment/--scenario first."
        )
    return DEFAULT_USER_INPUTS_PATH


def _reset_scenario_for_run(scenario_id: str, *, snapshot_id: str) -> Path:
    normalized_scenario_id = str(scenario_id or "").strip()
    if not normalized_scenario_id:
        raise ValueError("run_method.py requires --scenario so each method run can rebuild from the current graph state")
    normalized_snapshot_id = str(snapshot_id or "").strip()
    if not normalized_snapshot_id:
        raise ValueError("run_method.py requires --snapshot-id so Multiagents reads an existing graph snapshot")
    scenario_path = resolve_scenario_source_path(normalized_scenario_id)
    command = [
        str(PYTHON_EXE),
        "-m",
        "control_runtime.integrations.scenario.init_scenario",
        "--graph-snapshot-id",
        normalized_snapshot_id,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=build_project_python_env())
    return scenario_path


def _resolve_case_count(
    user_inputs_path: Path,
    *,
    start_index: int = 1,
    count: int | None = None,
    task_ids: List[str] | None = None,
) -> int:
    payload = json.loads(user_inputs_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No records found in {user_inputs_path}")
    if start_index <= 0:
        raise ValueError("--start-index must be positive")
    selected_task_ids = parse_task_ids(task_ids)
    if selected_task_ids:
        if start_index != 1:
            raise ValueError("--task-ids cannot be combined with --start-index")
        found_ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            task_metadata = record.get("task_metadata") or {}
            task_id = str(
                (task_metadata.get("task_id") if isinstance(task_metadata, dict) else "")
                or record.get("task_id")
                or ""
            ).strip()
            if task_id:
                found_ids.add(task_id)
        missing_ids = [task_id for task_id in selected_task_ids if task_id not in found_ids]
        if missing_ids:
            raise ValueError(
                "--task-ids contains task ids absent from "
                f"{user_inputs_path}: {', '.join(missing_ids)}"
            )
        return len(selected_task_ids)
    remaining_count = len(records) - start_index + 1
    if remaining_count <= 0:
        raise RuntimeError(f"No records found in {user_inputs_path} from start_index={start_index}")
    if count is None:
        return remaining_count
    requested_count = int(count)
    if requested_count <= 0:
        raise ValueError("--count must be positive")
    if requested_count > remaining_count:
        raise ValueError(
            f"Requested {requested_count} cases from start_index={start_index}, but only {remaining_count} remain"
        )
    return requested_count


def _normalized_run_label(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in str(value or ""))
    return normalized.strip("-")


def _run_stamp(*, run_label: str = "") -> str:
    stamp = _timestamp()
    label = _normalized_run_label(run_label)
    return f"{stamp}-{label}" if label else stamp


def _run_ours(
    *,
    method_id: str,
    experiment_id: str,
    scenario_id: str,
    snapshot_id: str,
    start_index: int,
    count: int | None = None,
    task_ids: List[str] | None = None,
    run_label: str = "",
    max_rounds: int = 3,
    disable_rag: bool = False,
    use_local_model: bool = False,
    use_deepseek: bool = False,
) -> tuple[Path, Path]:
    user_inputs_path = _build_user_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not user_inputs_path.exists():
        raise FileNotFoundError(
            f"Missing {user_inputs_path}. Run Multiagents/experiments/scripts/build_user_inputs.py first."
        )
    selected_task_ids = parse_task_ids(task_ids)
    case_count = _resolve_case_count(
        user_inputs_path,
        start_index=start_index,
        count=count,
        task_ids=selected_task_ids,
    )

    stamp = _run_stamp(run_label=run_label)
    method_slug = method_id.lower()
    result_output = RAW_RUN_DIR / f"{method_slug}_runs_{stamp}.jsonl"
    workflow_output = RAW_RUN_DIR / f"{method_slug}_trajectories_{stamp}.jsonl"
    spec_output = RAW_RUN_DIR / f"{method_slug}_specs_{stamp}.jsonl"
    summary_output = _summary_output_path(
        method_slug=method_slug,
        stamp=stamp,
        scenario_id=scenario_id,
        model_suffix=_summary_model_suffix(
            is_workflow_method=True,
            use_local_model=use_local_model,
            use_deepseek=use_deepseek,
        ),
    )

    command = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "experiments" / "scripts" / "run_workflow_experiment.py"),
        "--user-inputs",
        str(user_inputs_path),
        "--count",
        str(case_count),
        "--start-index",
        str(start_index),
        "--max-rounds",
        str(max_rounds),
        "--snapshot-id",
        snapshot_id,
        "--scenario-prefix",
        f"experiment-{method_slug}",
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
        "--workflow-output",
        str(workflow_output),
        "--spec-output",
        str(spec_output),
        "--summary-output",
        str(summary_output),
        "--agent-trajectory-output-root",
        str(RAW_RUN_DIR / f"{method_slug}_agent_trajectories_{stamp}"),
    ]
    if selected_task_ids:
        command.extend(["--task-ids", ",".join(selected_task_ids)])
    if disable_rag:
        command.append("--disable-rag")
    if use_local_model:
        command.append("--local-model")
    if use_deepseek:
        command.append("--deepseek")
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)
    return result_output, summary_output


def _run_single_agent(
    *,
    method_id: str,
    experiment_id: str,
    scenario_id: str,
    snapshot_id: str,
    start_index: int,
    count: int | None = None,
    task_ids: List[str] | None = None,
    run_label: str = "",
    use_local_model: bool = False,
    use_qwen: bool = False,
) -> tuple[Path, Path]:
    user_inputs_path = _build_user_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not user_inputs_path.exists():
        raise FileNotFoundError(
            f"Missing {user_inputs_path}. Run Multiagents/experiments/scripts/build_user_inputs.py first."
        )
    selected_task_ids = parse_task_ids(task_ids)
    case_count = _resolve_case_count(
        user_inputs_path,
        start_index=start_index,
        count=count,
        task_ids=selected_task_ids,
    )
    stamp = _run_stamp(run_label=run_label)
    result_output = RAW_RUN_DIR / f"{method_id.lower()}_runs_{stamp}.jsonl"
    summary_output = _summary_output_path(
        method_slug=method_id.lower(),
        stamp=stamp,
        scenario_id=scenario_id,
        model_suffix=_summary_model_suffix(
            is_workflow_method=False,
            use_local_model=use_local_model,
            use_qwen=use_qwen,
        ),
    )
    command = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "experiments" / "scripts" / "run_single_agent_experiment.py"),
        "--user-inputs",
        str(user_inputs_path),
        "--count",
        str(case_count),
        "--start-index",
        str(start_index),
        "--max-rounds",
        "3" if method_id == "B3" else "1",
        "--snapshot-id",
        snapshot_id,
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
    if selected_task_ids:
        command.extend(["--task-ids", ",".join(selected_task_ids)])
    if method_id == "B1":
        command.append("--disable-rag")
    if use_local_model:
        command.append("--local-model")
    if use_qwen:
        command.append("--qwen")
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)
    return result_output, summary_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one method in the experiment matrix.")
    parser.add_argument(
        "--method",
        required=True,
        help="Method id: B1 / B2 / B3 / Ours / Ours_wo_RAG / Ours_wo_ClosedLoop",
    )
    parser.add_argument("--experiment", default="", help="Experiment id, e.g. E1")
    parser.add_argument("--scenario", default="", help="Scenario id, e.g. S2")
    parser.add_argument("--snapshot-id", default="", help="Existing network graph snapshot id to read, e.g. live-e1")
    parser.add_argument("--start-index", type=int, default=1, help="1-based user_input record index to start from.")
    parser.add_argument("--count", type=int, default=None, help="Number of cases to run from --start-index.")
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated stable task ids. If omitted, use --start-index and --count.",
    )
    parser.add_argument("--run-label", default="", help="Unique worker label included in output artifact names.")
    parser.add_argument("--skip-ledger", action="store_true", help="Do not append a batch row; the parent launcher will aggregate worker results.")
    parser.add_argument(
        "--local-model",
        action="store_true",
        dest="use_local_model",
        help="Use the OpenAI-compatible local vLLM endpoint configured by vLLM_URL and vLLM_MODEL_NAME.",
    )
    parser.add_argument("--deepseek", action="store_true", dest="use_deepseek", help="Use deepseek-v4-flash for all agents (Ours variants only).")
    parser.add_argument("--qwen", action="store_true", dest="use_qwen", help="Use qwen3-30b-a3b-instruct for single-agent methods.")
    args = parser.parse_args()

    methods = _load_methods()
    experiments = _load_matrix()
    method_id = str(args.method).strip()
    if method_id not in methods:
        raise ValueError(f"Unknown method id: {method_id}")
    experiment_id = str(args.experiment or "").strip()
    scenario_id = str(args.scenario or "").strip()
    start_index = int(args.start_index)
    if start_index <= 0:
        raise ValueError("--start-index must be positive")
    case_count = int(args.count) if args.count is not None else None
    if case_count is not None and case_count <= 0:
        raise ValueError("--count must be positive")
    task_ids = parse_task_ids(args.task_ids)
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

    snapshot_id = str(args.snapshot_id or "").strip()
    scenario_path = _reset_scenario_for_run(scenario_id, snapshot_id=snapshot_id)

    if method_id == "Ours":
        result_path, summary_path = _run_ours(
            method_id=method_id,
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            snapshot_id=snapshot_id,
            start_index=start_index,
            count=case_count,
            task_ids=task_ids,
            run_label=args.run_label,
            use_local_model=args.use_local_model,
            use_deepseek=args.use_deepseek,
        )
    elif method_id == "Ours_wo_RAG":
        result_path, summary_path = _run_ours(
            method_id=method_id,
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            snapshot_id=snapshot_id,
            start_index=start_index,
            count=case_count,
            task_ids=task_ids,
            run_label=args.run_label,
            max_rounds=3,
            disable_rag=True,
            use_local_model=args.use_local_model,
            use_deepseek=args.use_deepseek,
        )
    elif method_id == "Ours_wo_ClosedLoop":
        result_path, summary_path = _run_ours(
            method_id=method_id,
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            snapshot_id=snapshot_id,
            start_index=start_index,
            count=case_count,
            task_ids=task_ids,
            run_label=args.run_label,
            max_rounds=1,
            disable_rag=False,
            use_local_model=args.use_local_model,
            use_deepseek=args.use_deepseek,
        )
    else:
        result_path, summary_path = _run_single_agent(
            method_id=method_id,
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            snapshot_id=snapshot_id,
            start_index=start_index,
            count=case_count,
            task_ids=task_ids,
            run_label=args.run_label,
            use_local_model=args.use_local_model,
            use_qwen=args.use_qwen,
        )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = f"{method_id}-{_run_stamp(run_label=args.run_label)}"
    if not args.skip_ledger:
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
            "snapshot_before": str(scenario_path),
            "snapshot_after": "",
            "result_path": str(result_path),
            "summary_path": str(summary_path),
            "notes": "Scenario cache rebuilt from latest graph snapshot before method run. Run aggregate_results.py for per-task ledger rows.",
            }
        )
    print(
        json.dumps(
            {
                "method": method_id,
                "scenario": scenario_id,
                "scenario_source": str(scenario_path),
                "result_path": str(result_path),
                "summary_path": str(summary_path),
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
