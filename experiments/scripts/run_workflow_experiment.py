from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.collect_workflow_trajectories import collect_workflow_trajectories, load_user_input_records
from training.common import processed_dir
from experiments.scripts.common import write_json


def _average_elapsed_ms(results: List[Mapping[str, Any]]) -> float:
    values: List[float] = []
    for item in results:
        value = item.get("elapsed_ms")
        if isinstance(value, (int, float)):
            values.append(float(value))
            continue
        if isinstance(value, str) and value.strip():
            try:
                values.append(float(value))
            except ValueError:
                continue
    return round(sum(values) / len(values), 4) if values else 0.0


def _normalize_dispatch_receipts(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    for feedback_name in ("qos_feedback", "mobility_feedback"):
        feedback = item.get(feedback_name)
        if not isinstance(feedback, dict):
            continue
        value = feedback.get("dispatch_receipts")
        if not isinstance(value, list):
            continue
        for receipt in value:
            if isinstance(receipt, dict):
                receipts.append(receipt)
    return receipts


def _receipt_applied(receipt: Mapping[str, Any]) -> bool:
    return (
        str(receipt.get("status") or "").strip().lower() == "success"
        and str(receipt.get("execution_status") or "").strip().upper() == "APPLIED"
        and str(receipt.get("compliance_status") or "").strip().upper() == "COMPLIANT"
    )


def _case_status_code(item: Mapping[str, Any]) -> int | None:
    if item.get("status") == "error":
        return None
    receipts = _normalize_dispatch_receipts(item)
    if not receipts:
        return None
    failed_codes = [
        int(receipt["status_code"])
        for receipt in receipts
        if isinstance(receipt.get("status_code"), int) and not _receipt_applied(receipt)
    ]
    if failed_codes:
        return failed_codes[0]
    success_codes = [
        int(receipt["status_code"])
        for receipt in receipts
        if isinstance(receipt.get("status_code"), int)
    ]
    return success_codes[0] if success_codes else None


def _case_is_correct(item: Mapping[str, Any]) -> bool:
    if item.get("status") != "success" or not item.get("completed"):
        return False
    receipts = _normalize_dispatch_receipts(item)
    return bool(receipts) and all(_receipt_applied(receipt) for receipt in receipts)


def _expected_domains_from_user_input(user_input: str) -> set[str]:
    lowered = str(user_input or "").strip().lower()
    qos_only = any(
        token in lowered
        for token in (
            "only sm policy",
            "sm policy only",
            "only qos",
            "qos only",
            "只改 sm policy",
            "只调整 sm policy",
            "只改sm policy",
            "只调整sm policy",
            "只做 qos",
            "只做qos",
        )
    )
    no_mobility = any(
        token in lowered
        for token in (
            "don't touch am policy",
            "do not touch am policy",
            "don't change am policy",
            "do not change am policy",
            "keep am policy unchanged",
            "不要动 am policy",
            "不要改 am policy",
            "不要动am policy",
            "不要改am policy",
            "不改 am policy",
            "不改am policy",
        )
    )
    mobility_only = any(
        token in lowered
        for token in (
            "only am policy",
            "am policy only",
            "only mobility",
            "mobility only",
            "只改 am policy",
            "只调整 am policy",
            "只改am policy",
            "只调整am policy",
            "只做 mobility",
            "只做mobility",
        )
    )
    no_qos = any(
        token in lowered
        for token in (
            "don't touch sm policy",
            "do not touch sm policy",
            "don't change sm policy",
            "do not change sm policy",
            "keep sm policy unchanged",
            "不要动 sm policy",
            "不要改 sm policy",
            "不要动sm policy",
            "不要改sm policy",
            "不改 sm policy",
            "不改sm policy",
        )
    )
    if qos_only or no_mobility:
        return {"qos"}
    if mobility_only or no_qos:
        return {"mobility"}
    return set()


def _main_judgement_wrong(item: Mapping[str, Any]) -> bool:
    expected = _expected_domains_from_user_input(str(item.get("user_input") or ""))
    if not expected:
        return False
    global_intent = item.get("global_intent")
    if not isinstance(global_intent, dict):
        return True
    actual = {
        str(value or "").strip().lower()
        for value in (global_intent.get("requested_domains") or [])
        if str(value or "").strip()
    }
    return actual != expected


def summarize_run_results(
    results: List[Mapping[str, Any]],
    *,
    result_output: Path,
    workflow_output: Path,
) -> Dict[str, Any]:
    error_types: Counter[str] = Counter()
    error_status_codes: Counter[str] = Counter()
    round_histogram: Counter[int] = Counter()
    retry_histogram: Counter[int] = Counter()
    correct_cases: List[Dict[str, Any]] = []
    error_cases: List[Dict[str, Any]] = []
    main_judgement_wrong_cases: List[Dict[str, Any]] = []

    for item in results:
        round_histogram[int(item.get("round_count") or 0)] += 1
        retry_histogram[int(item.get("retry_count") or 0)] += 1
        if _main_judgement_wrong(item):
            main_judgement_wrong_cases.append(
                {
                    "record_index": item.get("record_index"),
                    "task_id": item.get("task_id"),
                    "category": item.get("category"),
                    "scenario_id": item.get("scenario_id"),
                    "user_input": item.get("user_input"),
                    "global_intent": item.get("global_intent"),
                }
            )

        status_code = _case_status_code(item)
        if _case_is_correct(item):
            correct_cases.append(
                {
                    "record_index": item.get("record_index"),
                    "task_id": item.get("task_id"),
                    "category": item.get("category"),
                    "scenario_id": item.get("scenario_id"),
                    "status_code": status_code,
                    "user_input": item.get("user_input"),
                }
            )
            continue

        if item.get("status") == "error":
            error_type = str(item.get("error_type") or "unknown")
            error_types[error_type] += 1
        if status_code is not None:
            error_status_codes[str(status_code)] += 1
        error_cases.append(
            {
                "record_index": item.get("record_index"),
                "task_id": item.get("task_id"),
                "category": item.get("category"),
                "scenario_id": item.get("scenario_id"),
                "status": item.get("status"),
                "completed": item.get("completed"),
                "error_type": item.get("error_type"),
                "error": item.get("error"),
                "status_code": status_code,
                "qos_feedback": item.get("qos_feedback"),
                "mobility_feedback": item.get("mobility_feedback"),
                "user_input": item.get("user_input"),
            }
        )

    total = len(results)
    correct_count = len(correct_cases)
    error_count = total - correct_count

    return {
        "total_cases": total,
        "correct_case_count": correct_count,
        "error_case_count": error_count,
        "correct_case_rate": round(correct_count / total, 4) if total else 0.0,
        "error_case_rate": round(error_count / total, 4) if total else 0.0,
        "error_type_breakdown": dict(sorted(error_types.items())),
        "error_status_code_breakdown": dict(sorted(error_status_codes.items(), key=lambda item: item[0])),
        "main_judgement_wrong_count": len(main_judgement_wrong_cases),
        "main_judgement_wrong_cases": main_judgement_wrong_cases,
        "round_count_histogram": dict(sorted(round_histogram.items())),
        "retry_count_histogram": dict(sorted(retry_histogram.items())),
        "avg_elapsed_ms": _average_elapsed_ms(results),
        "correct_cases": correct_cases,
        "error_cases": error_cases,
        "artifacts": {
            "run_records": str(result_output),
            "workflow_trajectories": str(workflow_output),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run workflow experiments in batch and summarize correct/error cases.",
    )
    parser.add_argument(
        "--user-inputs",
        type=Path,
        default=PROJECT_ROOT / "workflow_experiment_user_inputs.json",
        help="Normalized experiment input records generated by prepare_workflow_experiment.py.",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1, help="1-based record index to start from.")
    parser.add_argument("--target-success-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--snapshot-id", type=str, default="", help="Existing network graph snapshot id to read.")
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument("--deepseek", action="store_true", dest="use_deepseek", help="Use deepseek-v4-flash for all agents.")
    parser.add_argument("--scenario-prefix", type=str, default="workflow-experiment")
    parser.add_argument(
        "--scenario-tag",
        action="append",
        default=["workflow_experiment"],
        help="Additional scenario tag. Can be repeated.",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=processed_dir("workflow") / "workflow_experiment_runs.jsonl",
    )
    parser.add_argument(
        "--workflow-output",
        type=Path,
        default=processed_dir("workflow") / "workflow_experiment_trajectories.jsonl",
    )
    parser.add_argument(
        "--spec-output",
        type=Path,
        default=processed_dir("workflow") / "workflow_experiment_specs.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=processed_dir("workflow") / "workflow_experiment_summary.json",
    )
    parser.add_argument(
        "--agent-trajectory-output-root",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.start_index <= 0:
        raise ValueError("--start-index must be positive")
    if args.max_rounds <= 0:
        raise ValueError("--max-rounds must be positive")
    if not 0.0 < args.target_success_rate < 1.0:
        raise ValueError("--target-success-rate must be between 0 and 1")

    records = load_user_input_records(
        user_inputs_path=args.user_inputs,
        network_input_path=None,
        count=args.count,
        target_success_rate=args.target_success_rate,
        seed=args.seed,
        start_index=args.start_index,
    )
    collection = collect_workflow_trajectories(
        records=records,
        result_output=args.result_output,
        spec_output=args.spec_output,
        workflow_output=args.workflow_output,
        agent_trajectory_output_root=args.agent_trajectory_output_root,
        scenario_prefix=args.scenario_prefix,
        scenario_tags=list(args.scenario_tag),
        max_rounds=args.max_rounds,
        rag_enabled=not args.disable_rag,
        reset_scenario_each_run=True,
        snapshot_id=args.snapshot_id,
        use_deepseek=args.use_deepseek,
    )
    results = collection["results"]
    summary = summarize_run_results(
        results,
        result_output=args.result_output,
        workflow_output=args.workflow_output,
    )
    projection_summary = collection["projection_summary"]
    summary["artifacts"]["normalized_specs"] = str(args.spec_output)
    summary["artifacts"]["agent_trajectory_outputs"] = projection_summary.get("agent_outputs", {})
    write_json(args.summary_output, summary)

    print(f"Ran {summary['total_cases']} experiment cases")
    print(f"Correct cases: {summary['correct_case_count']}")
    print(f"Error cases: {summary['error_case_count']}")
    print(f"Correct case rate: {summary['correct_case_rate']:.2%}")
    print(f"Main judgement wrong cases: {summary['main_judgement_wrong_count']}")
    if summary["error_status_code_breakdown"]:
        print(f"Error status codes: {summary['error_status_code_breakdown']}")
    print(f"Run records -> {args.result_output}")
    print(f"Summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
