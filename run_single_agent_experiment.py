from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.collect_workflow_trajectories import load_user_input_records
from sft_data.common import processed_dir
from workflows.single_agent_orchestrator import SingleAgentOrchestrator


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl_incremental(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def summarize_run_results(results: List[Mapping[str, Any]], *, result_output: Path) -> Dict[str, Any]:
    total = len(results)
    completed = sum(1 for item in results if item.get("completed"))
    errors = sum(1 for item in results if item.get("status") == "error")
    return {
        "total_cases": total,
        "completed_case_count": completed,
        "error_case_count": errors,
        "completed_case_rate": round(completed / total, 4) if total else 0.0,
        "error_case_rate": round(errors / total, 4) if total else 0.0,
        "artifacts": {"run_records": str(result_output)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-agent experiments in batch.")
    parser.add_argument("--user-inputs", type=Path, default=PROJECT_ROOT / "experiment" / "generated_user_inputs.json")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--target-success-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--scenario-prefix", type=str, default="single-agent-experiment")
    parser.add_argument("--scenario-tag", action="append", default=["single_agent_experiment"])
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument(
        "--result-output",
        type=Path,
        default=processed_dir("workflow") / "single_agent_experiment_runs.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=processed_dir("workflow") / "single_agent_experiment_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_user_input_records(
        user_inputs_path=args.user_inputs,
        network_input_path=None,
        count=args.count,
        target_success_rate=args.target_success_rate,
        seed=args.seed,
    )
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text("", encoding="utf-8")

    results: List[Dict[str, Any]] = []
    total = len(records)
    for record in records:
        scenario_id = str(record.get("scenario_id") or f"{args.scenario_prefix}-{int(record['record_index']):05d}").strip()
        scenario_tags = [str(item).strip() for item in [*(record.get("scenario_tags") or []), *(args.scenario_tag or [])] if str(item).strip()]
        orchestrator = SingleAgentOrchestrator(
            max_rounds=args.max_rounds,
            use_local_model=False,
            rag_enabled=not args.disable_rag,
        )
        preview = str(record["user_input"]).replace("\n", " ")[:120]
        print(f"[{record['record_index']}/{total}] single-agent start: {preview}", flush=True)
        try:
            run_result = orchestrator.run(
                str(record["user_input"]),
                scenario_id=scenario_id,
                scenario_tags=scenario_tags,
            )
            result_record = {
                "record_index": record["record_index"],
                "scenario_id": scenario_id,
                "scenario_tags": scenario_tags,
                "user_input": record["user_input"],
                "messages": record["messages"],
                "context": record["context"],
                "status": "success",
                "error_type": None,
                "error": None,
                "session_id": run_result.session_id,
                "snapshot_id": run_result.snapshot_id,
                "completed": run_result.completed,
                "global_intent": run_result.global_intent,
                "unified_plan": run_result.unified_plan,
                "qos_feedback": run_result.qos_feedback,
                "mobility_feedback": run_result.mobility_feedback,
                "diagnosis": run_result.diagnosis,
                "round_count": run_result.round_count,
                "retry_count": run_result.retry_count,
                "round_traces": run_result.round_traces,
            }
        except Exception as exc:
            result_record = {
                "record_index": record["record_index"],
                "scenario_id": scenario_id,
                "scenario_tags": scenario_tags,
                "user_input": record["user_input"],
                "messages": record["messages"],
                "context": record["context"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "session_id": "",
                "snapshot_id": "",
                "completed": False,
                "global_intent": {},
                "unified_plan": {},
                "qos_feedback": {},
                "mobility_feedback": {},
                "diagnosis": {},
                "round_count": 0,
                "retry_count": 0,
                "round_traces": [],
            }
        results.append(result_record)
        _write_jsonl_incremental(args.result_output, result_record)

    summary = summarize_run_results(results, result_output=args.result_output)
    _write_json(args.summary_output, summary)
    print(f"Ran {summary['total_cases']} single-agent cases")
    print(f"Completed cases: {summary['completed_case_count']}")
    print(f"Error cases: {summary['error_case_count']}")
    print(f"Run records -> {args.result_output}")
    print(f"Summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
