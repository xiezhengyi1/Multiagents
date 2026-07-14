from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from control_runtime.integrations.scenario.init_scenario import rebuild_ue_related_tables_from_graph_snapshot
from control_runtime.orchestrators.single_agent_orchestrator import SingleAgentOrchestrator
from experiments.paths import default_catalog_input_path, resolve_scenario_source_path
from experiments.scripts.common import append_jsonl, reset_simulator_state, write_json
from training.collect_workflow_trajectories import load_user_input_records
from training.common import processed_dir


QWEN_SINGLE_AGENT_MODEL = "qwen3-30b-a3b-instruct"


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


def summarize_run_results(results: List[Mapping[str, Any]], *, result_output: Path) -> Dict[str, Any]:
    total = len(results)
    completed = sum(1 for item in results if item.get("completed"))
    failed = sum(1 for item in results if not item.get("completed"))
    exceptions = sum(1 for item in results if item.get("status") == "error")
    return {
        "total_cases": total,
        "completed_case_count": completed,
        "error_case_count": failed,
        "exception_case_count": exceptions,
        "completed_case_rate": round(completed / total, 4) if total else 0.0,
        "error_case_rate": round(failed / total, 4) if total else 0.0,
        "exception_case_rate": round(exceptions / total, 4) if total else 0.0,
        "avg_elapsed_ms": _average_elapsed_ms(results),
        "artifacts": {"run_records": str(result_output)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-agent experiments in batch.")
    parser.add_argument("--user-inputs", type=Path, default=default_catalog_input_path())
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1, help="1-based record index to start from.")
    parser.add_argument("--target-success-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--snapshot-id", type=str, default="", help="Existing network graph snapshot id to read.")
    parser.add_argument("--scenario-prefix", type=str, default="single-agent-experiment")
    parser.add_argument("--scenario-tag", action="append", default=["single_agent_experiment"])
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument(
        "--qwen",
        action="store_true",
        dest="use_qwen",
        help=f"Use {QWEN_SINGLE_AGENT_MODEL} for the single control agent.",
    )
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
    if args.start_index <= 0:
        raise ValueError("--start-index must be positive")
    snapshot_id = str(args.snapshot_id or "").strip()
    if not snapshot_id:
        raise ValueError("--snapshot-id is required so Multiagents reads an existing graph snapshot")
    records = load_user_input_records(
        user_inputs_path=args.user_inputs,
        network_input_path=None,
        count=args.count,
        target_success_rate=args.target_success_rate,
        seed=args.seed,
        start_index=args.start_index,
    )
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text("", encoding="utf-8")

    results: List[Dict[str, Any]] = []
    total = len(records)
    for record in records:
        scenario_id = str(record.get("scenario_id") or f"{args.scenario_prefix}-{int(record['record_index']):05d}").strip()
        scenario_tags = [
            str(item).strip()
            for item in [*(record.get("scenario_tags") or []), *(args.scenario_tag or [])]
            if str(item).strip()
        ]
        try:
            scenario_source = resolve_scenario_source_path(scenario_id)
        except Exception as exc:
            raise RuntimeError(
                "single-agent experiment requires each record to carry a scenario_id defined in "
                "experiments/configs/scenarios.json; "
                f"failed to resolve scenario_id={scenario_id}: {exc}"
            ) from exc
        rebuild_ue_related_tables_from_graph_snapshot(snapshot_id)
        orchestrator = SingleAgentOrchestrator(
            max_rounds=args.max_rounds,
            use_local_model=False,
            rag_enabled=not args.disable_rag,
            single_model_name=QWEN_SINGLE_AGENT_MODEL if args.use_qwen else "",
        )
        preview = str(record["user_input"]).replace("\n", " ")[:120]
        print(f"[{record['record_index']}/{total}] single-agent start: {preview}", flush=True)
        run_started_at = time.perf_counter()
        try:
            run_result = orchestrator.run(
                str(record["user_input"]),
                scenario_id=scenario_id,
                scenario_tags=scenario_tags,
                snapshot_id=snapshot_id,
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
        result_record["elapsed_ms"] = round((time.perf_counter() - run_started_at) * 1000, 3)
        results.append(result_record)
        append_jsonl(args.result_output, result_record)
        reset_simulator_state()

    summary = summarize_run_results(results, result_output=args.result_output)
    write_json(args.summary_output, summary)
    print(f"Ran {summary['total_cases']} single-agent cases")
    print(f"Completed cases: {summary['completed_case_count']}")
    print(f"Error cases: {summary['error_case_count']}")
    print(f"Run records -> {args.result_output}")
    print(f"Summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
