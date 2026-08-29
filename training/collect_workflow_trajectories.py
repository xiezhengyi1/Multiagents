from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from database.init_db import init_db
from experiments.scripts.generate_user_inputs import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_TARGET_SUCCESS_RATE,
    build_output,
    load_network_state,
)
from experiments.scripts.common import parse_task_ids
from experiments.paths import default_catalog_input_path, workflow_experiment_input_path
from control_runtime.integrations.scenario.init_scenario import rebuild_ue_related_tables_from_graph_snapshot
from control_runtime.integrations.storage import get_latest_snapshot_metadata
from training.artifacts import processed_dir, read_json, read_jsonl, write_json, write_jsonl
from training.projection import (
    WORKFLOW_AGENT_NAMES,
    build_finetuning_record,
    build_workflow_record,
    index_traces_by_session,
    load_traces,
    sort_traces,
)
from control_runtime.orchestrators.main_control_orchestrator import MainControlOrchestrator


def _normalize_record(record: Mapping[str, Any], index: int) -> Dict[str, Any]:
    messages = record.get("messages")
    if messages is not None and not isinstance(messages, list):
        raise TypeError(f"record #{index} field 'messages' must be a list when present")

    normalized_messages: List[Dict[str, str]] = []
    if isinstance(messages, list):
        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, Mapping):
                raise TypeError(f"record #{index} message #{message_index} must be an object")
            role = str(message.get("role") or "").strip().lower() or "user"
            content = str(message.get("content") or "")
            if not content.strip():
                continue
            normalized_messages.append({"role": role, "content": content})

    user_input = str(record.get("user_input") or record.get("userInput") or "").strip()
    if not user_input:
        if not normalized_messages:
            raise ValueError(f"record #{index} is missing user_input and usable messages")
        if normalized_messages[-1]["role"] != "user":
            raise ValueError(f"record #{index} messages must end with a user role")
        user_input = normalized_messages[-1]["content"].strip()

    if not normalized_messages:
        normalized_messages = [{"role": "user", "content": user_input}]

    scenario_tags = record.get("scenario_tags") or record.get("scenarioTags") or []
    if scenario_tags and not isinstance(scenario_tags, list):
        raise TypeError(f"record #{index} field 'scenario_tags' must be a list when present")

    task_metadata = record.get("task_metadata") or {}
    if task_metadata and not isinstance(task_metadata, Mapping):
        raise TypeError(f"record #{index} field 'task_metadata' must be an object when present")

    return {
        "record_index": index,
        "user_input": user_input,
        "messages": normalized_messages,
        "context": str(record.get("context") or record.get("conversation_context") or ""),
        "scenario_id": str(record.get("scenario_id") or record.get("scenarioId") or "").strip(),
        "scenario_tags": [str(item).strip() for item in scenario_tags if str(item).strip()],
        "task_metadata": dict(task_metadata or {}),
    }


def load_user_input_records(
    *,
    user_inputs_path: Path | None,
    network_input_path: Path | None,
    count: int,
    target_success_rate: float,
    seed: int,
    start_index: int = 1,
    task_ids: Iterable[str] | str | None = None,
) -> List[Dict[str, Any]]:
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    selected_task_ids = parse_task_ids(task_ids)
    if selected_task_ids and start_index != 1:
        raise ValueError("--task-ids cannot be combined with --start-index")
    if user_inputs_path is not None:
        if not user_inputs_path.exists():
            raise FileNotFoundError(user_inputs_path)
        payload = read_jsonl(user_inputs_path) if user_inputs_path.suffix.lower() == ".jsonl" else read_json(user_inputs_path)
        if isinstance(payload, Mapping):
            raw_records = payload.get("records")
            if not isinstance(raw_records, list):
                raise TypeError(f"{user_inputs_path} must contain a top-level 'records' list")
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise TypeError(f"{user_inputs_path} must contain a JSON object or array")
        if selected_task_ids:
            requested_ids = set(selected_task_ids)
            indexed_records = list(enumerate(raw_records, start=1))
            found_ids: set[str] = set()
            selected_records = []
            for index, record in indexed_records:
                if not isinstance(record, Mapping):
                    continue
                task_metadata = record.get("task_metadata") or {}
                task_id = str(
                    (task_metadata.get("task_id") if isinstance(task_metadata, Mapping) else "")
                    or record.get("task_id")
                    or ""
                ).strip()
                if task_id in requested_ids:
                    found_ids.add(task_id)
                    selected_records.append((index, record))
            missing_ids = [task_id for task_id in selected_task_ids if task_id not in found_ids]
            if missing_ids:
                raise ValueError(
                    "--task-ids contains task ids absent from "
                    f"{user_inputs_path}: {', '.join(missing_ids)}"
                )
            return [_normalize_record(record, index) for index, record in selected_records]

        selected_records = raw_records[start_index - 1 : start_index - 1 + count]
        if not selected_records:
            raise RuntimeError(f"No records found in {user_inputs_path} from start_index={start_index}")
        return [_normalize_record(record, index) for index, record in enumerate(selected_records, start=start_index)]

    if selected_task_ids:
        raise ValueError("--task-ids requires --user-inputs; generated network inputs have no stable task ids")
    network_state = load_network_state(network_input_path)
    payload = build_output(
        network_state,
        count=count,
        target_success_rate=target_success_rate,
        seed=seed,
    )
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise TypeError("generated user input payload must contain a 'records' list")
    selected_records = raw_records[start_index - 1 : start_index - 1 + count]
    if not selected_records:
        raise RuntimeError(f"No generated records available from start_index={start_index}")
    return [_normalize_record(record, index) for index, record in enumerate(selected_records, start=start_index)]


def summarize_run_results(
    results: List[Mapping[str, Any]],
    *,
    result_output: Path,
    workflow_output: Path,
    spec_output: Path | None,
    projection_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    total = len(results)
    successful_invocations = sum(1 for item in results if item.get("status") == "success")
    completed = sum(1 for item in results if item.get("completed"))
    failed = sum(1 for item in results if not item.get("completed"))
    exceptions = sum(1 for item in results if item.get("status") == "error")

    artifacts: Dict[str, Any] = {
        "run_records": str(result_output),
        "workflow_trajectories": str(workflow_output),
        "agent_outputs": dict(projection_summary.get("agent_outputs") or {}),
    }
    if spec_output is not None:
        artifacts["normalized_specs"] = str(spec_output)

    return {
        "total_cases": total,
        "successful_invocation_count": successful_invocations,
        "completed_case_count": completed,
        "error_case_count": failed,
        "exception_case_count": exceptions,
        "successful_invocation_rate": round(successful_invocations / total, 4) if total else 0.0,
        "completed_case_rate": round(completed / total, 4) if total else 0.0,
        "error_case_rate": round(failed / total, 4) if total else 0.0,
        "exception_case_rate": round(exceptions / total, 4) if total else 0.0,
        "artifacts": artifacts,
    }

def _scenario_id(record: Mapping[str, Any], *, prefix: str) -> str:
    explicit = str(record.get("scenario_id") or "").strip()
    if explicit:
        return explicit
    return f"{prefix}-{int(record['record_index']):05d}"


def _merge_tags(record_tags: Iterable[str], extra_tags: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    for item in [*record_tags, *extra_tags]:
        text = str(item).strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _agent_trajectory_output_path(agent_name: str, *, output_root: Path | None) -> Path:
    if output_root is None:
        return processed_dir(agent_name) / f"{agent_name}_trajectories_v1.jsonl"
    return output_root / agent_name / f"{agent_name}_trajectories_v1.jsonl"


def _initialize_projection_outputs(
    *,
    workflow_output: Path,
    agent_output_root: Path | None,
) -> Dict[str, Any]:
    write_jsonl(workflow_output, [])
    agent_paths: Dict[str, str] = {}
    for agent_name in WORKFLOW_AGENT_NAMES:
        output_path = _agent_trajectory_output_path(agent_name, output_root=agent_output_root)
        write_jsonl(output_path, [])
        agent_paths[agent_name] = str(output_path)
    return {
        "workflow_count": 0,
        "workflow_output": str(workflow_output),
        "agent_counts": {agent_name: 0 for agent_name in WORKFLOW_AGENT_NAMES},
        "agent_outputs": agent_paths,
        "session_ids": [],
        "skipped_without_session": 0,
    }


def _append_projected_case(
    *,
    result: Mapping[str, Any],
    workflow_output: Path,
    projection_summary: Dict[str, Any],
) -> None:
    session_id = str(result.get("session_id") or "").strip()
    if not session_id:
        projection_summary["skipped_without_session"] += 1
        return

    session_traces = index_traces_by_session(load_traces(WORKFLOW_AGENT_NAMES)).get(session_id)
    if session_traces is None:
        raise RuntimeError(f"Missing raw traces for workflow session {session_id}")
    if "main_control" not in session_traces:
        raise RuntimeError(f"Workflow session {session_id} is missing main_control raw trace")

    workflow_record = build_workflow_record(
        result,
        session_traces=session_traces,
        agent_order=WORKFLOW_AGENT_NAMES,
    )
    write_jsonl(workflow_output, [workflow_record], mode="a")
    projection_summary["workflow_count"] += 1
    projection_summary["session_ids"].append(session_id)

    for agent_name in WORKFLOW_AGENT_NAMES:
        output_path = Path(projection_summary["agent_outputs"][agent_name])
        for trace in sort_traces(session_traces.get(agent_name, [])):
            write_jsonl(output_path, [build_finetuning_record(trace)], mode="a")
            projection_summary["agent_counts"][agent_name] += 1


def collect_workflow_trajectories(
    *,
    records: List[Dict[str, Any]],
    result_output: Path,
    spec_output: Path | None,
    workflow_output: Path,
    agent_trajectory_output_root: Path | None,
    scenario_prefix: str,
    scenario_tags: List[str],
    max_rounds: int,
    rag_enabled: bool,
    reset_scenario_each_run: bool,
    snapshot_id: str = "",
    use_local_model: bool = False,
    use_deepseek: bool = False,
    case_finished_callback: Callable[[], None] | None = None,
) -> Dict[str, Any]:
    init_db()
    bound_snapshot_id = str(snapshot_id or "").strip()
    if not bound_snapshot_id:
        latest_snapshot = get_latest_snapshot_metadata() or {}
        bound_snapshot_id = str(latest_snapshot.get("snapshot_id") or "").strip()
    if not bound_snapshot_id:
        raise RuntimeError("No latest snapshot found. Seed the scenario snapshot once before collecting trajectories.")

    results: List[Dict[str, Any]] = []
    if spec_output is not None:
        write_jsonl(spec_output, [])
        for record in records:
            spec_record = {
                **record,
                "scenario_id": _scenario_id(record, prefix=scenario_prefix),
                "scenario_tags": _merge_tags(record.get("scenario_tags", []), scenario_tags),
            }
            write_jsonl(spec_output, [spec_record], mode="a")

    write_jsonl(result_output, [])
    projection_summary = _initialize_projection_outputs(
        workflow_output=workflow_output,
        agent_output_root=agent_trajectory_output_root,
    )

    total = len(records)
    for record_position, record in enumerate(records):
        # Reset at the hand-off to the next case, after the previous case's
        # result and projected traces are durable.  Doing it at the end of
        # every iteration needlessly starts a reset after the final case and
        # made worker completion appear to hang on that unrelated reset.
        if record_position > 0 and case_finished_callback is not None:
            case_finished_callback()
        scenario_id = _scenario_id(record, prefix=scenario_prefix)
        merged_tags = _merge_tags(record.get("scenario_tags", []), scenario_tags)
        if reset_scenario_each_run:
            rebuild_ue_related_tables_from_graph_snapshot(bound_snapshot_id)
        orchestrator = MainControlOrchestrator(
            max_rounds=max_rounds,
            use_local_model=use_local_model,
            rag_enabled=rag_enabled,
            use_deepseek=use_deepseek,
        )
        preview = str(record["user_input"]).replace("\n", " ")[:120]
        print(f"[{record['record_index']}/{total}] workflow start: {preview}", flush=True)

        run_started_at = time.perf_counter()
        try:
            run_result = orchestrator.run(
                str(record["user_input"]),
                scenario_id=scenario_id,
                scenario_tags=merged_tags,
                snapshot_id=bound_snapshot_id,
            )
            agent_elapsed_ms = run_result.agent_elapsed_ms
            result_record = {
                "record_index": record["record_index"],
                "task_id": str((record.get("task_metadata") or {}).get("task_id") or "").strip(),
                "category": str((record.get("task_metadata") or {}).get("category") or "").strip(),
                "scenario_id": scenario_id,
                "scenario_tags": merged_tags,
                "user_input": record["user_input"],
                "messages": record["messages"],
                "context": record["context"],
                "task_metadata": dict(record.get("task_metadata") or {}),
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
                "agent_elapsed_ms": dict(agent_elapsed_ms),
                "agent_elapsed_total_ms": float(agent_elapsed_ms["total"]),
                "main_schema_projection_count": run_result.main_schema_projection_count,
                "main_schema_projected_field_count": run_result.main_schema_projected_field_count,
                "main_schema_projected_fields": run_result.main_schema_projected_fields,
            }
            print(
                f"[{record['record_index']}/{total}] workflow done: session_id={run_result.session_id} completed={run_result.completed}",
                flush=True,
            )
        except Exception as exc:
            result_record = {
                "record_index": record["record_index"],
                "task_id": str((record.get("task_metadata") or {}).get("task_id") or "").strip(),
                "category": str((record.get("task_metadata") or {}).get("category") or "").strip(),
                "scenario_id": scenario_id,
                "scenario_tags": merged_tags,
                "user_input": record["user_input"],
                "messages": record["messages"],
                "context": record["context"],
                "task_metadata": dict(record.get("task_metadata") or {}),
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
                "agent_elapsed_ms": {},
                "agent_elapsed_total_ms": 0.0,
                "main_schema_projection_count": orchestrator.main_agent.schema_projection_metrics["count"],
                "main_schema_projected_field_count": orchestrator.main_agent.schema_projection_metrics["field_count"],
                "main_schema_projected_fields": orchestrator.main_agent.schema_projection_metrics["fields"],
            }
            print(
                f"[{record['record_index']}/{total}] workflow error: scenario_id={scenario_id} error={type(exc).__name__}: {exc}",
                flush=True,
            )

        result_record["elapsed_ms"] = round((time.perf_counter() - run_started_at) * 1000, 3)
        results.append(result_record)
        write_jsonl(result_output, [result_record], mode="a")
        _append_projected_case(
            result=result_record,
            workflow_output=workflow_output,
            projection_summary=projection_summary,
        )
    return {
        "results": results,
        "projection_summary": projection_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full main-control workflow in batch using generated user inputs and collect session trajectories.",
    )
    parser.add_argument(
        "--user-inputs",
        type=Path,
        default=None,
        help="Path to generated user inputs (.json/.jsonl). If omitted, the script first tries Multiagents/experiments/generated_inputs/workflow_experiment_user_inputs.json, then Multiagents/experiments/generated_inputs/user_inputs.json, then generates inputs on the fly.",
    )
    parser.add_argument(
        "--network-input",
        type=Path,
        default=None,
        help="Optional network-state JSON used when generating inputs on the fly.",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--start-index", type=int, default=1, help="1-based record index to start from.")
    parser.add_argument("--target-success-rate", type=float, default=DEFAULT_TARGET_SUCCESS_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--snapshot-id", type=str, default="", help="Existing network graph snapshot id to read.")
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument(
        "--local-model",
        action="store_true",
        dest="use_local_model",
        help="Use the OpenAI-compatible local vLLM endpoint configured by vLLM_URL and vLLM_MODEL_NAME.",
    )
    parser.add_argument("--deepseek", action="store_true", dest="use_deepseek", help="Use deepseek-v4-flash for all agents.")
    parser.add_argument("--scenario-prefix", type=str, default="workflow-user-input")
    parser.add_argument(
        "--scenario-tag",
        action="append",
        default=["generated_user_input", "workflow_batch"],
        help="Additional scenario tag. Can be provided multiple times.",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=processed_dir("workflow") / "trajectory_runs_v1.jsonl",
    )
    parser.add_argument(
        "--spec-output",
        type=Path,
        default=processed_dir("workflow") / "trajectory_specs_v1.jsonl",
    )
    parser.add_argument(
        "--workflow-output",
        type=Path,
        default=processed_dir("workflow") / "workflow_trajectories_v1.jsonl",
        help="Tool-bearing workflow trajectory output generated from this batch's sessions.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=processed_dir("workflow") / "workflow_trajectory_summary_v1.json",
        help="Batch summary JSON output.",
    )
    parser.add_argument(
        "--agent-trajectory-output-root",
        type=Path,
        default=None,
        help="Optional root directory for per-agent trajectory outputs. Defaults to each agent's processed directory.",
    )
    parser.add_argument(
        "--no-reset-scenario",
        action="store_true",
        help="Reuse scenario state across runs. Default behavior resets before each run to avoid cross-run contamination.",
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

    default_user_input_candidates = [
        workflow_experiment_input_path(),
        default_catalog_input_path(),
    ]
    user_inputs_path = args.user_inputs
    if user_inputs_path is None:
        for candidate in default_user_input_candidates:
            if candidate.exists():
                user_inputs_path = candidate
                break

    records = load_user_input_records(
        user_inputs_path=user_inputs_path,
        network_input_path=args.network_input,
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
        reset_scenario_each_run=not args.no_reset_scenario,
        snapshot_id=args.snapshot_id,
        use_local_model=args.use_local_model,
        use_deepseek=args.use_deepseek,
    )
    results = collection["results"]
    projection_summary = collection["projection_summary"]

    summary = summarize_run_results(
        results,
        result_output=args.result_output,
        workflow_output=args.workflow_output,
        spec_output=args.spec_output,
        projection_summary=projection_summary,
    )
    write_json(args.summary_output, summary)

    success_count = sum(1 for item in results if item["status"] == "success")
    completed_count = sum(1 for item in results if item["completed"])
    error_count = sum(1 for item in results if item["status"] == "error")
    print(f"Collected {len(results)} workflow runs")
    print(f"Successful invocations: {success_count}")
    print(f"Completed workflows: {completed_count}")
    print(f"Error rows: {error_count}")
    print(f"Run records -> {args.result_output}")
    print(f"Workflow trajectories -> {projection_summary['workflow_output']}")
    for agent_name in WORKFLOW_AGENT_NAMES:
        agent_count = projection_summary["agent_counts"].get(agent_name, 0)
        agent_output = projection_summary["agent_outputs"].get(agent_name, "")
        print(f"{agent_name} trajectories: {agent_count} -> {agent_output}")
    if projection_summary["skipped_without_session"]:
        print(f"Skipped trajectory projection for rows without session_id: {projection_summary['skipped_without_session']}")
    if args.spec_output is not None:
        print(f"Normalized specs -> {args.spec_output}")
    print(f"Summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
