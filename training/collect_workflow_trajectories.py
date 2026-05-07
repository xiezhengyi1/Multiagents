from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = PACKAGE_ROOT / "src"
for candidate in (WORKSPACE_ROOT, PACKAGE_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from database.init_db import init_db
from generate_user_inputs import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_TARGET_SUCCESS_RATE,
    build_output,
    load_network_state,
)
from experiments.paths import default_catalog_input_path, workflow_experiment_input_path
from control_runtime.integrations.scenario.init_scenario import rebuild_ue_related_tables_from_graph_snapshot
from control_runtime.integrations.storage import get_latest_snapshot_metadata
from training.common import ensure_agent_layout, processed_dir
from training.schemas import write_jsonl
from training.trajectory_projection import (
    WORKFLOW_AGENT_NAMES,
    build_workflow_trajectory_record,
    index_projected_traces_by_session,
    load_projected_traces_by_agent,
    sort_traces,
)
from control_runtime.orchestrators.main_control_orchestrator import MainControlOrchestrator


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(payload)
    return rows


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

    return {
        "record_index": index,
        "user_input": user_input,
        "messages": normalized_messages,
        "context": str(record.get("context") or record.get("conversation_context") or ""),
        "scenario_id": str(record.get("scenario_id") or record.get("scenarioId") or "").strip(),
        "scenario_tags": [str(item).strip() for item in scenario_tags if str(item).strip()],
    }


def load_user_input_records(
    *,
    user_inputs_path: Path | None,
    network_input_path: Path | None,
    count: int,
    target_success_rate: float,
    seed: int,
    start_index: int = 1,
) -> List[Dict[str, Any]]:
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    if user_inputs_path is not None:
        if not user_inputs_path.exists():
            raise FileNotFoundError(user_inputs_path)
        payload = _read_jsonl(user_inputs_path) if user_inputs_path.suffix.lower() == ".jsonl" else _read_json(user_inputs_path)
        if isinstance(payload, Mapping):
            raw_records = payload.get("records")
            if not isinstance(raw_records, list):
                raise TypeError(f"{user_inputs_path} must contain a top-level 'records' list")
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise TypeError(f"{user_inputs_path} must contain a JSON object or array")
        selected_records = raw_records[start_index - 1 :]
        if not selected_records:
            raise RuntimeError(f"No records found in {user_inputs_path} from start_index={start_index}")
        return [_normalize_record(record, index) for index, record in enumerate(selected_records, start=start_index)]

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
    selected_records = raw_records[start_index - 1 :]
    if not selected_records:
        raise RuntimeError(f"No generated records available from start_index={start_index}")
    return [_normalize_record(record, index) for index, record in enumerate(selected_records, start=start_index)]


def _write_jsonl_incremental(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _batch_session_ids(results: Iterable[Mapping[str, Any]]) -> List[str]:
    session_ids: List[str] = []
    for result in results:
        session_id = str(result.get("session_id") or "").strip()
        if session_id and session_id not in session_ids:
            session_ids.append(session_id)
    return session_ids


def _agent_trajectory_output_path(agent_name: str, *, output_root: Path | None) -> Path:
    if output_root is None:
        return processed_dir(agent_name) / f"{agent_name}_trajectories_v1.jsonl"
    return output_root / agent_name / f"{agent_name}_trajectories_v1.jsonl"


def _store_projected_trajectories(
    *,
    results: List[Dict[str, Any]],
    workflow_output: Path,
    agent_output_root: Path | None,
) -> Dict[str, Any]:
    session_ids = _batch_session_ids(results)
    traces_by_agent = load_projected_traces_by_agent(PACKAGE_ROOT, agent_names=WORKFLOW_AGENT_NAMES)
    traces_by_session = index_projected_traces_by_session(traces_by_agent)

    workflow_records = []
    agent_records: Dict[str, List[Any]] = {agent_name: [] for agent_name in WORKFLOW_AGENT_NAMES}
    skipped_without_session = 0

    for result in results:
        session_id = str(result.get("session_id") or "").strip()
        if not session_id:
            skipped_without_session += 1
            continue
        session_traces = traces_by_session.get(session_id)
        if session_traces is None:
            raise RuntimeError(f"Missing raw traces for workflow session {session_id}")
        if "main_control" not in session_traces:
            raise RuntimeError(f"Workflow session {session_id} is missing main_control raw trace")
        workflow_records.append(
            build_workflow_trajectory_record(
                result,
                session_traces=session_traces,
                agent_order=WORKFLOW_AGENT_NAMES,
            )
        )
        for agent_name in WORKFLOW_AGENT_NAMES:
            agent_records[agent_name].extend(sort_traces(session_traces.get(agent_name, [])))

    workflow_count = write_jsonl(workflow_output, workflow_records)

    agent_counts: Dict[str, int] = {}
    agent_paths: Dict[str, str] = {}
    for agent_name, records in agent_records.items():
        ensure_agent_layout(agent_name)
        output_path = _agent_trajectory_output_path(agent_name, output_root=agent_output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        agent_counts[agent_name] = write_jsonl(output_path, records)
        agent_paths[agent_name] = str(output_path)

    return {
        "workflow_count": workflow_count,
        "workflow_output": str(workflow_output),
        "agent_counts": agent_counts,
        "agent_outputs": agent_paths,
        "session_ids": session_ids,
        "skipped_without_session": skipped_without_session,
    }


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
    use_deepseek: bool = False,
) -> Dict[str, Any]:
    ensure_agent_layout("workflow")
    init_db()
    bound_snapshot_id = str(snapshot_id or "").strip()
    if not bound_snapshot_id:
        latest_snapshot = get_latest_snapshot_metadata() or {}
        bound_snapshot_id = str(latest_snapshot.get("snapshot_id") or "").strip()
    if not bound_snapshot_id:
        raise RuntimeError("No latest snapshot found. Seed the scenario snapshot once before collecting trajectories.")

    results: List[Dict[str, Any]] = []
    if spec_output is not None:
        spec_output.parent.mkdir(parents=True, exist_ok=True)
        spec_output.write_text("", encoding="utf-8")
        for record in records:
            spec_record = {
                **record,
                "scenario_id": _scenario_id(record, prefix=scenario_prefix),
                "scenario_tags": _merge_tags(record.get("scenario_tags", []), scenario_tags),
            }
            _write_jsonl_incremental(spec_output, spec_record)

    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text("", encoding="utf-8")

    total = len(records)
    for record in records:
        scenario_id = _scenario_id(record, prefix=scenario_prefix)
        merged_tags = _merge_tags(record.get("scenario_tags", []), scenario_tags)
        if reset_scenario_each_run:
            rebuild_ue_related_tables_from_graph_snapshot(bound_snapshot_id)
        orchestrator = MainControlOrchestrator(
            max_rounds=max_rounds,
            use_local_model=False,
            rag_enabled=rag_enabled,
            use_deepseek=use_deepseek,
        )
        preview = str(record["user_input"]).replace("\n", " ")[:120]
        print(f"[{record['record_index']}/{total}] workflow start: {preview}", flush=True)

        try:
            run_result = orchestrator.run(
                str(record["user_input"]),
                scenario_id=scenario_id,
                scenario_tags=merged_tags,
                snapshot_id=bound_snapshot_id,
            )
            result_record = {
                "record_index": record["record_index"],
                "scenario_id": scenario_id,
                "scenario_tags": merged_tags,
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
            print(
                f"[{record['record_index']}/{total}] workflow done: session_id={run_result.session_id} completed={run_result.completed}",
                flush=True,
            )
        except Exception as exc:
            result_record = {
                "record_index": record["record_index"],
                "scenario_id": scenario_id,
                "scenario_tags": merged_tags,
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
            print(
                f"[{record['record_index']}/{total}] workflow error: scenario_id={scenario_id} error={type(exc).__name__}: {exc}",
                flush=True,
            )

        results.append(result_record)
        _write_jsonl_incremental(result_output, result_record)

    projection_summary = _store_projected_trajectories(
        results=results,
        workflow_output=workflow_output,
        agent_output_root=agent_trajectory_output_root,
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
    _write_json(args.summary_output, summary)

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
