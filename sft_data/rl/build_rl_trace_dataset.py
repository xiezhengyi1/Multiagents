from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import ensure_agent_layout, load_trace_records, processed_dir, rejects_dir
from sft_data.formatting import normalize_messages
from sft_data.schemas import (
    AgenticRlTraceRecord,
    BuildReport,
    DatasetMessage,
    MinimalTraceRecord,
    load_build_report,
    save_build_report,
    write_jsonl,
)


AGENTS = ("intent_encoding", "optimization_strategy")


def _build_record(trace: MinimalTraceRecord) -> AgenticRlTraceRecord:
    if not trace.input_messages:
        raise ValueError(f"Trace {trace.trace_id} has no input_messages")
    observation = [DatasetMessage(**message) for message in normalize_messages(trace.input_messages)]
    tool_trajectory: List[Dict[str, Any]] = []
    results_by_call_id = {
        str(result.get("tool_call_id") or ""): result
        for result in trace.tool_results
        if isinstance(result, dict)
    }
    for tool_call in trace.tool_calls:
        call_id = str(tool_call.get("id") or "")
        tool_trajectory.append(
            {
                "tool_name": tool_call.get("name"),
                "tool_args": tool_call.get("args"),
                "tool_result": results_by_call_id.get(call_id),
            }
        )
    return AgenticRlTraceRecord(
        sample_id=f"rl:{trace.trace_id}",
        agent=trace.agent_name,
        observation=observation,
        tool_trajectory=tool_trajectory,
        final_output=trace.structured_response,
        metadata={
            "trace_id": trace.trace_id,
            "timestamp": trace.timestamp,
            "session_id": trace.session_id,
            "snapshot_id": trace.snapshot_id,
            "thread_id": trace.thread_id,
            "model_name": trace.model_name,
            "status": trace.status,
            "error": trace.error,
        },
    )


def build_rl_trace_records(project_root: Path, agents: Tuple[str, ...] = AGENTS) -> Tuple[List[AgenticRlTraceRecord], List[Dict[str, Any]], BuildReport]:
    records: List[AgenticRlTraceRecord] = []
    rejects: List[Dict[str, Any]] = []
    raw_total = 0
    for agent_name in agents:
        ensure_agent_layout(agent_name)
        trace_file = project_root / "sft_data" / agent_name / "raw_traces" / f"{agent_name}.jsonl"
        traces = load_trace_records(trace_file, MinimalTraceRecord)
        raw_total += len(traces)
        for trace in traces:
            try:
                records.append(_build_record(trace))
            except Exception as exc:
                rejects.append({"kind": "rl_trace_build_failed", "trace_id": trace.trace_id, "reason": str(exc)})
    report = BuildReport(rl_trace_samples=len(records), reject_total=len(rejects), artifact_total=raw_total)
    return records, rejects, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simplified RL trace data from agent traces.")
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir("rl") / "agentic_rl_trace_v1.jsonl",
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=rejects_dir("rl") / "agentic_rl_trace_rejects_v1.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=processed_dir("rl") / "build_report_v1.json",
    )
    args = parser.parse_args()

    records, rejects, report = build_rl_trace_records(PROJECT_ROOT)
    write_jsonl(args.output, records)
    write_jsonl(args.reject_output, rejects)
    merged_report = load_build_report(args.report_output)
    merged_report.rl_trace_samples = report.rl_trace_samples
    merged_report.artifact_total = report.artifact_total
    merged_report.reject_total = report.reject_total
    save_build_report(args.report_output, merged_report)
    print(f"Wrote {len(records)} RL trace rows to {args.output}")
    print(f"Rejected {len(rejects)} rows")


if __name__ == "__main__":
    main()
