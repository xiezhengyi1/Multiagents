from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import ensure_agent_layout, load_trace_records, processed_dir, rejects_dir
from sft_data.formatting import format_tool_call, format_tool_result, normalize_messages
from sft_data.schemas import (
    BuildReport,
    DatasetMessage,
    MinimalTraceRecord,
    ToolCallWarmupRecord,
    load_build_report,
    save_build_report,
    write_jsonl,
)


AGENTS = ("intent_encoding", "optimization_strategy")


def _pair_tool_steps(trace: MinimalTraceRecord) -> List[Dict[str, Any]]:
    results_by_call_id = {
        str(result.get("tool_call_id") or ""): result
        for result in trace.tool_results
        if isinstance(result, dict)
    }
    paired: List[Dict[str, Any]] = []
    for tool_call in trace.tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError(f"Trace {trace.trace_id} contains a non-object tool_call")
        call_id = str(tool_call.get("id") or "")
        if not call_id:
            raise ValueError(f"Trace {trace.trace_id} has a tool_call without id")
        result = results_by_call_id.get(call_id)
        if result is None:
            raise ValueError(f"Trace {trace.trace_id} is missing tool_result for call {call_id}")
        paired.append({"tool_call": tool_call, "tool_result": result})
    if len(paired) != len(trace.tool_results):
        raise ValueError(f"Trace {trace.trace_id} has unmatched tool_results")
    return paired


def _build_record(trace: MinimalTraceRecord) -> ToolCallWarmupRecord | None:
    if trace.status != "success" or not trace.tool_calls:
        return None
    steps = _pair_tool_steps(trace)
    messages = [DatasetMessage(**message) for message in normalize_messages(trace.input_messages)]
    for step in steps:
        messages.append(DatasetMessage(role="assistant", content=format_tool_call(step["tool_call"])))
        messages.append(DatasetMessage(role="tool", content=format_tool_result(step["tool_result"])))
    messages.append(
        DatasetMessage(
            role="assistant",
            content=f"<final_response>{json.dumps(trace.structured_response, ensure_ascii=False)}</final_response>",
        )
    )
    return ToolCallWarmupRecord(
        sample_id=f"warmup:{trace.trace_id}",
        agent=trace.agent_name,
        messages=messages,
        metadata={
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "snapshot_id": trace.snapshot_id,
            "tool_count": len(trace.tool_calls),
            "model_name": trace.model_name,
        },
    )


def build_tool_warmup_records(project_root: Path, agents: Tuple[str, ...] = AGENTS) -> Tuple[List[ToolCallWarmupRecord], List[Dict[str, Any]], BuildReport]:
    records: List[ToolCallWarmupRecord] = []
    rejects: List[Dict[str, Any]] = []
    raw_total = 0
    for agent_name in agents:
        ensure_agent_layout(agent_name)
        trace_file = project_root / "sft_data" / agent_name / "raw_traces" / f"{agent_name}.jsonl"
        traces = load_trace_records(trace_file, MinimalTraceRecord)
        raw_total += len(traces)
        for trace in traces:
            try:
                record = _build_record(trace)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                rejects.append({"kind": "tool_warmup_build_failed", "trace_id": trace.trace_id, "reason": str(exc)})
    report = BuildReport(tool_warmup_samples=len(records), reject_total=len(rejects), artifact_total=raw_total)
    return records, rejects, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tool-call warmup SFT data from agent traces.")
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir("tool_call") / "tool_call_warmup_v1.jsonl",
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=rejects_dir("tool_call") / "tool_call_warmup_rejects_v1.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=processed_dir("tool_call") / "build_report_v1.json",
    )
    args = parser.parse_args()

    records, rejects, report = build_tool_warmup_records(PROJECT_ROOT)
    write_jsonl(args.output, records)
    write_jsonl(args.reject_output, rejects)
    merged_report = load_build_report(args.report_output)
    merged_report.tool_warmup_samples = report.tool_warmup_samples
    merged_report.artifact_total = report.artifact_total
    merged_report.reject_total = report.reject_total
    save_build_report(args.report_output, merged_report)
    print(f"Wrote {len(records)} tool warmup rows to {args.output}")
    print(f"Rejected {len(rejects)} rows")


if __name__ == "__main__":
    main()
