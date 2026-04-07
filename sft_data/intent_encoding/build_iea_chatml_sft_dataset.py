from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import dataset_dir, dataset_output_path, load_trace_records
from sft_data.schemas import (
    BuildReport,
    ChatmlSftRecord,
    DatasetMessage,
    MinimalTraceRecord,
    load_build_report,
    save_build_report,
    write_jsonl,
)
from utils import evaluate_intent_encoding_tool_usage


def _extract_tool_call_name_and_args(content: str) -> Tuple[str, Dict[str, object]] | Tuple[None, None]:
    text = str(content or "").strip()
    think_suffix = "</think>"
    if text.startswith("<think>") and think_suffix in text:
        text = text[text.find(think_suffix) + len(think_suffix) :].strip()
    if not text.startswith("<tool_call name=\"") or not text.endswith("</tool_call>"):
        return None, None
    prefix = "<tool_call name=\""
    name_end = text.find("\">", len(prefix))
    if name_end < 0:
        return None, None
    name = text[len(prefix) : name_end]
    payload = text[name_end + 2 : -len("</tool_call>")].strip()
    try:
        args = json.loads(payload)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(args, dict):
        return None, None
    return name, args


def _tool_result_has_legacy_wrapper(content: str) -> bool:
    if not content.startswith("<tool_result>") or not content.endswith("</tool_result>"):
        return False
    inner = content[len("<tool_result>") : -len("</tool_result>")].strip()
    if not inner.startswith("{"):
        return False
    try:
        payload = json.loads(inner)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "content" in payload and any(
        key in payload for key in ("tool_call_id", "name", "status")
    )


def _is_think_message(content: str) -> bool:
    stripped = str(content or "").strip()
    if stripped.startswith("<think>") and stripped.endswith("</think>"):
        return True
    tool_name, tool_args = _extract_tool_call_name_and_args(stripped)
    if tool_name not in {"think", "think_tool"}:
        return False
    return isinstance(tool_args, dict) and str(tool_args.get("message") or "").strip() != ""


def _normalize_iea_trajectory(trace: MinimalTraceRecord) -> List[DatasetMessage]:
    normalized: List[DatasetMessage] = []
    pending_think: str | None = None

    for message in trace.message_trajectory:
        content = str(message.content or "")
        if not content.strip():
            continue

        if message.role == "assistant":
            tool_name, tool_args = _extract_tool_call_name_and_args(content)
            if tool_name in {"think", "think_tool"}:
                think_text = str((tool_args or {}).get("message") or "").strip()
                if not think_text:
                    raise ValueError("think tool call is missing a non-empty message")
                pending_think = f"<think>{think_text}</think>"
                continue
            if tool_name:
                if pending_think is not None:
                    normalized.append(DatasetMessage(role="assistant", content=f"{pending_think}\n{content}"))
                    pending_think = None
                else:
                    normalized.append(DatasetMessage(role="assistant", content=content))
                continue

            if _is_think_message(content):
                pending_think = content
                continue

            if pending_think is not None:
                normalized.append(DatasetMessage(role="assistant", content=pending_think))
                pending_think = None
            normalized.append(DatasetMessage(role="assistant", content=content))
            continue

        if message.role == "tool" and str(message.tool_name or "").strip() in {"think", "think_tool"}:
            continue

        if pending_think is not None:
            normalized.append(DatasetMessage(role="assistant", content=pending_think))
            pending_think = None
        normalized.append(DatasetMessage(role=message.role, content=content))

    if pending_think is not None:
        normalized.append(DatasetMessage(role="assistant", content=pending_think))
    return normalized


def trace_to_iea_chatml_record(
    trace: MinimalTraceRecord,
    *,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> ChatmlSftRecord:
    if trace.status != "success":
        raise ValueError("trace status must be success")
    if not trace.message_trajectory:
        raise ValueError("trace message_trajectory is empty")

    messages = _normalize_iea_trajectory(trace)
    if not messages:
        raise ValueError("trace produced no non-empty trajectory messages")
    tool_usage_report = evaluate_intent_encoding_tool_usage(
        trace.model_dump(mode="json"),
        enable_semantic_judge=enable_semantic_judge,
        semantic_judge=semantic_judge,
    )
    failed_tools = list(tool_usage_report.get("failed_tools") or [])
    hard_validation = tool_usage_report.get("hard_validation") or tool_usage_report
    semantic_validation = tool_usage_report.get("semantic_validation") or None

    return ChatmlSftRecord(
        sample_id=f"iea-chatml:{trace.trace_id}",
        task="iea_chatml_sft",
        agent=trace.agent_name,
        messages=messages,
        metadata={
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "snapshot_id": trace.snapshot_id,
            "tool_count": len(trace.tool_calls),
            "model_name": trace.model_name,
            "tool_usage_valid": bool(tool_usage_report.get("is_valid")),
            "tool_usage_report": tool_usage_report,
            "hard_tool_usage_report": hard_validation,
            "semantic_tool_usage_report": semantic_validation,
            "failed_tools": failed_tools,
            "failed_tool_names": [item.get("tool_name") for item in failed_tools if item.get("tool_name")],
            "semantic_judge_enabled": semantic_validation is not None,
            "semantic_judge_summary": None if semantic_validation is None else semantic_validation.get("summary"),
        },
    )


def _build_tool_usage_failure_reason(record: ChatmlSftRecord) -> str:
    semantic_validation = record.metadata.get("semantic_tool_usage_report") or {}
    failed_tools = list(record.metadata.get("failed_tools") or [])
    if failed_tools:
        labels = []
        for item in failed_tools:
            tool_name = str(item.get("tool_name") or "unknown_tool")
            tool_call_id = str(item.get("tool_call_id") or "")
            issues = "; ".join(str(issue) for issue in item.get("issues") or [])
            label = f"{tool_name}({tool_call_id})" if tool_call_id else tool_name
            labels.append(f"{label}: {issues}" if issues else label)
        suffix = ""
        semantic_summary = str(semantic_validation.get("summary") or "").strip()
        if semantic_summary:
            suffix = f"; semantic summary: {semantic_summary}"
        return "trace failed intent encoding tool usage validation; failed tools: " + " | ".join(labels) + suffix

    trace_errors = list((record.metadata.get("tool_usage_report") or {}).get("trace_errors") or [])
    if trace_errors:
        return "trace failed intent encoding tool usage validation; trace errors: " + " | ".join(str(item) for item in trace_errors)

    return "trace failed intent encoding tool usage validation"


def split_iea_chatml_records(records: List[ChatmlSftRecord]) -> Tuple[List[ChatmlSftRecord], List[ChatmlSftRecord]]:
    # 关键步骤：将成功样本与工具使用失败样本拆开，便于微调与误差分析分别消费。
    success_records: List[ChatmlSftRecord] = []
    failure_records: List[ChatmlSftRecord] = []
    for record in records:
        if bool(record.metadata.get("tool_usage_valid")):
            success_records.append(record)
        else:
            failure_records.append(record)
    return success_records, failure_records


def _count_chatml_build_errors(rejects: List[Dict[str, str]]) -> int:
    return sum(1 for reject in rejects if reject.get("kind") != "iea_tool_usage_validation_failed")


def build_iea_chatml_records(
    project_root: Path,
    *,
    session_ids: Optional[Set[str]] = None,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> Tuple[List[ChatmlSftRecord], List[Dict[str, str]], BuildReport]:
    trace_path = project_root / "sft_data" / "intent_encoding" / "raw_traces" / "intent_encoding.jsonl"
    traces = load_trace_records(trace_path, MinimalTraceRecord)
    records: List[ChatmlSftRecord] = []
    rejects: List[Dict[str, str]] = []

    for trace in traces:
        if session_ids is not None and trace.session_id not in session_ids:
            continue
        try:
            record = trace_to_iea_chatml_record(
                trace,
                enable_semantic_judge=enable_semantic_judge,
                semantic_judge=semantic_judge,
            )
            records.append(record)
            if not bool(record.metadata.get("tool_usage_valid")):
                rejects.append(
                    {
                        "kind": "iea_tool_usage_validation_failed",
                        "trace_id": trace.trace_id,
                        "reason": _build_tool_usage_failure_reason(record),
                        "failed_tools": record.metadata.get("failed_tools", []),
                        "semantic_tool_usage_report": record.metadata.get("semantic_tool_usage_report"),
                    }
                )
        except Exception as exc:
            rejects.append(
                {
                    "kind": "iea_chatml_build_failed",
                    "trace_id": trace.trace_id,
                    "reason": str(exc),
                }
            )

    report = BuildReport(
        artifact_total=len(traces),
        paired_total=len(records),
        iea_sft_samples=len(records),
        reject_total=len(rejects),
    )
    return records, rejects, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IEA ChatML SFT data from intent encoding traces.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_output_path("intent_encoding", "chatml", "success", "iea_chatml_sft_v1.jsonl"),
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "chatml", "failure", "iea_chatml_sft_v1.jsonl"),
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "chatml", "failure", "iea_chatml_rejects_v1.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=dataset_dir("intent_encoding", "chatml") / "build_report_v1.json",
    )
    args = parser.parse_args()

    records, rejects, report = build_iea_chatml_records(args.project_root)
    success_records, failure_records = split_iea_chatml_records(records)
    write_jsonl(args.output, success_records)
    write_jsonl(args.failure_output, failure_records)
    write_jsonl(args.reject_output, rejects)
    merged_report = load_build_report(args.report_output)
    merged_report.artifact_total = report.artifact_total
    merged_report.paired_total = len(records)
    merged_report.iea_sft_samples = len(success_records)
    merged_report.reject_total = len(failure_records) + _count_chatml_build_errors(rejects)
    save_build_report(args.report_output, merged_report)
    print(f"Wrote {len(success_records)} IEA ChatML success rows to {args.output}")
    print(f"Wrote {len(failure_records)} IEA ChatML failure rows to {args.failure_output}")
    print(f"Rejected {len(rejects)} rows")


if __name__ == "__main__":
    main()
