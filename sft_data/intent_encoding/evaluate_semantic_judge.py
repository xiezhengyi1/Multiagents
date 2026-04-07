from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import evaluator_dir, load_trace_records
from sft_data.schemas import MinimalTraceRecord, write_jsonl
from utils import ToolUsageSemanticJudge, evaluate_intent_encoding_tool_usage


def _extract_user_text(trace: MinimalTraceRecord) -> str:
    for message in reversed(trace.input_messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or message.get("type") or "").strip().lower()
        if role in {"user", "human"}:
            return str(message.get("content") or "").strip()
    return ""


def _trim_preview(text: str, *, limit: int = 160) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _iter_filtered_traces(
    traces: Sequence[MinimalTraceRecord],
    *,
    session_ids: Optional[Set[str]],
    status_filter: str,
) -> Iterable[MinimalTraceRecord]:
    for trace in traces:
        if session_ids is not None and trace.session_id not in session_ids:
            continue
        if status_filter != "all" and trace.status != status_filter:
            continue
        yield trace


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_counts(rows: Iterable[str], *, limit: int = 10) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for item in rows:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [{"name": name, "count": count} for name, count in ordered[:limit]]


def _build_evaluation_row(trace: MinimalTraceRecord, report: Mapping[str, Any]) -> Dict[str, Any]:
    hard_validation = report.get("hard_validation") or report
    semantic_validation = report.get("semantic_validation") or {}
    hard_valid = bool(hard_validation.get("is_valid"))
    semantic_enabled = bool(semantic_validation)
    semantic_valid = None if not semantic_enabled else bool(semantic_validation.get("is_valid"))
    disagreement_tags: List[str] = []

    # 关键步骤：把“硬规则通过但语义 Judge 失败”的成功 trace 单独标成潜在 false positive。
    if hard_valid and semantic_valid is False:
        disagreement_tags.append("semantic_false_positive_candidate")
    if not hard_valid and semantic_valid is True:
        disagreement_tags.append("semantic_false_negative_candidate")
    if not hard_valid and semantic_valid is False:
        disagreement_tags.append("hard_and_semantic_invalid")

    semantic_failed_tools = list(semantic_validation.get("failed_tools") or [])
    return {
        "trace_id": trace.trace_id,
        "session_id": trace.session_id,
        "status": trace.status,
        "model_name": trace.model_name,
        "tool_count": len(trace.tool_calls),
        "user_input_preview": _trim_preview(_extract_user_text(trace)),
        "overall_valid": bool(report.get("is_valid")),
        "hard_valid": hard_valid,
        "semantic_valid": semantic_valid,
        "disagreement_tags": disagreement_tags,
        "failed_tool_names": [item.get("tool_name") for item in report.get("failed_tools") or [] if item.get("tool_name")],
        "hard_trace_errors": list(hard_validation.get("trace_errors") or []),
        "semantic_trace_errors": list(semantic_validation.get("trace_errors") or []),
        "semantic_summary": semantic_validation.get("summary"),
        "semantic_failed_tools": semantic_failed_tools,
        "semantic_dimension_scores": dict(semantic_validation.get("dimension_scores") or {}),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, total_filtered: int) -> Dict[str, Any]:
    hard_valid_count = 0
    semantic_valid_count = 0
    overall_valid_count = 0
    semantic_only_invalid_count = 0
    hard_only_invalid_count = 0
    both_invalid_count = 0
    semantic_tool_names: List[str] = []
    dimension_totals: Dict[str, float] = {}
    dimension_counts: Dict[str, int] = {}

    for row in rows:
        hard_valid = bool(row.get("hard_valid"))
        semantic_valid = row.get("semantic_valid")
        overall_valid = bool(row.get("overall_valid"))
        if hard_valid:
            hard_valid_count += 1
        if semantic_valid is True:
            semantic_valid_count += 1
        if overall_valid:
            overall_valid_count += 1
        if hard_valid and semantic_valid is False:
            semantic_only_invalid_count += 1
        elif not hard_valid and semantic_valid is True:
            hard_only_invalid_count += 1
        elif not hard_valid and semantic_valid is False:
            both_invalid_count += 1

        for item in row.get("semantic_failed_tools") or []:
            tool_name = str((item or {}).get("tool_name") or "").strip()
            if tool_name:
                semantic_tool_names.append(tool_name)

        for name, value in (row.get("semantic_dimension_scores") or {}).items():
            numeric_value = _safe_float(value)
            if numeric_value is None:
                continue
            dimension_totals[name] = dimension_totals.get(name, 0.0) + numeric_value
            dimension_counts[name] = dimension_counts.get(name, 0) + 1

    dimension_averages = {
        name: round(dimension_totals[name] / dimension_counts[name], 4)
        for name in sorted(dimension_totals)
        if dimension_counts.get(name)
    }
    semantic_fp_candidates = [row for row in rows if "semantic_false_positive_candidate" in (row.get("disagreement_tags") or [])]
    return {
        "total_filtered_traces": total_filtered,
        "evaluated_traces": len(rows),
        "hard_valid_count": hard_valid_count,
        "semantic_valid_count": semantic_valid_count,
        "overall_valid_count": overall_valid_count,
        "semantic_only_invalid_count": semantic_only_invalid_count,
        "hard_only_invalid_count": hard_only_invalid_count,
        "both_invalid_count": both_invalid_count,
        "semantic_false_positive_candidate_count": len(semantic_fp_candidates),
        "semantic_false_positive_candidate_trace_ids": [row.get("trace_id") for row in semantic_fp_candidates],
        "top_semantic_failed_tools": _top_counts(semantic_tool_names),
        "average_dimension_scores": dimension_averages,
    }


def evaluate_semantic_judge_on_traces(
    project_root: Path,
    *,
    limit: int | None = None,
    session_ids: Optional[Set[str]] = None,
    status_filter: str = "success",
    semantic_judge: Any = None,
    judge_model_name: str = "qwen-plus",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trace_path = project_root / "sft_data" / "intent_encoding" / "raw_traces" / "intent_encoding.jsonl"
    traces = load_trace_records(trace_path, MinimalTraceRecord)
    filtered_traces = list(
        _iter_filtered_traces(
            traces,
            session_ids=session_ids,
            status_filter=status_filter,
        )
    )
    if limit is not None:
        filtered_traces = filtered_traces[:limit]

    judge = semantic_judge if semantic_judge is not None else ToolUsageSemanticJudge(model_name=judge_model_name)
    rows: List[Dict[str, Any]] = []

    for trace in filtered_traces:
        report = evaluate_intent_encoding_tool_usage(
            trace.model_dump(mode="json"),
            enable_semantic_judge=True,
            semantic_judge=judge,
        )
        rows.append(_build_evaluation_row(trace, report))

    summary = _summarize_rows(rows, total_filtered=len(filtered_traces))
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the intent_encoding semantic judge on real traces.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of filtered traces to evaluate.")
    parser.add_argument(
        "--status",
        choices=["success", "error", "all"],
        default="success",
        help="Filter traces by status before evaluation.",
    )
    parser.add_argument(
        "--session-id",
        action="append",
        default=None,
        help="Optional session id filter. Repeat the flag to evaluate multiple sessions.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="qwen-plus",
        help="Model name used by ToolUsageSemanticJudge when no custom judge is injected.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=evaluator_dir("intent_encoding", "semantic_judge") / "semantic_judge_eval_v1.jsonl",
        help="Per-trace evaluation output JSONL.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=evaluator_dir("intent_encoding", "semantic_judge") / "semantic_judge_eval_summary_v1.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=evaluator_dir("intent_encoding", "semantic_judge") / "semantic_judge_false_positive_candidates_v1.jsonl",
        help="JSONL path for semantic false-positive candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    rows, summary = evaluate_semantic_judge_on_traces(
        args.project_root,
        limit=args.limit,
        session_ids=set(args.session_id) if args.session_id else None,
        status_filter=args.status,
        judge_model_name=args.judge_model,
    )
    candidates = [row for row in rows if "semantic_false_positive_candidate" in (row.get("disagreement_tags") or [])]

    write_jsonl(args.output, rows)
    write_jsonl(args.candidates_output, candidates)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()