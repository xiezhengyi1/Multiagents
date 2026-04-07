"""
Trace 级别的工具调用评测执行器。

使用 evalscope 的 general_fc 基准，对每条 trace 中的 tool_call 进行：
  1. 调用时机评测 — 当前上下文是否应该调用该工具 (tool_call_f1)
  2. 参数正确性评测 — schema_accuracy
  3. 返回结果正确性评测 — 本地规则校验 (非 LLM 评测)

中文标注：与 evaluate_evalscope_call_decision.py 共享 evalscope TaskConfig 构建逻辑，
但输入数据来自 trace 而非手工用例。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalscope import TaskConfig, run_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import eval_dataset_dir, eval_run_dir, evaluator_dir
from sft_data.schemas import write_jsonl
from sft_data.tool_call.build_evalscope_trace_tool_dataset import (
    TraceToolCallEvalRecord,
    build_trace_tool_eval_records,
    build_trace_tool_eval_summary,
)
from sft_data.tool_call.evaluate_evalscope_call_decision import (
    _resolve_api_config,
    _to_jsonable,
    extract_evalscope_summary,
)


# ---------------------------------------------------------------------------
# 数据集物化
# ---------------------------------------------------------------------------

def _record_to_evalscope_json(record: TraceToolCallEvalRecord) -> Dict[str, Any]:
    """将 trace eval record 转为 evalscope general_fc 格式。"""
    return {
        "messages": [msg.model_dump(mode="json") for msg in record.messages],
        "tools": list(record.tools),
        "should_call_tool": bool(record.should_call_tool),
        "metadata": {
            "sample_id": record.sample_id,
            "trace_id": record.trace_id,
            "step_index": record.step_index,
            "expected_tool_name": record.expected_tool_name,
            "expected_tool_args": record.expected_tool_args,
            **record.metadata,
        },
    }


def materialize_trace_tool_dataset(
    output_dir: Path,
    *,
    agent_name: str = "intent_encoding",
    subset_name: str = "trace_tools",
    records: Optional[List[TraceToolCallEvalRecord]] = None,
) -> Path:
    rows = records if records is not None else build_trace_tool_eval_records(agent_name)
    if not rows:
        raise RuntimeError(f"No trace tool eval records found for agent '{agent_name}'. Run the agent first to generate traces.")
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / f"{subset_name}.jsonl"
    write_jsonl(subset_path, [_record_to_evalscope_json(r) for r in rows])
    return subset_path


# ---------------------------------------------------------------------------
# 本地结果正确性校验（不依赖 LLM）
# ---------------------------------------------------------------------------

def validate_tool_results(records: List[TraceToolCallEvalRecord]) -> List[Dict[str, Any]]:
    """
    中文标注：本地规则校验 tool_result 的正确性。
    返回每条记录的校验结果。
    """
    validations: List[Dict[str, Any]] = []
    for record in records:
        verdict: Dict[str, Any] = {
            "sample_id": record.sample_id,
            "trace_id": record.trace_id,
            "tool_name": record.expected_tool_name,
            "step_index": record.step_index,
            "checks": [],
        }

        # 中文标注：检查1 — tool 是否返回了结果
        has_result = record.actual_tool_result is not None and record.actual_tool_result.strip() != ""
        verdict["checks"].append({
            "check": "has_result",
            "passed": has_result,
            "detail": "Tool returned non-empty result" if has_result else "Tool returned empty or missing result",
        })

        # 中文标注：检查2 — result_status 是否为 success
        status_ok = record.result_status in {"success", None}
        verdict["checks"].append({
            "check": "status_success",
            "passed": status_ok,
            "detail": f"status={record.result_status}",
        })

        # 中文标注：检查3 — 结果中是否包含错误模式
        error_patterns = ["error", "traceback", "exception", "failed to"]
        result_text = (record.actual_tool_result or "").lower()
        contains_error = any(pattern in result_text for pattern in error_patterns)
        verdict["checks"].append({
            "check": "no_error_pattern",
            "passed": not contains_error,
            "detail": "Result contains error pattern" if contains_error else "No error patterns detected",
        })

        # 中文标注：检查4 — 对特定工具校验返回格式
        if record.expected_tool_name in {"get_ue_context", "get_ue_flow_catalog"}:
            # 这些工具应返回包含 supi 信息的结构化结果
            has_supi = "supi" in result_text or "imsi" in result_text
            verdict["checks"].append({
                "check": "contains_expected_field",
                "passed": has_supi,
                "detail": "Result contains supi/imsi reference" if has_supi else "Result missing expected supi/imsi field",
            })

        verdict["all_passed"] = all(c["passed"] for c in verdict["checks"])
        validations.append(verdict)

    return validations


# ---------------------------------------------------------------------------
# evalscope 集成评测
# ---------------------------------------------------------------------------

def build_trace_tool_task_config(
    dataset_dir: Path,
    *,
    model_name: str,
    agent_name: str = "intent_encoding",
    subset_name: str = "trace_tools",
    work_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TaskConfig:
    api_key, api_url = _resolve_api_config()
    return TaskConfig(
        model=model_name,
        api_url=api_url,
        api_key=api_key or "EMPTY",
        datasets=["general_fc"],
        dataset_args={
            "general_fc": {
                "local_path": str(dataset_dir),
                "subset_list": [subset_name],
            }
        },
        work_dir=str(work_dir or (eval_run_dir(agent_name, "evalscope_trace", model_name) / "workdir")),
        limit=limit,
    )


def run_trace_tool_evaluation(
    *,
    dataset_dir: Path,
    model_name: str,
    agent_name: str = "intent_encoding",
    subset_name: str = "trace_tools",
    work_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    run_task_fn: Any = run_task,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    中文标注：完整评测流程 — evalscope LLM 评测 + 本地结果校验。
    返回 (evalscope_raw_result, evalscope_summary, local_validations)
    """
    # 1. evalscope 评测 (调用时机 + 参数正确性)
    task_cfg = build_trace_tool_task_config(
        dataset_dir,
        model_name=model_name,
        agent_name=agent_name,
        subset_name=subset_name,
        work_dir=work_dir,
        limit=limit,
    )
    raw_result = run_task_fn(task_cfg)
    if not isinstance(raw_result, dict):
        raise TypeError("evalscope run_task must return a mapping")
    normalized = _to_jsonable(dict(raw_result))
    summary = extract_evalscope_summary(normalized)

    # 2. 本地结果正确性校验
    records = build_trace_tool_eval_records(agent_name)
    validations = validate_tool_results(records)

    # 中文标注：合并 evalscope 指标 + 本地校验指标
    total_checks = sum(len(v["checks"]) for v in validations)
    passed_checks = sum(sum(1 for c in v["checks"] if c["passed"]) for v in validations)
    summary["local_result_accuracy"] = passed_checks / total_checks if total_checks > 0 else None
    summary["local_checks_total"] = total_checks
    summary["local_checks_passed"] = passed_checks
    summary["local_all_passed_count"] = sum(1 for v in validations if v["all_passed"])
    summary["local_all_passed_rate"] = (
        summary["local_all_passed_count"] / len(validations) if validations else None
    )

    return dict(normalized), summary, validations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trace-level tool calls using evalscope + local validation.")
    parser.add_argument("--agent", type=str, default="intent_encoding")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--subset-name", type=str, default="trace_tools")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--validations-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or eval_dataset_dir(args.agent, "evalscope_trace")
    run_dir = eval_run_dir(args.agent, "evalscope_trace", args.model)

    # 物化数据集
    records = build_trace_tool_eval_records(args.agent)
    dataset_summary = build_trace_tool_eval_summary(records)
    materialize_trace_tool_dataset(dataset_dir, agent_name=args.agent, subset_name=args.subset_name, records=records)

    # 执行评测
    raw_result, summary, validations = run_trace_tool_evaluation(
        dataset_dir=dataset_dir,
        model_name=args.model,
        agent_name=args.agent,
        subset_name=args.subset_name,
        limit=args.limit,
    )

    # 输出
    summary_path = args.summary_output or (run_dir / "trace_eval_summary_v1.json")
    validations_path = args.validations_output or (run_dir / "trace_result_validations_v1.json")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    validations_path.parent.mkdir(parents=True, exist_ok=True)
    validations_path.write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Dataset Summary ===")
    print(json.dumps(dataset_summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
