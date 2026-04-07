from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from evalscope import TaskConfig, run_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import eval_dataset_dir, eval_run_dir
from sft_data.schemas import write_jsonl
from sft_data.tool_call.build_evalscope_call_decision_dataset import (
    DEFAULT_AGENT_NAME,
    EvalScopeToolCallDecisionRecord,
    build_evalscope_call_decision_records,
)


def _record_to_evalscope_json(record: EvalScopeToolCallDecisionRecord) -> Dict[str, Any]:
    return {
        "messages": [message.model_dump(mode="json") for message in record.messages],
        "tools": list(record.tools),
        "should_call_tool": bool(record.should_call_tool),
        "metadata": {
            "sample_id": record.sample_id,
            **dict(record.metadata),
        },
    }


def materialize_evalscope_dataset(
    output_dir: Path,
    *,
    agent_name: str = DEFAULT_AGENT_NAME,
    subset_name: str = "default",
    records: List[EvalScopeToolCallDecisionRecord] | None = None,
) -> Path:
    rows = records if records is not None else build_evalscope_call_decision_records(agent_name=agent_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / f"{subset_name}.jsonl"
    write_jsonl(subset_path, [_record_to_evalscope_json(record) for record in rows])
    return subset_path


def _resolve_api_config() -> Tuple[str | None, str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    api_url = os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return api_key, api_url


def build_evalscope_task_config(
    dataset_dir: Path,
    *,
    model_name: str,
    agent_name: str = DEFAULT_AGENT_NAME,
    subset_name: str = "default",
    work_dir: Path | None = None,
    limit: int | None = None,
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
        work_dir=str(work_dir or (eval_run_dir(agent_name, "evalscope", model_name) / "workdir")),
        limit=limit,
    )


def _collect_metric_values(node: Any, collected: Dict[str, Any]) -> None:
    if isinstance(node, Mapping):
        metric_name = node.get("metric_name") or node.get("name")
        score = node.get("score")
        if isinstance(metric_name, str) and metric_name and score is not None:
            collected[metric_name] = score
        for value in node.values():
            _collect_metric_values(value, collected)
        return
    if isinstance(node, list):
        for item in node:
            _collect_metric_values(item, collected)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def extract_evalscope_summary(raw_result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    _collect_metric_values(raw_result, metrics)
    return {
        "tool_call_f1": metrics.get("tool_call_f1"),
        "schema_accuracy": metrics.get("schema_accuracy"),
        "count_finish_reason_tool_call": metrics.get("count_finish_reason_tool_call"),
        "count_successful_tool_call": metrics.get("count_successful_tool_call"),
    }


def run_evalscope_call_decision_evaluation(
    *,
    dataset_dir: Path,
    model_name: str,
    agent_name: str = DEFAULT_AGENT_NAME,
    subset_name: str = "default",
    work_dir: Path | None = None,
    limit: int | None = None,
    run_task_fn: Any = run_task,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    task_cfg = build_evalscope_task_config(
        dataset_dir,
        model_name=model_name,
        agent_name=agent_name,
        subset_name=subset_name,
        work_dir=work_dir,
        limit=limit,
    )
    raw_result = run_task_fn(task_cfg)
    if not isinstance(raw_result, Mapping):
        raise TypeError("EvalScope run_task must return a mapping result")
    normalized_result = _to_jsonable(dict(raw_result))
    if not isinstance(normalized_result, Mapping):
        raise TypeError("Normalized EvalScope result must be a mapping")
    return dict(normalized_result), extract_evalscope_summary(normalized_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the EvalScope general_fc benchmark on the local call/not-call dataset.")
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_NAME,
        help="Agent name used to scope EvalScope artifacts.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory containing EvalScope general_fc subset JSONL files.",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="default",
        help="Subset name used by EvalScope for the local benchmark file.",
    )
    parser.add_argument(
        "--raw-result-output",
        type=Path,
        default=None,
        help="Path to store raw EvalScope run_task result JSON.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Path to store extracted EvalScope metric summary.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        help="Model name passed to EvalScope TaskConfig.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="EvalScope work directory for reports and artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional EvalScope per-subset limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or eval_dataset_dir(args.agent, "evalscope")
    run_dir = eval_run_dir(args.agent, "evalscope", args.model)
    raw_result_output = args.raw_result_output or (run_dir / "raw_result_v1.json")
    summary_output = args.summary_output or (run_dir / "summary_v1.json")
    work_dir = args.work_dir or (run_dir / "workdir")
    materialize_evalscope_dataset(
        dataset_dir,
        agent_name=args.agent,
        subset_name=args.subset_name,
        records=build_evalscope_call_decision_records(agent_name=args.agent),
    )
    raw_result, summary = run_evalscope_call_decision_evaluation(
        dataset_dir=dataset_dir,
        model_name=args.model,
        agent_name=args.agent,
        subset_name=args.subset_name,
        work_dir=work_dir,
        limit=args.limit,
    )
    raw_result_output.parent.mkdir(parents=True, exist_ok=True)
    raw_result_output.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()