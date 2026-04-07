"""
Trace 级别的工具调用评测数据集构建器。

与 build_evalscope_call_decision_dataset.py (单步 call/not-call 边界) 不同，
本模块从 agent 的 raw_traces 中提取每条 trace 的所有 tool_call，
为 evalscope 生成 multi-turn function-call 评测样本。

评测维度：
  1. 调用时机 (call_timing) — 当前对话上下文下，此步是否该调该工具
  2. 返回结果正确性 (result_correctness) — tool_result 是否与预期一致
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import eval_dataset_dir, evaluator_dir, load_trace_records, raw_trace_dir, trace_file
from sft_data.schemas import DatasetMessage, MinimalTraceRecord, write_jsonl
from sft_data.tool_call.build_evalscope_call_decision_dataset import _build_toolset


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class TraceToolCallEvalRecord(BaseModel):
    """一条 trace 中单个 tool_call 的评测记录。"""
    sample_id: str
    trace_id: str
    agent: str
    step_index: int = Field(description="该 tool_call 在 message_trajectory 中的步骤索引")
    # 中文标注：多轮上下文 = trace 开头到本次 tool_call 之前的全部消息
    messages: List[DatasetMessage] = Field(description="截至本次调用前的上下文消息(含 system)")
    tools: List[Dict[str, Any]] = Field(description="可用工具列表 (OpenAI function-calling schema)")
    # 中文标注：调用时机评测字段
    should_call_tool: bool = True
    expected_tool_name: Optional[str] = None
    expected_tool_args: Optional[Dict[str, Any]] = None
    # 中文标注：返回结果正确性评测字段
    actual_tool_result: Optional[str] = None
    result_status: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceToolEvalSummary(BaseModel):
    """评测数据集的统计摘要。"""
    total_traces: int = 0
    success_traces: int = 0
    error_traces: int = 0
    total_tool_calls: int = 0
    tool_name_distribution: Dict[str, int] = Field(default_factory=dict)
    agents: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# trace 解析 → 评测记录
# ---------------------------------------------------------------------------

def _extract_context_before_step(
    trajectory: List[Dict[str, Any]],
    step_index: int,
) -> List[DatasetMessage]:
    """截取 message_trajectory 中 step_index 之前的消息作为评测上下文。"""
    context: List[DatasetMessage] = []
    for msg in trajectory:
        msg_step = msg.get("step_index", -1)
        if msg_step >= step_index:
            break
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            context.append(DatasetMessage(role=role, content=content))
    return context


def _find_tool_result(tool_call_id: str, tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """根据 tool_call_id 找到对应的 tool_result。"""
    for result in tool_results:
        if result.get("tool_call_id") == tool_call_id:
            return result
    return None


def build_trace_tool_eval_records(
    agent_name: str,
    *,
    trace_records: Optional[List[MinimalTraceRecord]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[TraceToolCallEvalRecord]:
    """
    从 agent 的 raw_traces 构建 trace 级别工具调用评测记录。

    中文标注：核心逻辑 — 遍历每条 trace 的 tool_calls，
    为每个 tool_call 生成一条评测样本，包含截至该调用前的全部上下文。
    """
    if trace_records is None:
        tf = trace_file(agent_name)
        if not tf.exists():
            return []
        trace_records = load_trace_records(tf, MinimalTraceRecord)

    if tools is None:
        tools = _build_toolset()

    records: List[TraceToolCallEvalRecord] = []

    for trace in trace_records:
        trajectory = [msg.model_dump(mode="json") for msg in trace.message_trajectory]

        for call_index, tool_call in enumerate(trace.tool_calls):
            tool_name = str(tool_call.get("name") or "").strip()
            if not tool_name or tool_name == "think":
                continue

            tool_call_id = str(tool_call.get("id") or "")
            tool_args = tool_call.get("args") or {}

            # 中文标注：定位该 tool_call 在 trajectory 中的 step_index
            step_index = _locate_tool_call_step(trajectory, tool_call_id, tool_name, call_index)

            # 中文标注：截取上下文
            context_messages = _extract_context_before_step(trajectory, step_index)

            # 中文标注：找到对应的 tool_result
            result = _find_tool_result(tool_call_id, trace.tool_results)
            actual_result = str(result.get("content") or "") if result else None
            result_status = str(result.get("status") or "") if result else None

            sample_id = f"trace-eval-{trace.agent_name}-{trace.trace_id[-8:]}-call-{call_index}"
            records.append(
                TraceToolCallEvalRecord(
                    sample_id=sample_id,
                    trace_id=trace.trace_id,
                    agent=trace.agent_name,
                    step_index=step_index,
                    messages=context_messages,
                    tools=tools,
                    should_call_tool=True,
                    expected_tool_name=tool_name,
                    expected_tool_args=tool_args if isinstance(tool_args, dict) else None,
                    actual_tool_result=actual_result,
                    result_status=result_status,
                    metadata={
                        "trace_id": trace.trace_id,
                        "session_id": trace.session_id,
                        "snapshot_id": trace.snapshot_id,
                        "model_name": trace.model_name,
                        "trace_status": trace.status,
                        "total_tool_calls_in_trace": len(trace.tool_calls),
                        "call_index_in_trace": call_index,
                    },
                )
            )

    return records


def _locate_tool_call_step(
    trajectory: List[Dict[str, Any]],
    tool_call_id: str,
    tool_name: str,
    call_index: int,
) -> int:
    """在 trajectory 中定位 tool_call 对应的 step_index。"""
    # 中文标注：优先匹配 tool_call_id，回退到 tool_name + 位置
    name_matches = []
    for msg in trajectory:
        if msg.get("tool_call_id") == tool_call_id and msg.get("role") == "assistant":
            return msg.get("step_index", 0)
        if msg.get("tool_name") == tool_name and msg.get("role") == "assistant":
            name_matches.append(msg.get("step_index", 0))

    if call_index < len(name_matches):
        return name_matches[call_index]
    if name_matches:
        return name_matches[-1]
    return 0


def build_trace_tool_eval_summary(records: List[TraceToolCallEvalRecord]) -> TraceToolEvalSummary:
    """生成评测数据集的统计摘要。"""
    trace_ids = set()
    agents = set()
    tool_counts: Dict[str, int] = {}

    for record in records:
        trace_ids.add(record.trace_id)
        agents.add(record.agent)
        if record.expected_tool_name:
            tool_counts[record.expected_tool_name] = tool_counts.get(record.expected_tool_name, 0) + 1

    success_traces = len({r.trace_id for r in records if r.metadata.get("trace_status") == "success"})
    error_traces = len({r.trace_id for r in records if r.metadata.get("trace_status") == "error"})

    return TraceToolEvalSummary(
        total_traces=len(trace_ids),
        success_traces=success_traces,
        error_traces=error_traces,
        total_tool_calls=len(records),
        tool_name_distribution=dict(sorted(tool_counts.items())),
        agents=sorted(agents),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build trace-level tool call evaluation dataset from agent raw traces.")
    parser.add_argument("--agent", type=str, default="intent_encoding", help="Agent name.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Summary JSON path.")
    args = parser.parse_args()

    output_path = args.output or (eval_dataset_dir(args.agent, "evalscope") / "trace_tool_eval_records_v1.jsonl")
    summary_path = args.summary_output or (evaluator_dir(args.agent, "evalscope") / "trace_tool_eval_summary_v1.json")

    records = build_trace_tool_eval_records(args.agent)
    summary = build_trace_tool_eval_summary(records)

    write_jsonl(output_path, records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
