from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.tools import (
    get_knowledge_by_key,
    get_ue_context,
    get_ue_flow_catalog,
    search_flow_targets_by_name,
    search_semantic_knowledge,
)
from sft_data.common import eval_dataset_dir, evaluator_dir
from sft_data.schemas import DatasetMessage, write_jsonl


DEFAULT_AGENT_NAME = "intent_encoding"


class EvalScopeToolCallDecisionRecord(BaseModel):
    sample_id: str
    task: str = "general_function_call_boundary"
    agent: str = DEFAULT_AGENT_NAME
    messages: List[DatasetMessage]
    tools: List[Dict[str, Any]]
    should_call_tool: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _normalize_parameters_schema(tool: Any) -> Dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    properties = dict(getattr(tool, "args", None) or {})
    if args_schema is None or not hasattr(args_schema, "model_fields"):
        return {
            "type": "object",
            "properties": properties,
            "required": [],
            "additionalProperties": False,
        }

    required = [
        field_name
        for field_name, field_info in args_schema.model_fields.items()
        if field_name != "runtime" and field_info.is_required()
    ]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _build_toolset() -> List[Dict[str, Any]]:
    # 关键步骤：这里只评真实业务工具边界，不把 think_tool 纳入单步 call/not-call 基准。
    toolset: List[Dict[str, Any]] = []
    for tool in [get_ue_context, get_ue_flow_catalog, search_flow_targets_by_name, search_semantic_knowledge, get_knowledge_by_key]:
        toolset.append(
            {
                "type": "function",
                "function": {
                    "name": str(getattr(tool, "name", "") or "").strip(),
                    "description": str(getattr(tool, "description", "") or "").strip(),
                    "parameters": _normalize_parameters_schema(tool),
                },
            }
        )
    return toolset


def _build_record(
    *,
    sample_id: str,
    user_text: str,
    should_call_tool: bool,
    rationale: str,
    boundary_type: str,
    agent_name: str = DEFAULT_AGENT_NAME,
    expected_tool_name: str | None = None,
    expected_args: Dict[str, Any] | None = None,
    allowed_tool_names: List[str] | None = None,
) -> EvalScopeToolCallDecisionRecord:
    return EvalScopeToolCallDecisionRecord(
        sample_id=sample_id,
        agent=agent_name,
        messages=[DatasetMessage(role="user", content=user_text)],
        tools=_build_toolset(),
        should_call_tool=should_call_tool,
        metadata={
            "benchmark_style": "EvalScope-GeneralFunctionCall",
            "boundary_type": boundary_type,
            "rationale": rationale,
            "expected_tool_name": expected_tool_name,
            "expected_tool_args": expected_args,
            "allowed_tool_names": allowed_tool_names or ([] if expected_tool_name is None else [expected_tool_name]),
        },
    )


def build_evalscope_call_decision_records(agent_name: str = DEFAULT_AGENT_NAME) -> List[EvalScopeToolCallDecisionRecord]:
    return [
        _build_record(
            sample_id="evalscope-iea-001",
            user_text="imsi-20893001 下面那个视频流有点卡，先帮我把优先级往上抬一下。",
            should_call_tool=True,
            agent_name=agent_name,
            expected_tool_name="get_ue_flow_catalog",
            expected_args={"supi": "imsi-20893001"},
            rationale="用户给了 SUPI，但没有给 flow_id，需要先查 UE catalog 才能定位具体视频流。",
            boundary_type="flow_resolution",
        ),
        _build_record(
            sample_id="evalscope-iea-002",
            user_text="先别直接改，先看一下 imsi-20893001 现在默认 QoS 和活动策略，再判断控制流还能不能提速。",
            should_call_tool=True,
            agent_name=agent_name,
            expected_tool_name="get_ue_context",
            expected_args={"supi": "imsi-20893001"},
            rationale="用户明确要求先查看当前 UE 状态和策略上下文，单步决策应直接命中 get_ue_context。",
            boundary_type="current_policy_context",
        ),
        _build_record(
            sample_id="evalscope-iea-003",
            user_text="SmPolicyDecision 里面那个 pccRules 到底是什么意思，你先给我按标准对象解释清楚。",
            should_call_tool=True,
            agent_name=agent_name,
            expected_tool_name="get_knowledge_by_key",
            expected_args={"key": "SmPolicyDecision", "category": "sm_policy"},
            rationale="用户直接点名精确 3GPP 对象，应该先用 exact lookup，而不是泛化语义搜索。",
            boundary_type="exact_schema_lookup",
        ),
        _build_record(
            sample_id="evalscope-iea-004",
            user_text="我想知道 policy trigger 这类触发条件彼此语义上有什么差别，不是问某个固定字段。",
            should_call_tool=True,
            agent_name=agent_name,
            expected_tool_name="search_semantic_knowledge",
            expected_args={"query": "policy trigger semantic difference", "category": "sm_policy"},
            rationale="这是描述性知识查询，不是精确对象键，更适合先做 semantic search。",
            boundary_type="descriptive_knowledge_search",
        ),
        _build_record(
            sample_id="evalscope-iea-005",
            user_text="Traffic descriptor 和 Route selection descriptor 这两个到底差在哪，按标准对象给我讲。",
            should_call_tool=True,
            agent_name=agent_name,
            expected_tool_name="get_knowledge_by_key",
            expected_args={"key": "Traffic descriptor", "category": "ursp"},
            allowed_tool_names=["get_knowledge_by_key"],
            rationale="用户直接点名标准对象，单步边界应优先 exact lookup，而不是 descriptive search。",
            boundary_type="exact_descriptor_lookup",
        ),
        _build_record(
            sample_id="evalscope-iea-006",
            user_text="Cloud Render 那条 telemetry 先帮我定位一下，我忘了是哪个 UE 了。",
            should_call_tool=True,
            agent_name=agent_name,
            expected_tool_name="search_flow_targets_by_name",
            expected_args={"app_name": "Cloud Render", "flow_name": "telemetry", "limit": 5},
            rationale="用户只给了 app_name/flow_name，没有 SUPI，应该先做跨 UE 的语义目标检索。",
            boundary_type="semantic_target_lookup",
        ),
        _build_record(
            sample_id="evalscope-iea-007",
            user_text="把这句用户请求改得更正式一点：把视频流处理一下。",
            should_call_tool=False,
            agent_name=agent_name,
            rationale="这是纯文本改写任务，不依赖实时 UE 数据或标准知识。",
            boundary_type="pure_rewrite",
        ),
        _build_record(
            sample_id="evalscope-iea-008",
            user_text="已知 supi=imsi-20893001、app_id=app-0001、flow_id=flow-0001，先给我起一个 increase 意图草稿。",
            should_call_tool=False,
            agent_name=agent_name,
            rationale="关键标识已经完整给出，单步边界不需要再调用运行时工具。",
            boundary_type="fully_grounded_request",
        ),
        _build_record(
            sample_id="evalscope-iea-009",
            user_text="你刚才编码出来的意思，给我用一句中文再说一遍就行。",
            should_call_tool=False,
            agent_name=agent_name,
            rationale="这是对已有结果的总结，不需要重新访问工具。",
            boundary_type="followup_summary",
        ),
        _build_record(
            sample_id="evalscope-iea-010",
            user_text="只给我一个空的 OperationIntent JSON 模板，别去查任何实时信息。",
            should_call_tool=False,
            agent_name=agent_name,
            rationale="用户已经明确禁止查询实时信息，边界应判断为 not-call。",
            boundary_type="explicit_no_tool",
        ),
        _build_record(
            sample_id="evalscope-iea-011",
            user_text="你为什么通常会先查 flow catalog 再编码意图，简单解释一下。",
            should_call_tool=False,
            agent_name=agent_name,
            rationale="这是 agent 行为说明，不需要访问运行时数据。",
            boundary_type="process_explanation",
        ),
        _build_record(
            sample_id="evalscope-iea-012",
            user_text="把这句翻译成英文：请把控制流时延降到 10ms。",
            should_call_tool=False,
            agent_name=agent_name,
            rationale="翻译任务不依赖外部工具。",
            boundary_type="translation",
        ),
    ]


def build_evalscope_call_decision_summary(records: List[EvalScopeToolCallDecisionRecord]) -> Dict[str, Any]:
    call_records = [record for record in records if bool(record.should_call_tool)]
    not_call_records = [record for record in records if not bool(record.should_call_tool)]
    tool_name_counts: Dict[str, int] = {}
    for record in call_records:
        tool_name = str(record.metadata.get("expected_tool_name") or "").strip()
        if not tool_name:
            continue
        tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
    return {
        "total_records": len(records),
        "call_records": len(call_records),
        "not_call_records": len(not_call_records),
        "tool_name_distribution": dict(sorted(tool_name_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an EvalScope-style single-step call/not-call benchmark dataset.")
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_NAME,
        help="Agent name used to scope dataset outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path for the benchmark dataset.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary JSON path.",
    )
    args = parser.parse_args()

    output_path = args.output or (eval_dataset_dir(args.agent, "evalscope") / "evalscope_call_decision_records_v1.jsonl")
    summary_output_path = args.summary_output or (evaluator_dir(args.agent, "evalscope") / "evalscope_call_decision_dataset_summary_v1.json")
    records = build_evalscope_call_decision_records(agent_name=args.agent)
    summary = build_evalscope_call_decision_summary(records)
    write_jsonl(output_path, records)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()