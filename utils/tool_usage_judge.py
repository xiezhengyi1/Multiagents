from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

from agents.BaseAgent import BaseAgent


INTENT_ENCODING_SEMANTIC_JUDGE_PROMPT = """
You are a strict evaluator for an agent's multi-step tool usage trace.

Your job is to judge tool-decision correctness rather than syntax-only correctness.
This evaluator is NOT a general grader for the final OperationIntent JSON quality.
Do not fail a trace solely because the final response is over-specified, aggressive, under-filled,
or imperfectly conservative if the tool decisions themselves were appropriate.

Scope boundary:
- Focus on whether tools were needed, which tool was chosen, when it was called, whether arguments
    were grounded, whether the tool result was used in a way that implies the tool choice was wrong,
    and whether there was redundancy.
- Do NOT treat `think` / `think_tool` or the final JSON output as business tools.
- If you notice non-tool issues in the final JSON (for example unsupported numeric targets,
    empty audit fields, or imperfect schema filling), record them as warnings, not as tool failures,
    unless they clearly prove a missing, wrong, or unnecessary tool call.

Focus on these dimensions:
1. need_tool: whether a tool was actually needed.
2. right_tool: whether the chosen tool was appropriate for the missing information.
3. right_timing: whether the tool was called at the correct moment.
4. args_grounded: whether tool arguments were grounded in user input or prior observations.
5. result_grounded: whether the tool results support the tool-level interpretation that led to another tool decision.
6. redundancy: whether the trace contains unnecessary tool calls.
7. final_consistency: only use this when the final response contradiction clearly implies a tool-decision mistake.

Scoring rules:
- Use 1.0 for fully correct, 0.5 for partially correct, 0.0 for incorrect.
- Only mark a failure when the evidence is concrete from the trace.
- Do not invent missing facts.
- If the trace is business-correct but underspecified, keep it valid and explain the uncertainty in summary.
- Set `is_valid=false` only for actual tool-usage failures.
- Use `warnings` for non-blocking output-quality concerns.
- `failed_tools` must reference actual external tools or missing external tools only.

Return only the structured JSON schema requested by the caller.
""".strip()


_PSEUDO_TOOL_NAMES = {"think", "think_tool", "none (final output)", "final output"}
_NON_BLOCKING_FAILURE_TYPES = {"result_grounded", "final_consistency"}
_TOOL_DECISION_DIMENSIONS = ("need_tool", "right_tool", "right_timing", "args_grounded", "redundancy")
_TOOL_DECISION_ERROR_MARKERS = (
        "missing tool",
        "wrong tool",
        "redundant",
        "unnecessary",
        "missing knowledge tool",
        "knowledge tool call",
        "should use",
        "should have used",
        "must use",
        "must be followed",
        "exact lookup",
)


class SemanticJudgeFailure(BaseModel):
    tool_call_id: str | None = None
    tool_name: str
    failure_type: str
    reason: str
    evidence: List[str] = Field(default_factory=list)


class IntentEncodingSemanticJudgeResult(BaseModel):
    is_valid: bool = True
    summary: str = ""
    trace_errors: List[str] = Field(default_factory=list)
    failed_tools: List[SemanticJudgeFailure] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)


def _normalize_tool_name(name: str | None) -> str:
    return str(name or "").strip().casefold()


def _looks_like_tool_decision_error(message: str) -> bool:
    normalized = str(message or "").strip().casefold()
    return any(marker in normalized for marker in _TOOL_DECISION_ERROR_MARKERS)


def _score_is_perfect(value: Any) -> bool:
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return False


def _normalize_semantic_judge_result(result: IntentEncodingSemanticJudgeResult) -> IntentEncodingSemanticJudgeResult:
    warnings = list(result.warnings)
    normalized_failures: List[SemanticJudgeFailure] = []
    pseudo_failures_seen = False

    for failure in result.failed_tools:
        if _normalize_tool_name(failure.tool_name) in _PSEUDO_TOOL_NAMES:
            pseudo_failures_seen = True
            warnings.append(f"{failure.tool_name}: {failure.reason}")
            continue
        normalized_failures.append(failure)

    decision_dims_perfect = all(
        _score_is_perfect(result.dimension_scores.get(name))
        for name in _TOOL_DECISION_DIMENSIONS
        if name in result.dimension_scores
    )
    tool_decision_trace_errors = [item for item in result.trace_errors if _looks_like_tool_decision_error(item)]
    non_blocking_failures_only = normalized_failures and all(
        failure.failure_type in _NON_BLOCKING_FAILURE_TYPES for failure in normalized_failures
    )
    pseudo_only_non_blocking = pseudo_failures_seen and not normalized_failures and decision_dims_perfect and not tool_decision_trace_errors

    if non_blocking_failures_only and decision_dims_perfect and not tool_decision_trace_errors:
        warnings.extend(result.trace_errors)
        warnings.extend(f"{failure.tool_name}: {failure.reason}" for failure in normalized_failures)
        normalized_failures = []
        trace_errors: List[str] = []
        is_valid = True
    elif pseudo_only_non_blocking:
        warnings.extend(result.trace_errors)
        trace_errors = []
        is_valid = True
    else:
        trace_errors = list(result.trace_errors)
        is_valid = bool(result.is_valid) and not normalized_failures and not trace_errors

    return IntentEncodingSemanticJudgeResult(
        is_valid=is_valid,
        summary=result.summary,
        trace_errors=trace_errors,
        failed_tools=normalized_failures,
        warnings=warnings,
        dimension_scores=dict(result.dimension_scores),
    )


def _normalize_trace_for_judge(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """压缩trace，只保留语义Judge需要的关键信息。"""
    return {
        "agent_name": trace.get("agent_name"),
        "input_messages": trace.get("input_messages"),
        "message_trajectory": trace.get("message_trajectory"),
        "tool_calls": trace.get("tool_calls"),
        "tool_results": trace.get("tool_results"),
        "structured_response": trace.get("structured_response"),
        "status": trace.get("status"),
    }


def build_intent_encoding_semantic_judge_messages(trace: Mapping[str, Any]) -> List[Dict[str, str]]:
    """构造语义Judge的输入消息。"""
    payload = json.dumps(_normalize_trace_for_judge(trace), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": INTENT_ENCODING_SEMANTIC_JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate whether this intent_encoding trace used tools correctly at the business level.\n\n"
                "Return a structured judgement with failed tools, concrete reasons, and dimension scores.\n\n"
                f"Trace JSON:\n{payload}"
            ),
        },
    ]


class ToolUsageSemanticJudge:
    """基于项目现有 ChatOpenAI 配置的语义Judge。"""

    def __init__(self, llm: Any = None, *, model_name: str = "qwen-plus", temperature: float = 0.0) -> None:
        self.llm = llm or BaseAgent(model_name=model_name, temperature=temperature).get_llm()

    def evaluate_intent_encoding(self, trace: Mapping[str, Any]) -> Dict[str, Any]:
        messages = build_intent_encoding_semantic_judge_messages(trace)
        runner = self.llm.with_structured_output(IntentEncodingSemanticJudgeResult)
        result = runner.invoke(messages)
        if isinstance(result, IntentEncodingSemanticJudgeResult):
            return _normalize_semantic_judge_result(result).model_dump(mode="json")
        if isinstance(result, str):
            parsed = IntentEncodingSemanticJudgeResult.model_validate_json(result)
            return _normalize_semantic_judge_result(parsed).model_dump(mode="json")
        parsed = IntentEncodingSemanticJudgeResult.model_validate(result)
        return _normalize_semantic_judge_result(parsed).model_dump(mode="json")


def evaluate_intent_encoding_semantic_tool_usage(trace: Mapping[str, Any], *, judge: Any = None) -> Dict[str, Any]:
    """执行意图编码轨迹的语义Judge评测。"""
    if judge is None:
        return ToolUsageSemanticJudge().evaluate_intent_encoding(trace)
    if hasattr(judge, "evaluate_intent_encoding"):
        result = judge.evaluate_intent_encoding(trace)
    elif callable(judge):
        result = judge(trace)
    else:
        raise TypeError("semantic judge must be callable or expose evaluate_intent_encoding()")

    if isinstance(result, IntentEncodingSemanticJudgeResult):
        return _normalize_semantic_judge_result(result).model_dump(mode="json")
    if isinstance(result, str):
        parsed = IntentEncodingSemanticJudgeResult.model_validate_json(result)
        return _normalize_semantic_judge_result(parsed).model_dump(mode="json")
    parsed = IntentEncodingSemanticJudgeResult.model_validate(result)
    return _normalize_semantic_judge_result(parsed).model_dump(mode="json")


__all__ = [
    "INTENT_ENCODING_SEMANTIC_JUDGE_PROMPT",
    "IntentEncodingSemanticJudgeResult",
    "SemanticJudgeFailure",
    "ToolUsageSemanticJudge",
    "build_intent_encoding_semantic_judge_messages",
    "evaluate_intent_encoding_semantic_tool_usage",
]