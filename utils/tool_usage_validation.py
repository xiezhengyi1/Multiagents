from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Mapping


_THINK_TOOL_NAMES = {"think", "think_tool"}
_KNOWLEDGE_TOOL_NAMES = {"search_semantic_knowledge", "get_knowledge_by_key"}
_REQUIRED_TRACE_KEYS = {
    "agent_name",
    "input_messages",
    "message_trajectory",
    "tool_calls",
    "tool_results",
    "structured_response",
    "status",
}
_EXPLICIT_3GPP_OBJECT_NAMES = {
    "smpolicydecision",
    "smpolicycontextdata",
    "pccrule",
    "pccrules",
    "qosdata",
    "sessionrule",
    "traffic descriptor",
    "route selection descriptor",
    "ursp",
    "npcf_smpolicycontrol",
    "npcf_uepolicycontrol",
}


def _build_validation_report(*, is_valid: bool, trace_errors: List[str], tool_validations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构造统一的工具验证结果，兼容总体结论与逐工具明细。"""
    failed_tools = [
        {
            "tool_call_id": item.get("tool_call_id"),
            "tool_name": item.get("tool_name"),
            "issues": list(item.get("issues") or []),
        }
        for item in tool_validations
        if not bool(item.get("passed"))
    ]
    return {
        "is_valid": is_valid,
        "trace_errors": trace_errors,
        "tool_validations": tool_validations,
        "failed_tools": failed_tools,
    }


def _empty_report(trace_error: str) -> Dict[str, Any]:
    """返回仅包含trace级失败原因的空验证结果。"""
    return _build_validation_report(is_valid=False, trace_errors=[trace_error], tool_validations=[])


def _normalize_trace(trace: Mapping[str, Any]) -> Dict[str, Any] | None:
    """规范化并验证trace对象，确保包含所有必需键且结构正确。"""
    if not isinstance(trace, Mapping) or not _REQUIRED_TRACE_KEYS.issubset(trace.keys()):
        return None
    if not all(isinstance(trace.get(key), list) for key in ("input_messages", "message_trajectory", "tool_calls", "tool_results")):
        return None
    return dict(trace)


def _normalize_text(value: Any) -> str:
    """将值规范化为字符串（针对非字符串对象使用JSON序列化）。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_tool_name(name: Any) -> str:
    """规范化工具名称，去除空白字符。"""
    return str(name or "").strip()


def _normalize_tool_args(args: Any) -> str | None:
    """将工具参数序列化为统一格式的JSON字符串。"""
    if not isinstance(args, dict):
        return None
    return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_identifier(value: Any) -> str:
    """统一标识符比较格式。"""
    return str(value or "").strip().casefold()


def _normalize_lookup_token(value: Any) -> str:
    """将名称规整为便于模糊比较的token。"""
    return re.sub(r"[^a-z0-9_]+", "", str(value or "").strip().lower())


def _extract_request_text(trace: Mapping[str, Any]) -> str:
    """从trace的input_messages中提取所有用户请求文本。"""
    parts = [
        _normalize_text(message.get("content"))
        for message in trace.get("input_messages", [])
        if isinstance(message, Mapping) and str(message.get("role", message.get("type", ""))).strip().lower() in {"user", "human"}
    ]
    return "\n".join(filter(None, parts))


def _extract_explicit_3gpp_objects(request_text: str) -> List[str]:
    """提取请求中明确出现的3GPP对象名。"""
    lowered = request_text.lower()
    compact = _normalize_lookup_token(request_text)
    matches: List[str] = []
    for keyword in _EXPLICIT_3GPP_OBJECT_NAMES:
        if " " in keyword and keyword in lowered:
            matches.append(keyword)
        elif " " not in keyword and _normalize_lookup_token(keyword) in compact:
            matches.append(keyword)
    return matches


def _has_valid_message_trajectory(trace: Mapping[str, Any]) -> bool:
    """验证消息轨迹中每个元素的结构是否合法。"""
    return all(
        isinstance(message, Mapping) and {"role", "content", "step_index"}.issubset(message.keys())
        for message in trace.get("message_trajectory", [])
    )


def _collect_tool_result_issues(tool_calls: List[Any], tool_results: List[Any]) -> List[str]:
    """收集工具结果列表中的结构性问题，例如重复ID和孤儿结果。"""
    issues: List[str] = []
    call_ids = {
        str(call.get("id", "")).strip()
        for call in tool_calls
        if isinstance(call, Mapping) and str(call.get("id", "")).strip()
    }
    seen_result_ids: set[str] = set()

    for index, result in enumerate(tool_results):
        if not isinstance(result, Mapping):
            issues.append(f"tool_results[{index}] must be an object")
            continue

        result_call_id = str(result.get("tool_call_id", "")).strip()
        if not result_call_id:
            issues.append(f"tool_results[{index}] is missing tool_call_id")
            continue
        if result_call_id in seen_result_ids:
            issues.append(f"tool_results contains duplicate tool_call_id: {result_call_id}")
            continue

        seen_result_ids.add(result_call_id)
        if result_call_id not in call_ids:
            issues.append(f"tool_results contains orphan result for tool_call_id: {result_call_id}")

    return issues


def _extract_embedded_json(text: str) -> Mapping[str, Any] | None:
    """从带前缀的工具结果字符串中尽力提取JSON对象。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _extract_candidate_keys(result_text: str) -> List[str]:
    """从语义检索结果中提取候选ResultKey。"""
    keys: List[str] = []
    for line in result_text.splitlines():
        match = re.match(r"\s*\d+\.\s*([^\s|]+)", line)
        if match:
            keys.append(str(match.group(1) or "").strip())
    return keys


def _extract_structured_state(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """提取最终结构化结果中的关键信息，供硬规则核对。"""
    structured_response = trace.get("structured_response")
    structured = structured_response if isinstance(structured_response, Mapping) else {}
    app_names: set[str] = set()
    app_name = structured.get("app_name")
    if isinstance(app_name, str) and app_name.strip():
        app_names.add(_normalize_identifier(app_name))

    flow_names: set[str] = set()
    flows = structured.get("flows")
    if isinstance(flows, list):
        for flow in flows:
            if not isinstance(flow, Mapping):
                continue
            for key in ("flow_name", "name"):
                raw = flow.get(key)
                if isinstance(raw, str) and raw.strip():
                    flow_names.add(_normalize_identifier(raw))

    return {
        "supi": str(structured.get("supi") or "").strip(),
        "app_names": app_names,
        "flow_names": flow_names,
    }


def _add_issue(tool_validations: List[Dict[str, Any]], index: int, message: str) -> None:
    """向指定工具校验结果追加问题，并确保不重复。"""
    issues = list(tool_validations[index].get("issues") or [])
    if message not in issues:
        issues.append(message)
        tool_validations[index]["issues"] = issues
    tool_validations[index]["passed"] = False


def _append_trace_error(trace_errors: List[str], message: str) -> None:
    """追加trace级错误，避免重复。"""
    if message not in trace_errors:
        trace_errors.append(message)


def _append_missing_tool_validation(tool_validations: List[Dict[str, Any]], tool_name: str, message: str) -> None:
    """为缺失的必需工具追加伪失败项，便于最终汇总失败工具。"""
    tool_validations.append(
        {
            "tool_call_id": None,
            "tool_name": tool_name,
            "tool_kind": "action",
            "passed": False,
            "issues": [message],
        }
    )


def _validate_flow_catalog_result(action_call: Mapping[str, Any], structured_state: Mapping[str, Any]) -> List[str]:
    """校验flow catalog调用参数与最终结构化结果是否一致。"""
    issues: List[str] = []
    args = action_call.get("args") if isinstance(action_call.get("args"), Mapping) else {}
    result_text = _normalize_text(action_call.get("result_content"))
    call_supi = str(args.get("supi") or "").strip()
    response_supi = str(structured_state.get("supi") or "").strip()

    if response_supi and call_supi and _normalize_identifier(response_supi) != _normalize_identifier(call_supi):
        issues.append("flow catalog lookup supi does not match the final structured supi")
    if "UE Flow Catalog Query Failed" in result_text:
        issues.append("flow catalog lookup did not return a usable catalog")
        return issues

    payload = _extract_embedded_json(result_text)
    if not payload:
        return issues

    result_supi = str(payload.get("supi") or "").strip()
    if call_supi and result_supi and _normalize_identifier(call_supi) != _normalize_identifier(result_supi):
        issues.append("flow catalog result supi does not match the requested supi")

    app_catalog = payload.get("app_catalog") if isinstance(payload.get("app_catalog"), list) else []
    flow_catalog = payload.get("flow_catalog") if isinstance(payload.get("flow_catalog"), list) else []
    known_apps = {
        _normalize_identifier(item.get("app_name"))
        for item in app_catalog
        if isinstance(item, Mapping) and str(item.get("app_name") or "").strip()
    }
    known_flows = {
        _normalize_identifier(item.get("flow_name"))
        for item in flow_catalog
        if isinstance(item, Mapping) and str(item.get("flow_name") or "").strip()
    }

    expected_apps = set(structured_state.get("app_names") or set())
    if expected_apps and known_apps and not expected_apps.intersection(known_apps):
        issues.append("flow catalog result does not contain the final structured app")

    expected_flows = set(structured_state.get("flow_names") or set())
    if expected_flows and known_flows and not expected_flows.intersection(known_flows):
        issues.append("flow catalog result does not contain the final structured flow")

    return issues


def _validate_ue_context_result(action_call: Mapping[str, Any], structured_state: Mapping[str, Any]) -> List[str]:
    """校验UE context调用参数与最终结构化结果是否一致。"""
    issues: List[str] = []
    args = action_call.get("args") if isinstance(action_call.get("args"), Mapping) else {}
    result_text = _normalize_text(action_call.get("result_content"))
    call_supi = str(args.get("supi") or "").strip()
    response_supi = str(structured_state.get("supi") or "").strip()

    if response_supi and call_supi and _normalize_identifier(response_supi) != _normalize_identifier(call_supi):
        issues.append("UE context lookup supi does not match the final structured supi")
    if "UE Context Query Failed" in result_text or "UE Context Not Found" in result_text:
        issues.append("UE context lookup did not return a usable context")
        return issues

    payload = _extract_embedded_json(result_text)
    if payload:
        result_supi = str(payload.get("supi") or "").strip()
        if call_supi and result_supi and _normalize_identifier(call_supi) != _normalize_identifier(result_supi):
            issues.append("UE context result supi does not match the requested supi")
    return issues


def _validate_search_knowledge_result(action_call: Mapping[str, Any]) -> List[str]:
    """校验语义知识检索是否返回可继续精确检索的候选。"""
    issues: List[str] = []
    args = action_call.get("args") if isinstance(action_call.get("args"), Mapping) else {}
    query = str(args.get("query") or "").strip()
    result_text = _normalize_text(action_call.get("result_content"))
    if not query:
        issues.append("semantic knowledge search requires a non-empty query")
    if "Error executing knowledge search" in result_text or "No relevant knowledge found" in result_text:
        issues.append("semantic knowledge search did not return usable knowledge candidates")
    return issues


def _validate_flow_target_search_result(action_call: Mapping[str, Any], structured_state: Mapping[str, Any]) -> List[str]:
    """校验按 app/flow 名称的语义目标检索结果。"""
    issues: List[str] = []
    args = action_call.get("args") if isinstance(action_call.get("args"), Mapping) else {}
    app_name = str(args.get("app_name") or "").strip()
    flow_name = str(args.get("flow_name") or "").strip()
    if not app_name and not flow_name:
        issues.append("semantic flow target search requires app_name or flow_name")

    result_text = _normalize_text(action_call.get("result_content"))
    if "Semantic Flow Target Search Failed" in result_text:
        issues.append("semantic flow target search did not return a usable candidate list")
        return issues

    payload = _extract_embedded_json(result_text)
    if not payload:
        return issues

    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    if not candidates:
        issues.append("semantic flow target search did not return usable candidates")
        return issues

    known_apps = {
        _normalize_identifier(item.get("app_name"))
        for item in candidates
        if isinstance(item, Mapping) and str(item.get("app_name") or "").strip()
    }
    known_flows = {
        _normalize_identifier(item.get("flow_name"))
        for item in candidates
        if isinstance(item, Mapping) and str(item.get("flow_name") or "").strip()
    }
    known_supis = {
        str(item.get("supi") or "").strip()
        for item in candidates
        if isinstance(item, Mapping) and str(item.get("supi") or "").strip()
    }

    expected_apps = set(structured_state.get("app_names") or set())
    if expected_apps and known_apps and not expected_apps.intersection(known_apps):
        issues.append("semantic flow target search results do not contain the final structured app")

    expected_flows = set(structured_state.get("flow_names") or set())
    if expected_flows and known_flows and not expected_flows.intersection(known_flows):
        issues.append("semantic flow target search results do not contain the final structured flow")

    expected_supi = str(structured_state.get("supi") or "").strip()
    if expected_supi and known_supis and expected_supi not in known_supis:
        issues.append("semantic flow target search results do not contain the final structured supi")
    return issues


def _validate_exact_knowledge_result(action_call: Mapping[str, Any], explicit_objects: List[str]) -> List[str]:
    """校验精确知识检索key与结果。"""
    issues: List[str] = []
    args = action_call.get("args") if isinstance(action_call.get("args"), Mapping) else {}
    key = str(args.get("key") or "").strip()
    result_text = _normalize_text(action_call.get("result_content"))

    if not key:
        issues.append("exact knowledge lookup requires a non-empty key")
    if "Knowledge item not found" in result_text or "Error retrieving key" in result_text:
        issues.append("exact knowledge lookup did not return a usable knowledge item")

    if explicit_objects and key:
        normalized_key = _normalize_lookup_token(key)
        if not any(
            _normalize_lookup_token(item) in normalized_key or normalized_key in _normalize_lookup_token(item)
            for item in explicit_objects
        ):
            issues.append("exact knowledge lookup key does not match the requested 3GPP object")
    return issues


def _validate_user_clarification_result(action_call: Mapping[str, Any]) -> List[str]:
    """校验用户澄清工具是否真正拿到了可用回答。"""
    issues: List[str] = []
    args = action_call.get("args") if isinstance(action_call.get("args"), Mapping) else {}
    question = str(args.get("question") or "").strip()
    if not question:
        issues.append("user clarification requires a non-empty question")

    result_text = _normalize_text(action_call.get("result_content"))
    payload = _extract_embedded_json(result_text) or {}
    if not payload:
        issues.append("user clarification did not return a structured payload")
        return issues

    if str(payload.get("status") or "").strip().lower() != "success":
        issues.append("user clarification did not complete successfully")
    if not str(payload.get("normalized_response") or "").strip():
        issues.append("user clarification did not provide a usable response")
    return issues


def _merge_validation_reports(hard_report: Dict[str, Any], semantic_report: Dict[str, Any]) -> Dict[str, Any]:
    """合并硬规则和语义Judge的评测结果。"""
    merged_tool_validations = list(hard_report.get("tool_validations") or [])
    merged_trace_errors = list(hard_report.get("trace_errors") or [])

    for item in semantic_report.get("failed_tools") or []:
        reason = str(item.get("reason") or "semantic judge rejected the tool usage").strip()
        failure_type = str(item.get("failure_type") or "semantic_judge").strip()
        evidence = [str(entry) for entry in item.get("evidence") or []]
        issues = [f"{failure_type}: {reason}"]
        if evidence:
            issues.append("evidence: " + " | ".join(evidence))
        merged_tool_validations.append(
            {
                "tool_call_id": item.get("tool_call_id"),
                "tool_name": item.get("tool_name"),
                "tool_kind": "semantic_judge",
                "passed": False,
                "issues": issues,
            }
        )

    for message in semantic_report.get("trace_errors") or []:
        _append_trace_error(merged_trace_errors, str(message))

    merged = _build_validation_report(
        is_valid=bool(hard_report.get("is_valid")) and bool(semantic_report.get("is_valid")),
        trace_errors=merged_trace_errors,
        tool_validations=merged_tool_validations,
    )
    merged["hard_validation"] = hard_report
    merged["semantic_validation"] = semantic_report
    return merged


def _semantic_judge_is_enabled(explicit_flag: bool | None) -> bool:
    """判断是否启用语义Judge。"""
    if explicit_flag is not None:
        return explicit_flag
    raw = str(os.getenv("TOOL_USAGE_ENABLE_SEMANTIC_JUDGE", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def evaluate_intent_encoding_tool_usage(
    trace: Mapping[str, Any],
    *,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> Dict[str, Any]:
    """按工具调用逐项验证意图编码代理的工具使用，并可选合并语义Judge结果。"""
    normalized = _normalize_trace(trace)
    if not normalized:
        return _empty_report("trace is missing required keys or list fields")
    if str(normalized.get("status", "")).strip().lower() != "success":
        return _empty_report("trace status must be success")
    if not _has_valid_message_trajectory(normalized):
        return _empty_report("trace message_trajectory is invalid")

    request_text = _extract_request_text(normalized)
    if not request_text.strip():
        return _empty_report("trace is missing user request text")

    tool_calls = normalized["tool_calls"]
    tool_results = {
        str(result.get("tool_call_id", "")).strip(): result
        for result in normalized["tool_results"]
        if isinstance(result, Mapping)
    }
    trace_errors: List[str] = []
    trace_errors.extend(_collect_tool_result_issues(tool_calls, normalized["tool_results"]))
    tool_validations: List[Dict[str, Any]] = []
    action_calls: List[Dict[str, Any]] = []
    seen_non_think_calls: set[tuple[str, str]] = set()
    previous_name = ""
    structured_state = _extract_structured_state(normalized)
    explicit_objects = _extract_explicit_3gpp_objects(request_text)

    for index, call in enumerate(tool_calls):
        if not isinstance(call, Mapping):
            trace_errors.append(f"tool_calls[{index}] must be an object")
            continue

        call_id = str(call.get("id", "")).strip()
        name = _normalize_tool_name(call.get("name"))
        args = call.get("args")
        args_key = _normalize_tool_args(args)
        issues: List[str] = []

        if not call_id:
            issues.append("missing tool call id")
        if not name:
            issues.append("missing tool name")
        if args_key is None:
            issues.append("tool args must be a JSON object")

        result = tool_results.get(call_id) if call_id else None
        result_content = _normalize_text(result.get("content")) if isinstance(result, Mapping) else ""
        if not result:
            issues.append("missing matching tool result")
        else:
            if _normalize_tool_name(result.get("name")) != name:
                issues.append("tool result name does not match tool call")
            if str(result.get("status", "")).strip().lower() != "success":
                issues.append("tool result status must be success")

        if name in _THINK_TOOL_NAMES:
            if not isinstance(args, Mapping) or not str(args.get("message", "")).strip():
                issues.append("think tool requires a non-empty message")
        else:
            if index == 0 or previous_name not in _THINK_TOOL_NAMES:
                issues.append("non-think tool must be called immediately after a think tool")
            if args_key is not None:
                signature = (name, args_key)
                if signature in seen_non_think_calls:
                    issues.append("duplicate non-think tool call with identical arguments")
                else:
                    seen_non_think_calls.add(signature)

        tool_validations.append(
            {
                "tool_call_id": call_id,
                "tool_name": name,
                "tool_kind": "think" if name in _THINK_TOOL_NAMES else "action",
                "passed": not issues,
                "issues": issues,
            }
        )

        if name not in _THINK_TOOL_NAMES:
            action_calls.append(
                {
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "args": args if isinstance(args, Mapping) else {},
                    "result_content": result_content,
                    "validation_index": len(tool_validations) - 1,
                    "call_position": index,
                }
            )

        previous_name = name

    for action_call in action_calls:
        tool_name = str(action_call.get("tool_name") or "")
        validation_index = int(action_call.get("validation_index") or 0)
        if tool_name == "get_ue_flow_catalog":
            for message in _validate_flow_catalog_result(action_call, structured_state):
                _add_issue(tool_validations, validation_index, message)
        elif tool_name == "search_flow_targets_by_name":
            for message in _validate_flow_target_search_result(action_call, structured_state):
                _add_issue(tool_validations, validation_index, message)
        elif tool_name == "get_ue_context":
            for message in _validate_ue_context_result(action_call, structured_state):
                _add_issue(tool_validations, validation_index, message)
        elif tool_name == "search_semantic_knowledge":
            for message in _validate_search_knowledge_result(action_call):
                _add_issue(tool_validations, validation_index, message)
        elif tool_name == "get_knowledge_by_key":
            for message in _validate_exact_knowledge_result(action_call, explicit_objects):
                _add_issue(tool_validations, validation_index, message)
        elif tool_name == "ask_user_clarification":
            for message in _validate_user_clarification_result(action_call):
                _add_issue(tool_validations, validation_index, message)

    knowledge_calls = [item for item in action_calls if item.get("tool_name") in _KNOWLEDGE_TOOL_NAMES]
    search_calls = [item for item in action_calls if item.get("tool_name") == "search_semantic_knowledge"]
    if explicit_objects and not knowledge_calls:
        message = "request mentions explicit 3GPP objects but no knowledge tool was used"
        _append_trace_error(trace_errors, message)
        _append_missing_tool_validation(tool_validations, "get_knowledge_by_key", message)

    if explicit_objects:
        first_knowledge = next((item for item in action_calls if item.get("tool_name") in _KNOWLEDGE_TOOL_NAMES), None)
        if first_knowledge and first_knowledge.get("tool_name") != "get_knowledge_by_key":
            message = "explicit schema/object request should use get_knowledge_by_key before semantic search"
            _append_trace_error(trace_errors, message)
            _add_issue(tool_validations, int(first_knowledge.get("validation_index") or 0), message)

    for search_call in search_calls:
        next_action = next(
            (
                item
                for item in action_calls
                if int(item.get("call_position") or -1) > int(search_call.get("call_position") or -1)
            ),
            None,
        )
        if not next_action or next_action.get("tool_name") != "get_knowledge_by_key":
            message = "semantic knowledge search must be followed by get_knowledge_by_key for exact lookup"
            _append_trace_error(trace_errors, message)
            _add_issue(tool_validations, int(search_call.get("validation_index") or 0), message)
            continue

        candidate_keys = _extract_candidate_keys(_normalize_text(search_call.get("result_content")))
        exact_key = str((next_action.get("args") or {}).get("key") or "").strip()
        if candidate_keys and exact_key and exact_key not in candidate_keys:
            message = "exact knowledge lookup key was not selected from semantic search candidates"
            _append_trace_error(trace_errors, message)
            _add_issue(tool_validations, int(next_action.get("validation_index") or 0), message)

    hard_report = _build_validation_report(
        is_valid=not trace_errors and all(bool(item.get("passed")) for item in tool_validations),
        trace_errors=trace_errors,
        tool_validations=tool_validations,
    )

    if not _semantic_judge_is_enabled(enable_semantic_judge) and semantic_judge is None:
        return hard_report

    try:
        from .tool_usage_judge import evaluate_intent_encoding_semantic_tool_usage

        semantic_report = evaluate_intent_encoding_semantic_tool_usage(normalized, judge=semantic_judge)
    except Exception as exc:
        semantic_report = {
            "is_valid": False,
            "trace_errors": [f"semantic judge failed: {exc}"],
            "failed_tools": [],
            "summary": "semantic judge invocation failed",
            "dimension_scores": {},
        }

    return _merge_validation_reports(hard_report, semantic_report)


def validate_intent_encoding_tool_usage(
    trace: Mapping[str, Any],
    *,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> bool:
    """返回意图编码代理工具使用是否合法。"""
    return bool(
        evaluate_intent_encoding_tool_usage(
            trace,
            enable_semantic_judge=enable_semantic_judge,
            semantic_judge=semantic_judge,
        ).get("is_valid")
    )


_VALIDATORS: Dict[str, Callable[..., bool]] = {
    "intent_encoding": validate_intent_encoding_tool_usage,
}

_REPORT_VALIDATORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "intent_encoding": evaluate_intent_encoding_tool_usage,
}


def validate_tool_usage(
    trace: Mapping[str, Any],
    agent_name: str,
    *,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> bool:
    """根据代理名称路由到对应的工具使用验证函数。"""
    validator = _VALIDATORS.get(str(agent_name or "").strip())
    return (
        validator(trace, enable_semantic_judge=enable_semantic_judge, semantic_judge=semantic_judge)
        if validator
        else False
    )


def evaluate_tool_usage(
    trace: Mapping[str, Any],
    agent_name: str,
    *,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> Dict[str, Any]:
    """根据代理名称路由到对应的逐工具验证函数。"""
    validator = _REPORT_VALIDATORS.get(str(agent_name or "").strip())
    if not validator:
        return _empty_report(f"unsupported agent for tool usage validation: {str(agent_name or '').strip()}")
    return validator(trace, enable_semantic_judge=enable_semantic_judge, semantic_judge=semantic_judge)


__all__ = [
    "evaluate_intent_encoding_tool_usage",
    "evaluate_tool_usage",
    "validate_intent_encoding_tool_usage",
    "validate_tool_usage",
]