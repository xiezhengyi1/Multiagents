from __future__ import annotations

import json
import sys
from typing import Any, Callable, List

from langchain.tools import ToolRuntime, tool

from agents.tools.wrapper_think import tool_with_reason

from agent_runtime.core.context import AgentRuntimeContext


def _normalize_options(options: List[str] | None) -> List[str]:
    normalized: List[str] = []
    for option in options or []:
        text = str(option or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _stdin_supports_interaction() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def collect_user_clarification(
    question: str,
    options: List[str] | None = None,
    *,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[..., None] | None = None,
) -> dict[str, Any]:
    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ValueError("question is required")

    normalized_options = _normalize_options(options)
    input_reader = input if input_func is None else input_func
    output_writer = print if output_func is None else output_func

    # 关键步骤：用稳定的命令行格式向用户展示澄清问题与候选项。
    output_writer("\n[IEA Clarification]")
    output_writer(normalized_question)
    if normalized_options:
        output_writer("Options:")
        for index, option in enumerate(normalized_options, start=1):
            output_writer(f"{index}. {option}")

    prompt = "Enter option number or free text: " if normalized_options else "Enter your answer: "
    user_response = str(input_reader(prompt) or "").strip()
    if not user_response:
        raise ValueError("user clarification response is empty")

    response_payload: dict[str, Any] = {
        "status": "success",
        "question": normalized_question,
        "options": normalized_options,
        "user_response": user_response,
        "response_mode": "free_text",
        "normalized_response": user_response,
        "selected_option_index": None,
        "selected_option": None,
    }

    # 关键步骤：数字输入映射到选项文本，其余输入原样返回给 IEA。
    if normalized_options and user_response.isdigit():
        selected_index = int(user_response)
        if 1 <= selected_index <= len(normalized_options):
            selected_option = normalized_options[selected_index - 1]
            response_payload.update(
                {
                    "response_mode": "option",
                    "normalized_response": selected_option,
                    "selected_option_index": selected_index,
                    "selected_option": selected_option,
                }
            )

    return response_payload


@tool_with_reason
def ask_user_clarification(
    question: str,
    options: List[str] | None = None,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Ask the user for missing details through the command line.

    Provide a concise question. Optionally pass a short list of candidate answers.
    The user may answer with an option number or with free text.
    Returns a JSON string containing the original response and a normalized response.
    """
    try:
        if not str(question or "").strip():
            raise ValueError("question is required")
        runtime_context = runtime.context if runtime is not None else None
        if runtime_context is not None and not bool(getattr(runtime_context, "allow_user_interaction", False)):
            raise RuntimeError("interactive clarification is disabled for this runtime")
        if not _stdin_supports_interaction():
            raise RuntimeError("interactive stdin is unavailable for ask_user_clarification")
        payload = collect_user_clarification(question=question, options=options)
    except Exception as exc:
        payload = {
            "status": "failed",
            "question": str(question or "").strip(),
            "options": _normalize_options(options),
            "error": str(exc),
        }

    if runtime is not None and runtime.context is not None:
        payload["runtime"] = {
            "agent_name": runtime.context.agent_name,
            "session_id": runtime.context.session_id,
            "snapshot_id": runtime.context.snapshot_id,
            "allow_user_interaction": bool(getattr(runtime.context, "allow_user_interaction", False)),
        }

    return json.dumps(payload, ensure_ascii=False)


__all__ = ["ask_user_clarification", "collect_user_clarification"]
