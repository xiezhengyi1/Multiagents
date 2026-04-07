from __future__ import annotations

import json
from typing import Any, Dict, List


ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


def normalize_role(message: Dict[str, Any]) -> str:
    raw_role = str(message.get("role") or message.get("type") or "").strip().lower()
    return ROLE_ALIASES.get(raw_role, raw_role or "user")


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False)
        normalized.append({"role": normalize_role(message), "content": text})
    return normalized


def format_tool_call(tool_call: Dict[str, Any]) -> str:
    name = str(tool_call.get("name") or "").strip()
    args = tool_call.get("args")
    return f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>'


def format_tool_result(tool_result: Dict[str, Any]) -> str:
    content = tool_result.get("content")
    if isinstance(content, str):
        rendered = content
    else:
        rendered = json.dumps(content, ensure_ascii=False)
    return f"<tool_result>{rendered}</tool_result>"
