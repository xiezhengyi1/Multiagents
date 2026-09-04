from __future__ import annotations

from typing import Any

from .engine import PromptEngine
from .knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL


def render_planning_system_prompt() -> str:
    return PromptEngine().render(
        "planning/system.j2",
        osa_knowledge_search_skill=OSA_KNOWLEDGE_SEARCH_SKILL,
    )


def build_planning_user_prompt(
    *,
    normalized_user_intent: dict[str, Any],
    coordination_context: dict[str, Any],
    planning_evidence: dict[str, Any],
    available_tool_names: list[str],
) -> str:
    return PromptEngine().render(
        "planning/user.j2",
        normalized_user_intent=normalized_user_intent,
        coordination_context=coordination_context,
        planning_evidence=planning_evidence,
        tool_policy=_render_tool_manifest(available_tool_names),
    )


def _render_tool_manifest(available_tool_names: list[str]) -> str:
    lines = ["Callable tools in this round:"]
    lines.extend(
        (f"- `{name}`" for name in available_tool_names)
        if available_tool_names
        else ["- <none>"]
    )
    return "\n".join(lines)


__all__ = ["build_planning_user_prompt", "render_planning_system_prompt"]
