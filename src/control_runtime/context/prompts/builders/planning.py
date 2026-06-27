from __future__ import annotations

from typing import Any, Dict

from .base import PromptBuilder


class PlanningPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..planning import OSA_SYSTEM_PROMPT

        return OSA_SYSTEM_PROMPT

    def advisor_user_prompt(
        self,
        *,
        normalized_user_intent: Dict[str, Any],
        coordination_context: Dict[str, Any],
        planning_evidence: Dict[str, Any],
        available_tool_names: list[str] | None = None,
    ) -> str:
        from ..planning import build_advisor_user_prompt

        return build_advisor_user_prompt(
            normalized_user_intent=normalized_user_intent,
            coordination_context=coordination_context,
            planning_evidence=planning_evidence,
            available_tool_names=available_tool_names,
        )
