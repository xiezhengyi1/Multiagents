from __future__ import annotations

from typing import Any, Dict

from .base import PromptBuilder


class PlanningPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL

        return self.render_template(
            "planning/system.j2",
            osa_knowledge_search_skill=OSA_KNOWLEDGE_SEARCH_SKILL,
        )

    def advisor_user_prompt(
        self,
        *,
        normalized_user_intent: Dict[str, Any],
        coordination_context: Dict[str, Any],
        planning_evidence: Dict[str, Any],
        available_tool_names: list[str] | None = None,
    ) -> str:
        from ..planning import OSA_DYNAMIC_RULES, OSA_OUTPUT_FORMAT_RULES, render_round_tool_policy

        return self.render_template(
            "planning/user.j2",
            normalized_user_intent=normalized_user_intent,
            coordination_context=coordination_context,
            planning_evidence=planning_evidence,
            tool_policy=render_round_tool_policy(available_tool_names),
            dynamic_rules=OSA_DYNAMIC_RULES.strip(),
            output_format_rules=OSA_OUTPUT_FORMAT_RULES.strip(),
        )

    def validation_retry_prompt(
        self,
        *,
        base_prompt: str,
        issues: list[str],
        cached_planning_evidence: Dict[str, Any] | None = None,
    ) -> str:
        from ..planning import build_validation_retry_prompt

        return build_validation_retry_prompt(
            base_prompt=base_prompt,
            issues=issues,
            cached_planning_evidence=cached_planning_evidence,
        )
