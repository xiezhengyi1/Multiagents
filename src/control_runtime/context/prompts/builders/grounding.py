from __future__ import annotations

from .base import PromptBuilder


class GroundingPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..knowledge_search import IEA_KNOWLEDGE_SEARCH_SKILL

        return self.render_template(
            "grounding/system.j2",
            iea_knowledge_search_skill=IEA_KNOWLEDGE_SEARCH_SKILL,
        )

    def dynamic_rules(self) -> str:
        from ..grounding import IEA_DYNAMIC_RULES

        return IEA_DYNAMIC_RULES
