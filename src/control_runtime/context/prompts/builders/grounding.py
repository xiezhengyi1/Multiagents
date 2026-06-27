from __future__ import annotations

from .base import PromptBuilder


class GroundingPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..grounding import IEA_SYSTEM_PROMPT

        return IEA_SYSTEM_PROMPT

    def dynamic_rules(self) -> str:
        from ..grounding import IEA_DYNAMIC_RULES

        return IEA_DYNAMIC_RULES
