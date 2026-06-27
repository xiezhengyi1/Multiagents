from __future__ import annotations

from .base import PromptBuilder


class MainPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..main import MAIN_CONTROL_SYSTEM_PROMPT

        return MAIN_CONTROL_SYSTEM_PROMPT

    def dynamic_rules(self) -> str:
        from ..main import MAIN_CONTROL_DYNAMIC_RULES

        return MAIN_CONTROL_DYNAMIC_RULES
