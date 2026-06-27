from __future__ import annotations

from .base import PromptBuilder


class MainPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        return self.render_template("main/system.j2")

    def dynamic_rules(self) -> str:
        from ..main import MAIN_CONTROL_DYNAMIC_RULES

        return MAIN_CONTROL_DYNAMIC_RULES
