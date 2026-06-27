from __future__ import annotations

from .base import PromptBuilder


class SinglePromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        return self.render_template("single/system.j2")
