from __future__ import annotations

from .base import PromptBuilder


class SinglePromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..single import SINGLE_AGENT_ROUND_PROMPT

        return SINGLE_AGENT_ROUND_PROMPT
