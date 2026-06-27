from __future__ import annotations

from .base import PromptBuilder


class DispatchPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        return (
            "You are the Policy Dispatch execution component. "
            "Compile policies, dispatch them to PCF, run assurance checks, and return a deterministic FeedbackReport."
        )
