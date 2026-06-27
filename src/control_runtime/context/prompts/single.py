from __future__ import annotations

from .engine import PromptEngine


def _render_system_prompt() -> str:
    return PromptEngine().render("single/system.j2")


SINGLE_AGENT_ROUND_PROMPT = _render_system_prompt()


__all__ = ["SINGLE_AGENT_ROUND_PROMPT"]
