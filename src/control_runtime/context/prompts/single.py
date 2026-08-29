from __future__ import annotations

from .engine import PromptEngine


def render_single_system_prompt() -> str:
    return PromptEngine().render("single/system.j2")
