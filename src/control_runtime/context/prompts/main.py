from __future__ import annotations

from .engine import PromptEngine


MAIN_CONTROL_DYNAMIC_RULES = """
Dynamic routing rules for this round:
- Read the coordinator context before deciding reuse or regrounding.
- Round 1 must route to `intent_encoding`.
- Retry rounds may route to `optimization_strategy` only when target bindings can be reused.
- If retry evidence says binding is wrong, prefer `intent_encoding`.
- If retry evidence says binding is stable but policy execution or assurance failed, prefer `optimization_strategy`.
"""


def _render_system_prompt() -> str:
    return PromptEngine().render("main/system.j2")


MAIN_CONTROL_SYSTEM_PROMPT = _render_system_prompt()
MAIN_CONTROL_CORE_PROMPT = MAIN_CONTROL_SYSTEM_PROMPT


__all__ = ["MAIN_CONTROL_CORE_PROMPT", "MAIN_CONTROL_DYNAMIC_RULES", "MAIN_CONTROL_SYSTEM_PROMPT"]
