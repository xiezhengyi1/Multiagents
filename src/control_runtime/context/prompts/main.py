from __future__ import annotations

from .engine import PromptEngine


def render_main_system_prompt() -> str:
    return PromptEngine().render("main/system.j2")


def build_main_user_prompt(*, user_input: str, coordinator_context: str) -> str:
    return (
        f"User input:\n{user_input}\n\n"
        f"Coordinator context:\n{coordinator_context or 'N/A'}\n\n"
        "Dynamic routing rules for this round:\n"
        "- Read the coordinator context before deciding reuse or regrounding.\n"
        "- Round 1 must route to `intent_encoding`.\n"
        "- Retry rounds may route to `optimization_strategy` only when target bindings can be reused.\n"
        "- If retry evidence says binding is wrong, prefer `intent_encoding`.\n"
        "- If retry evidence says binding is stable but policy execution or assurance failed, prefer `optimization_strategy`.\n\n"
        "Resolve the staged semantic plan, round-level routing, retry scope, and explicit SUPI already present in the request. "
        "On a retry routed to intent_encoding, provide explicit intent_encoding_guidance; otherwise leave it empty unless needed."
    )


__all__ = ["build_main_user_prompt", "render_main_system_prompt"]
