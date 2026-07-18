from __future__ import annotations

from .engine import PromptEngine
from .knowledge_search import IEA_KNOWLEDGE_SEARCH_SKILL


IEA_DYNAMIC_RULES = """
Round discipline: use cached evidence; call a tool only for a required missing fact;
preserve stable bindings unless fresh evidence contradicts them.
"""


def _render_system_prompt() -> str:
    return PromptEngine().render(
        "grounding/system.j2",
        iea_knowledge_search_skill=IEA_KNOWLEDGE_SEARCH_SKILL,
    )


IEA_SYSTEM_PROMPT = _render_system_prompt()
IEA_CORE_PROMPT = IEA_SYSTEM_PROMPT


__all__ = ["IEA_CORE_PROMPT", "IEA_DYNAMIC_RULES", "IEA_SYSTEM_PROMPT"]
