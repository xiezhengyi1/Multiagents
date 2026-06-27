from __future__ import annotations

from .engine import PromptEngine
from .knowledge_search import IEA_KNOWLEDGE_SEARCH_SKILL


IEA_DYNAMIC_RULES = """
Dynamic grounding rules for this round:
- Treat Main's routing and retry scope from the user prompt as binding guidance.
- Preserve stable artifacts only when the evidence still supports them.
- Use cached evidence directly when it already grounds the answer.
- Call tools only when a required target is still ambiguous.
"""


def _render_system_prompt() -> str:
    return PromptEngine().render(
        "grounding/system.j2",
        iea_knowledge_search_skill=IEA_KNOWLEDGE_SEARCH_SKILL,
    )


IEA_SYSTEM_PROMPT = _render_system_prompt()
IEA_CORE_PROMPT = IEA_SYSTEM_PROMPT


__all__ = ["IEA_CORE_PROMPT", "IEA_DYNAMIC_RULES", "IEA_SYSTEM_PROMPT"]
