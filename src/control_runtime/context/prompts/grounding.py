from __future__ import annotations

from .engine import PromptEngine
from .knowledge_search import IEA_KNOWLEDGE_SEARCH_SKILL


def render_grounding_system_prompt() -> str:
    return PromptEngine().render(
        "grounding/system.j2",
        iea_knowledge_search_skill=IEA_KNOWLEDGE_SEARCH_SKILL,
    )


__all__ = ["render_grounding_system_prompt"]
