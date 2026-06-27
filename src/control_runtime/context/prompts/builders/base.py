from __future__ import annotations

from dataclasses import dataclass, field

from ..engine import PromptEngine


@dataclass
class PromptBuilder:
    engine: PromptEngine = field(default_factory=PromptEngine)

    def system_prompt(self) -> str:
        raise NotImplementedError
