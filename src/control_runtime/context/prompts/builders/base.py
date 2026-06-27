from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..engine import PromptEngine


@dataclass
class PromptBuilder:
    engine: PromptEngine = field(default_factory=PromptEngine)

    def render_template(self, template_name: str, **context: Any) -> str:
        return self.engine.render(template_name, **context)

    def system_prompt(self) -> str:
        raise NotImplementedError
