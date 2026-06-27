from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from ..budget import TokenBudget


class PromptEngine:
    """Jinja2 environment with package-local template loading."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader("control_runtime.context.prompts", "templates"),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: Any) -> str:
        template = self._env.get_template(template_name)
        return template.render(**context)

    def render_with_budget(
        self,
        template_name: str,
        budget: TokenBudget,
        **context: Any,
    ) -> str:
        rendered = self.render(template_name, **context)
        if budget.estimate(rendered) <= budget.limit:
            return rendered
        return rendered[: max(0, budget.limit * 4)]
