from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined

def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class PromptEngine:
    """Jinja2 environment with package-local template loading."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader("control_runtime.context.prompts", "templates"),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self._env.filters["json_dumps"] = _json_dumps

    def render(self, template_name: str, **context: Any) -> str:
        template = self._env.get_template(template_name)
        return template.render(**context)
