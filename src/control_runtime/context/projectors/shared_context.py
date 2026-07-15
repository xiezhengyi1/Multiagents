from __future__ import annotations

from typing import Any

from .base import json_mapping, without_empty_values


class SharedControlContextProjector:
    """Project only canonical, universally required intent into LLM context."""

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        initial_intent = without_empty_values(json_mapping(raw.get("initial_intent")))
        return {"initial_intent": initial_intent} if initial_intent else {}


__all__ = ["SharedControlContextProjector"]
