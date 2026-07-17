from __future__ import annotations

from typing import Any

from .base import json_mapping
from .global_intent import GlobalControlIntentProjector


class SharedControlContextProjector:
    """Project only canonical, universally required intent into LLM context."""

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        main_intent = raw.get("main_intent")
        return (
            {"main_intent": GlobalControlIntentProjector.project(main_intent)}
            if main_intent is not None
            else {}
        )


__all__ = ["SharedControlContextProjector"]
