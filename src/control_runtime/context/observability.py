from __future__ import annotations

from typing import Any, Mapping

from shared.runtime import TokenCounter


def measure_context_components(
    components: Mapping[str, Any],
    *,
    token_counter: TokenCounter | None = None,
) -> dict[str, int]:
    """Return trace-only token estimates for prompt component attribution."""
    counter = token_counter or TokenCounter()
    measured = {
        str(name): counter.count(_component_text(value))
        for name, value in components.items()
        if value not in (None, "", [], {})
    }
    measured["measured_total"] = sum(measured.values())
    return measured


def _component_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return str(value)


__all__ = ["measure_context_components"]
