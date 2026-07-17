from __future__ import annotations

from typing import Any

from ...domain.collaboration import PlanningContext
from .base import BaseProjector, field, json_mapping, without_empty_values
from .shared_context import SharedControlContextProjector


_MEMORY_CONTEXT_CHAR_LIMIT = 1200
_FEEDBACK_CONTEXT_CHAR_LIMIT = 1800


class PlanningContextProjector(BaseProjector):
    model = PlanningContext
    visible = (
        field("round_index"),
        field("shared_context"),
    )
    nested = {"shared_context": SharedControlContextProjector}

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        projected = super().project(raw)
        if _has_retry_delta(raw):
            retry_delta = without_empty_values(
                {
                    "memory_summary": _compact_head(raw.get("memory_context"), _MEMORY_CONTEXT_CHAR_LIMIT),
                    "feedback_summary": _compact_tail(raw.get("feedback_context"), _FEEDBACK_CONTEXT_CHAR_LIMIT),
                    "latest_handoff": _latest_handoff(raw.get("handoff_history")),
                    "revision_requests": raw.get("revision_requests"),
                    "unified_constraints": raw.get("unified_constraints"),
                }
            )
            if retry_delta:
                projected["retry_delta"] = retry_delta
        return projected


def _has_retry_delta(raw: dict[str, Any]) -> bool:
    return int(raw.get("round_index") or 1) > 1


def _compact_head(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [memory summary truncated]"


def _compact_tail(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return "... [older feedback omitted]\n" + text[-limit:].lstrip()


def _latest_handoff(value: Any) -> dict[str, Any]:
    handoffs = list(value or []) if isinstance(value, list) else []
    if not handoffs:
        return {}
    return _project_handoff_summary(handoffs[-1])


def _project_handoff_summary(handoff: Any) -> dict[str, Any]:
    raw = json_mapping(handoff)
    summary = without_empty_values(
        {
            "round_index": raw.get("round_index"),
            "summary": raw.get("summary"),
        }
    )
    for source_key, summary_key in (
        ("diagnosis", "diagnosis_summary"),
        ("planning_blocker", "planning_blocker_summary"),
        ("execution_reentry", "execution_reentry_summary"),
        ("negotiation_request", "negotiation_summary"),
    ):
        nested = json_mapping(raw.get(source_key))
        text = str(
            nested.get("summary")
            or nested.get("reason_summary")
            or nested.get("root_cause")
            or ""
        ).strip()
        if text:
            summary[summary_key] = text
    return summary
