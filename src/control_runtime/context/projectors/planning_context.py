from __future__ import annotations

from typing import Any

from ...domain.collaboration import PlanningContext
from .base import BaseProjector, field, json_mapping, without_empty_values


class PlanningContextProjector(BaseProjector):
    model = PlanningContext
    visible = (
        field("round_index"),
        field("session_id"),
        field("snapshot_id"),
        field("snapshot_metadata"),
        field("memory_context"),
        field("feedback_context"),
        field("active_domains"),
        field("main_round_strategy"),
        field("main_retry_scope"),
        field("main_investigation_targets"),
        field("main_uncertainty_flags"),
        field("main_routing_decision"),
        field("main_routing_rationale"),
        field("main_reuse_contract"),
    )

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        projected = super().project(raw)
        handoffs = [
            item
            for item in (
                _project_handoff_summary(handoff)
                for handoff in (raw.get("handoff_history") or [])[-2:]
            )
            if item
        ]
        if handoffs:
            projected["handoff_history"] = handoffs
        return projected


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
