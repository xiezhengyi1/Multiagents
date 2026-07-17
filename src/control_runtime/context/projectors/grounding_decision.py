from __future__ import annotations

from typing import Any

from ...domain.policy_plan import GroundingDecision
from .base import BaseProjector, exclude, field, json_mapping, without_empty_values
from .flow_selector import FlowSelectorProjector


class GroundingDecisionProjector(BaseProjector):
    model = GroundingDecision
    visible = (
        field("mobility_intent"),
        field("slice_migration_authorization"),
    )
    excluded = (
        exclude("open_questions", reason="Handled by artifact contracts, not the OSA prompt"),
        exclude("grounding_evidence", reason="Traceability payload, too verbose for the OSA prompt"),
    )

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        projected = super().project(raw)
        flows = [
            FlowSelectorProjector.project(flow)
            for flow in (raw.get("flows") or [])
            if isinstance(flow, dict)
        ]
        if flows:
            projected["flows"] = flows
        constraints = [
            without_empty_values(dict(constraint))
            for constraint in (raw.get("qos_operation_constraints") or [])
            if isinstance(constraint, dict)
        ]
        if constraints:
            projected["qos_operation_constraints"] = constraints
        return without_empty_values(projected)


__all__ = ["GroundingDecisionProjector"]
