from __future__ import annotations

from typing import Any

from ...domain.policy_plan import OperationIntent
from .base import BaseProjector, exclude, field, json_mapping, without_empty_values
from .flow_selector import FlowSelectorProjector
from .qos_envelope import QosTargetEnvelopeProjector


class OperationIntentProjector(BaseProjector):
    model = OperationIntent
    visible = (
        field("session_id"),
        field("snapshot_id"),
        field("supi"),
        field("app_id"),
        field("app_name"),
        field("raw_input"),
        field("requested_domains"),
        field("grounded_requested_domains"),
        field("domain_resolution"),
        field("domain_revision_needed"),
        field("domain_revision_rationale"),
        field("control_semantics"),
        field("mobility_intent"),
    )
    excluded = (
        exclude("open_questions", reason="Handled by artifact contracts, not LLM prompt projection"),
        exclude("grounding_evidence", reason="Traceability payload, too verbose for OSA prompt"),
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
        envelopes = [
            QosTargetEnvelopeProjector.project(envelope)
            for envelope in (raw.get("qos_target_envelopes") or [])
            if isinstance(envelope, dict)
        ]
        if envelopes:
            projected["qos_target_envelopes"] = envelopes
        return without_empty_values(projected)
