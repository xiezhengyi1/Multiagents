from __future__ import annotations

from ...domain.control_plane import GlobalControlIntent
from .base import BaseProjector, field


class GlobalControlIntentProjector(BaseProjector):
    model = GlobalControlIntent
    visible = (
        field("session_id"),
        field("snapshot_id"),
        field("supi"),
        field("round_strategy"),
        field("next_agent"),
        field("requested_domains"),
        field("domain_evidence"),
        field("control_semantics"),
        field("objective_profile"),
        field("investigation_targets"),
        field("uncertainty_flags"),
        field("retry_scope"),
        field("required_evidence"),
        field("forbidden_assumptions"),
        field("intent_encoding_guidance"),
        field("routing_decision"),
        field("routing_rationale"),
        field("reuse_contract"),
    )
