from __future__ import annotations

from ...domain.policy_plan import QosTargetEnvelope
from .base import BaseProjector, field


class QosTargetEnvelopeProjector(BaseProjector):
    model = QosTargetEnvelope
    visible = (
        field("flow_id"),
        field("app_id"),
        field("flow_name"),
        field("baseline_priority"),
        field("baseline_latency_ms"),
        field("baseline_jitter_ms"),
        field("baseline_packet_error_rate"),
        field("baseline_max_br_ul_mbps"),
        field("baseline_max_br_dl_mbps"),
        field("baseline_gbr_ul_mbps"),
        field("baseline_gbr_dl_mbps"),
        field("strictest_priority"),
        field("strictest_latency_ms"),
        field("strictest_jitter_ms"),
        field("strictest_packet_error_rate"),
        field("strictest_max_br_ul_mbps"),
        field("strictest_max_br_dl_mbps"),
        field("strictest_gbr_ul_mbps"),
        field("strictest_gbr_dl_mbps"),
        field("rationale"),
    )
