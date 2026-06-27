from __future__ import annotations

from ...domain.policy_plan import FlowSelector
from .base import BaseProjector, exclude, field


class FlowSelectorProjector(BaseProjector):
    """Project FlowSelector fields visible to LLM context."""

    model = FlowSelector
    visible = (
        field("supi", doc="UE identifier"),
        field("app_id", doc="Application identifier"),
        field("app_name", doc="Application name"),
        field("flow_id", doc="Flow identifier"),
        field("name", doc="Flow name"),
        field("service_type", doc="Service type string"),
        field("service_type_id", doc="Service type numeric ID"),
        field("bw_ul", doc="Requested UL bandwidth Mbps"),
        field("bw_dl", doc="Requested DL bandwidth Mbps"),
        field("gbr_ul", doc="Guaranteed UL bitrate Mbps"),
        field("gbr_dl", doc="Guaranteed DL bitrate Mbps"),
        field("lat", doc="Latency requirement ms"),
        field("loss_req", doc="Packet loss requirement"),
        field("jitter_req", doc="Jitter requirement ms"),
        field("priority", doc="Priority level"),
        field("five_tuple", doc="Resolved 5-tuple"),
        field("resolution_status", doc="Resolution status"),
        field("target_type", doc="Target scope"),
    )
    excluded = (
        exclude("description", reason="Redundant with name in prompt context"),
        exclude("current_bw_ul", reason="Runtime measurement, not a target contract"),
        exclude("current_bw_dl", reason="Runtime measurement, not a target contract"),
    )
