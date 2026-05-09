from __future__ import annotations

from typing import List

from ...domain.policy_plan import FlowSelector, QosTargetEnvelope


class QosEnvelopeBuilder:
    def build(
        self,
        *,
        flows: List[FlowSelector],
    ) -> List[QosTargetEnvelope]:
        if not flows:
            return []
        envelopes: List[QosTargetEnvelope] = []
        for flow in flows:
            flow_id = str(flow.flow_id or "").strip()
            if not flow_id or str(flow.resolution_status or "").strip().lower() != "resolved":
                continue
            envelopes.append(
                QosTargetEnvelope(
                    flow_id=flow_id,
                    app_id=str(flow.app_id or "").strip(),
                    flow_name=str(flow.name or flow_id).strip(),
                    baseline_priority=flow.priority,
                    baseline_latency_ms=flow.lat,
                    baseline_jitter_ms=flow.jitter_req,
                    baseline_packet_error_rate=flow.loss_req,
                    baseline_max_br_ul_mbps=flow.bw_ul,
                    baseline_max_br_dl_mbps=flow.bw_dl,
                    baseline_gbr_ul_mbps=flow.gbr_ul,
                    baseline_gbr_dl_mbps=flow.gbr_dl,
                    strictest_priority=flow.priority,
                    strictest_latency_ms=flow.lat,
                    strictest_jitter_ms=flow.jitter_req,
                    strictest_packet_error_rate=flow.loss_req,
                    strictest_max_br_ul_mbps=flow.bw_ul,
                    strictest_max_br_dl_mbps=flow.bw_dl,
                    strictest_gbr_ul_mbps=flow.gbr_ul,
                    strictest_gbr_dl_mbps=flow.gbr_dl,
                    rationale=[f"grounded_from_flow:{flow_id}"],
                )
            )
        return envelopes
