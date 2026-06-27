from __future__ import annotations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from ...domain.policy_plan import FlowSelector


class FlowCandidateEvidence(BaseModel):
    supi: str = ""
    app_id: str = ""
    app_name: Optional[str] = None
    flow_id: str = ""
    flow_name: str = ""
    service_type: Optional[str] = None
    service_type_id: Optional[int] = None
    score: float = 1.0


class ExplicitFlowTarget(BaseModel):
    flow_name: str = ""
    app_name: Optional[str] = None
    supi: Optional[str] = None


class IntentEvidence(BaseModel):
    user_input: str = ""
    supi: str = ""
    requested_domains: List[str] = Field(default_factory=list)
    retry_scope: str = ""
    explicit_app_id: str = ""
    explicit_app_name: str = ""
    explicit_flow_id: str = ""
    explicit_flow_name: str = ""
    explicit_flow_targets: List[ExplicitFlowTarget] = Field(default_factory=list)
    candidate_flows: List[FlowCandidateEvidence] = Field(default_factory=list)
    candidate_apps: List[Dict[str, Any]] = Field(default_factory=list)
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    main_requested_domains: List[str] = Field(default_factory=list)
    am_context_summary: Dict[str, Any] = Field(default_factory=dict)
    am_policy_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    catalog_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    semantic_candidates: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    am_context_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)


class IntentAdvisorDecision(BaseModel):
    mobility_intent: Dict[str, Any] = Field(default_factory=dict)
    grounded_requested_domains: List[str] = Field(default_factory=list)
    domain_revision_needed: bool = False
    domain_revision_rationale: str = ""
    domain_resolution: str = "confirmed"
    flows: List[FlowSelector] = Field(
        default_factory=list,
        description="QoS-relevant semantic flow selections produced by IEA.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_advisor_shapes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)

        mobility_intent = data.get("mobility_intent")
        am_policy_context = data.get("am_policy_context")
        if (not isinstance(mobility_intent, dict) or not mobility_intent) and isinstance(am_policy_context, dict):
            normalized_mobility = dict(am_policy_context)
            if "rfsp" in normalized_mobility and "current_rfsp" not in normalized_mobility:
                normalized_mobility["current_rfsp"] = normalized_mobility.get("rfsp")
            if "allowed_snssais" in normalized_mobility and "current_allowed_snssais" not in normalized_mobility:
                normalized_mobility["current_allowed_snssais"] = normalized_mobility.get("allowed_snssais")
            if "current_association_id" in normalized_mobility and "association_id" not in normalized_mobility:
                normalized_mobility["association_id"] = normalized_mobility.get("current_association_id")
            data["mobility_intent"] = normalized_mobility

        flows = data.get("flows")
        if isinstance(flows, list):
            data["flows"] = [cls._normalize_flow_shape(flow) for flow in flows]
        return data

    @staticmethod
    def _normalize_flow_shape(flow: Any) -> Any:
        if not isinstance(flow, dict):
            return flow
        flow_data = dict(flow)
        baseline = flow_data.get("sla_baseline")
        if not isinstance(baseline, dict):
            return flow_data
        baseline_to_flow = {
            "bandwidth_ul": "bw_ul",
            "bandwidth_dl": "bw_dl",
            "max_br_ul_mbps": "bw_ul",
            "max_br_dl_mbps": "bw_dl",
            "guaranteed_bandwidth_ul": "gbr_ul",
            "guaranteed_bandwidth_dl": "gbr_dl",
            "gbr_ul_mbps": "gbr_ul",
            "gbr_dl_mbps": "gbr_dl",
            "latency": "lat",
            "latency_ms": "lat",
            "loss_rate": "loss_req",
            "packet_error_rate": "loss_req",
            "jitter": "jitter_req",
            "jitter_ms": "jitter_req",
            "priority": "priority",
        }
        for source_key, target_key in baseline_to_flow.items():
            if source_key not in baseline:
                continue
            if flow_data.get(target_key) in (None, ""):
                flow_data[target_key] = baseline.get(source_key)
        return flow_data

    @model_validator(mode="after")
    def _validate_grounded_qos_flows(self) -> "IntentAdvisorDecision":
        for index, flow in enumerate(self.flows or []):
            resolution_status = str(flow.resolution_status or "").strip().lower()
            if resolution_status != "resolved":
                continue
            flow_id = str(flow.flow_id or "").strip()
            app_id = str(flow.app_id or "").strip()
            if not flow_id or not app_id:
                raise ValueError(
                    "resolved qos flow must include both flow_id and app_id "
                    f"(flows[{index}] has flow_id={flow_id or '<empty>'}, app_id={app_id or '<empty>'})"
                )
        return self


__all__ = [
    "ExplicitFlowTarget",
    "FlowCandidateEvidence",
    "IntentAdvisorDecision",
    "IntentEvidence",
]
