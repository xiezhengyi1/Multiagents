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


class IntentEvidence(BaseModel):
    user_input: str = ""
    supi: str = ""
    requested_domains: List[str] = Field(default_factory=list)
    retry_scope: str = ""
    explicit_app_id: str = ""
    explicit_app_name: str = ""
    explicit_flow_id: str = ""
    explicit_flow_name: str = ""
    candidate_flows: List[FlowCandidateEvidence] = Field(default_factory=list)
    candidate_apps: List[Dict[str, Any]] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    cache_hits: List[str] = Field(default_factory=list)
    operation_type_hint: str = "modify"
    mobility_intent_hint: Dict[str, Any] = Field(default_factory=dict)
    objective_profile_hint: str = ""
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    am_context_summary: Dict[str, Any] = Field(default_factory=dict)
    am_policy_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    # 缓存 catalog，避免 compile 阶段重复查询
    cached_catalog: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    cached_semantic_candidates: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    cached_am_context: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    cached_am_policy_candidates: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)


class IntentAdvisorDecision(BaseModel):
    selected_app_id: str = ""
    selected_flow_id: str = ""
    operation_type: str = "modify"
    raw_intent_summary: str = ""
    rationale: str = ""
    mobility_intent: Dict[str, Any] = Field(default_factory=dict)
    objective_profile_hint: str = ""
    flows: List[FlowSelector] = Field(
        default_factory=list,
        description="QoS-relevant semantic flow selections produced by IEA.",
    )

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


__all__ = ["IntentEvidence", "IntentAdvisorDecision", "FlowCandidateEvidence"]
