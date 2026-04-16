from __future__ import annotations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from domain.policy_plan import FlowSelector


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
    explicit_app_id: str = ""
    explicit_flow_id: str = ""
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
        description="QoS-relevant flow decisions produced by IEA, including target SLA fields.",
    )


__all__ = ["IntentEvidence", "IntentAdvisorDecision", "FlowCandidateEvidence"]
