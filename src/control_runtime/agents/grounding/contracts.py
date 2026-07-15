from __future__ import annotations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    am_context_summary: Dict[str, Any] = Field(default_factory=dict)
    am_policy_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    subscription_summary: Dict[str, Any] = Field(default_factory=dict)
    catalog_evidence_observed: bool = Field(default=False, exclude=True)
    catalog_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    semantic_candidates: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    am_context_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    subscription_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)


__all__ = [
    "ExplicitFlowTarget",
    "FlowCandidateEvidence",
    "IntentEvidence",
]
