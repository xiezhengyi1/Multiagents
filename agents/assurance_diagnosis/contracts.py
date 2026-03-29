from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AssuranceDiagnosisRequest(BaseModel):
    execution_feedback: Dict[str, Any] = Field(default_factory=dict)
    dispatch_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    assurance_verdicts: List[Dict[str, Any]] = Field(default_factory=list)
    telemetry_snapshot: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    upstream_context: Dict[str, Any] = Field(default_factory=dict)


class AssuranceDiagnosisResult(BaseModel):
    status: str = Field(default="insufficient_evidence")
    root_cause_category: str = Field(default="")
    root_cause: str = Field(default="")
    affected_policy_ids: List[str] = Field(default_factory=list)
    affected_flow_ids: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    reason_summary: str = Field(default="")
