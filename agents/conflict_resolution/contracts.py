from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ConflictResolutionRequest(BaseModel):
    candidate_policies: List[Dict[str, Any]] = Field(default_factory=list)
    resource_view: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    conflict_scope: Dict[str, Any] = Field(default_factory=dict)
    upstream_context: Dict[str, Any] = Field(default_factory=dict)


class ConflictResolutionResult(BaseModel):
    status: str = Field(default="no_conflict")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    affected_policy_ids: List[str] = Field(default_factory=list)
    affected_objects: List[str] = Field(default_factory=list)
    resolution_recommendations: List[str] = Field(default_factory=list)
    reason_summary: str = Field(default="")
