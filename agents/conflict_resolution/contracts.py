from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from domain.control_plane import MediatorDecision, UnifiedConstraintSet


class ConflictResolutionRequest(BaseModel):
    candidate_policies: List[Dict[str, Any]] = Field(default_factory=list)
    resource_view: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    conflict_scope: Dict[str, Any] = Field(default_factory=dict)
    upstream_context: Dict[str, Any] = Field(default_factory=dict)


class ConflictResolutionResult(BaseModel):
    status: str = Field(default="no_conflict")
    mediator_status: str = Field(default="approved")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    affected_policy_ids: List[str] = Field(default_factory=list)
    affected_objects: List[str] = Field(default_factory=list)
    affected_domains: List[str] = Field(default_factory=list)
    resolution_recommendations: List[str] = Field(default_factory=list)
    reason_summary: str = Field(default="")
    revision_requests: List[Dict[str, Any]] = Field(default_factory=list)
    unified_constraints: UnifiedConstraintSet = Field(default_factory=UnifiedConstraintSet)

    def to_mediator_decision(self) -> MediatorDecision:
        return MediatorDecision.model_validate(
            {
                "status": self.mediator_status,
                "reason_summary": self.reason_summary,
                "affected_domains": self.affected_domains,
                "revision_requests": self.revision_requests,
                "unified_constraints": self.unified_constraints.model_dump(mode="json"),
            }
        )
