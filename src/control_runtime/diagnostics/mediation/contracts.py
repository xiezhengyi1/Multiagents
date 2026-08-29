from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ...contracts import ControlDomain


class DomainStatus(str, Enum):
    READY = "ready"
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    INCOMPLETE_CONTEXT = "incomplete_context"
    SKIPPED = "skipped"
    FAILED = "failed"


class DomainVerdict(BaseModel):
    domain: ControlDomain
    status: DomainStatus
    summary: str = ""
    hard_conflicts: List[str] = Field(default_factory=list)
    soft_conflicts: List[str] = Field(default_factory=list)
    infeasible_reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class RevisionRequest(BaseModel):
    target_domain: ControlDomain
    conflict_type: str
    target_policy_ids: List[str] = Field(default_factory=list)
    target_objects: List[str] = Field(default_factory=list)
    reason: str = ""
    suggested_actions: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class UnifiedConstraintSet(BaseModel):
    hard_constraints: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class MediatorDecision(BaseModel):
    status: str = Field(default="approved")
    reason_summary: str = Field(default="")
    affected_domains: List[ControlDomain] = Field(default_factory=list)
    revision_requests: List[RevisionRequest] = Field(default_factory=list)
    unified_constraints: UnifiedConstraintSet = Field(default_factory=UnifiedConstraintSet)


class ConflictResolutionRequest(BaseModel):
    candidate_policies: List[Dict[str, Any]] = Field(default_factory=list)
    resource_view: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    conflict_scope: Dict[str, Any] = Field(default_factory=dict)
    upstream_context: Dict[str, Any] = Field(default_factory=dict)


class ConflictResolutionResult(BaseModel):
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
