from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from shared.contracts import ContractModel, ExecutionScope, JsonValue, NonEmptyStr

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


class DomainVerdict(ContractModel):
    domain: ControlDomain
    status: DomainStatus
    summary: NonEmptyStr
    hard_conflicts: list[NonEmptyStr]
    soft_conflicts: list[NonEmptyStr]
    infeasible_reasons: list[NonEmptyStr]
    metrics: dict[str, JsonValue]


class RevisionRequest(ContractModel):
    target_domain: ControlDomain
    conflict_type: NonEmptyStr
    target_policy_ids: list[NonEmptyStr]
    target_objects: list[NonEmptyStr]
    reason: NonEmptyStr
    suggested_actions: list[NonEmptyStr]
    hard_constraints: list[NonEmptyStr]
    evidence: list[NonEmptyStr]


class UnifiedConstraintSet(ContractModel):
    hard_constraints: list[NonEmptyStr]
    evidence: list[NonEmptyStr]


MediatorStatus = Literal["approved", "revise", "reject"]


class MediatorDecision(ContractModel):
    status: MediatorStatus
    reason_summary: NonEmptyStr
    affected_domains: list[ControlDomain]
    revision_requests: list[RevisionRequest]
    unified_constraints: UnifiedConstraintSet


ConflictType = Literal[
    "planner_blocking_conflict",
    "duplicate_policy",
    "binding_conflict",
    "unknown_resource_reference",
    "exclusive_resource_conflict",
    "slice_capacity_conflict",
    "target_subset_allowed_violation",
    "snssai_alignment_conflict",
    "ambr_qos_uplink_inconsistency",
    "ambr_qos_downlink_inconsistency",
    "service_area_missing_location_context",
]
ConflictResolutionStatus = Literal["resolved", "unresolved"]


class PolicyConflict(ContractModel):
    type: ConflictType
    domain: ControlDomain | None
    domains: list[ControlDomain]
    policy_ids: list[NonEmptyStr]
    objects: list[NonEmptyStr]
    resolution_status: ConflictResolutionStatus
    detail: NonEmptyStr

    @field_validator("domains", "policy_ids", "objects")
    @classmethod
    def validate_unique_lists(cls, values: list[object]) -> list[object]:
        if len(values) != len(set(values)):
            raise ValueError("conflict list values must be unique")
        return values


class SliceCapacity(ContractModel):
    total_ul: float
    total_dl: float
    guaranteed_ul: float
    guaranteed_dl: float
    current_ul: float
    current_dl: float
    remaining_ul: float
    remaining_dl: float


class FlowAllocation(ContractModel):
    current_slice_snssai: NonEmptyStr
    allocated_bandwidth_ul: float = Field(ge=0.0)
    allocated_bandwidth_dl: float = Field(ge=0.0)


class ConflictResourceView(ContractModel):
    valid_slice_keys: list[NonEmptyStr]
    valid_snssai_keys: list[NonEmptyStr]
    slice_capacity_by_snssai: dict[NonEmptyStr, SliceCapacity]
    flow_allocations: dict[NonEmptyStr, FlowAllocation]
    exclusive_resource_keys: list[NonEmptyStr]


class ConflictScope(ContractModel):
    domains: list[ControlDomain]

    @field_validator("domains")
    @classmethod
    def validate_unique_domains(cls, values: list[ControlDomain]) -> list[ControlDomain]:
        if len(values) != len(set(values)):
            raise ValueError("conflict scope domains must be unique")
        return values


class ConflictUpstreamContext(ContractModel):
    ue_context: dict[str, JsonValue]
    planner_cross_domain_verdicts: list[DomainVerdict]


class ConflictResolutionRequest(ContractModel):
    scope: ExecutionScope
    candidate_policies: list[dict[str, JsonValue]]
    resource_view: ConflictResourceView
    conflict_scope: ConflictScope
    upstream_context: ConflictUpstreamContext

    @model_validator(mode="after")
    def validate_scope(self) -> "ConflictResolutionRequest":
        if self.scope.snapshot_id is None:
            raise ValueError("conflict resolution requires scope.snapshot_id")
        return self


class ConflictResolutionResult(ContractModel):
    mediator_status: MediatorStatus
    conflicts: list[PolicyConflict]
    affected_policy_ids: list[NonEmptyStr]
    affected_objects: list[NonEmptyStr]
    affected_domains: list[ControlDomain]
    resolution_recommendations: list[NonEmptyStr]
    reason_summary: NonEmptyStr
    revision_requests: list[RevisionRequest]
    unified_constraints: UnifiedConstraintSet

    def to_mediator_decision(self) -> MediatorDecision:
        return MediatorDecision(
            status=self.mediator_status,
            reason_summary=self.reason_summary,
            affected_domains=self.affected_domains,
            revision_requests=self.revision_requests,
            unified_constraints=self.unified_constraints,
        )


__all__ = [
    "ConflictResolutionRequest",
    "ConflictResolutionResult",
    "ConflictResourceView",
    "ConflictScope",
    "ConflictType",
    "ConflictUpstreamContext",
    "DomainStatus",
    "DomainVerdict",
    "FlowAllocation",
    "MediatorDecision",
    "MediatorStatus",
    "PolicyConflict",
    "RevisionRequest",
    "SliceCapacity",
    "UnifiedConstraintSet",
]
