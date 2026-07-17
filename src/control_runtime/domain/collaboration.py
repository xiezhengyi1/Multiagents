from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from .control_plane import GlobalControlIntent
from .policy_plan import GroundingDecision


class SharedControlContext(BaseModel):
    """The Main decision shared with downstream stages without re-generation."""

    model_config = ConfigDict(extra="forbid")
    main_intent: GlobalControlIntent = Field(description="Canonical Main decision; Main is its sole producer")


class PlanningContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(default=1, description="Closed-loop round number")
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Bound planning snapshot identifier")
    memory_context: str = Field(default="", description="Retrieved memory context for the round")
    shared_context: SharedControlContext = Field(description="Typed Main-owned context shared by IEA, OSA, and tools")
    feedback_context: str = Field(default="", description="Aggregated feedback context from previous rounds")
    handoff_history: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured handoff history from previous rounds",
    )
    revision_requests: list[Dict[str, Any]] = Field(default_factory=list, description="Structured revision requests returned by Mediator")
    unified_constraints: Dict[str, Any] = Field(default_factory=dict, description="Structured hard constraints returned by Mediator")


class PlanningRequest(BaseModel):
    grounding_decision: GroundingDecision = Field(description="IEA bindings and constraints for Main's active stage")
    context: PlanningContext = Field(description="Collaboration context for downstream planning agents")


class AgentHandoff(BaseModel):
    round_index: int = Field(default=1, description="Closed-loop round number")
    source_agent: str = Field(default="", description="Upstream agent name")
    target_agent: str = Field(default="", description="Downstream agent name")
    artifact_type: str = Field(default="", description="Artifact type carried in the handoff")
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Snapshot identifier")
    summary: str = Field(default="", description="Human-readable handoff summary")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured handoff payload")


class CoordinationIssue(BaseModel):
    source_agent: str = Field(default="")
    issue_type: str = Field(default="")
    domain: str = Field(default="")
    binding_keys: list[str] = Field(default_factory=list)
    policy_objects: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    rationale: str = Field(default="")


class DomainNegotiationRequest(BaseModel):
    round_index: int = Field(default=1)
    source_agent: str = Field(default="intent_encoding")
    requires_domain_review: bool = False
    issues: list[CoordinationIssue] = Field(default_factory=list)
    recommended_consumers: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class PlanningBlockerReport(BaseModel):
    round_index: int = Field(default=1)
    source_agent: str = Field(default="optimization_strategy")
    planning_status: str = Field(default="needs_upstream_reground")
    missing_evidence: list[str] = Field(default_factory=list)
    blocked_targets: list[str] = Field(default_factory=list)
    upstream_requests: list[str] = Field(default_factory=list)
    planner_conflicts: list[str] = Field(default_factory=list)
    recommended_consumers: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class ExecutionReentryRequest(BaseModel):
    round_index: int = Field(default=1)
    source_agent: str = Field(default="policy_dispatch")
    recommended_consumers: list[str] = Field(default_factory=list)
    target_bindings_at_risk: list[str] = Field(default_factory=list)
    policy_objects_at_risk: list[str] = Field(default_factory=list)
    reason_by_domain: Dict[str, str] = Field(default_factory=dict)
    failure_scope: str = Field(default="none")
    failures: list[Dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
