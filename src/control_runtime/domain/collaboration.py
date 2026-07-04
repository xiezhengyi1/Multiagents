from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .policy_plan import OperationIntent


class SharedControlContext(BaseModel):
    """Typed blackboard shared across agents for user-level control goals."""

    raw_user_input: str = Field(default="", description="Original user request preserved across agent boundaries")
    main_control_semantics: Dict[str, Any] = Field(default_factory=dict, description="Main-owned staged control semantics")
    operation_constraints: list[Dict[str, Any]] = Field(default_factory=list, description="Hard operation constraints inferred by Main")
    shared_facts: Dict[str, Any] = Field(default_factory=dict, description="Compact cross-agent facts safe for every downstream agent")


class PlanningContext(BaseModel):
    round_index: int = Field(default=1, description="Closed-loop round number")
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Bound planning snapshot identifier")
    snapshot_metadata: Dict[str, Any] = Field(default_factory=dict, description="Snapshot metadata bound to this round")
    memory_context: str = Field(default="", description="Retrieved memory context for the round")
    shared_context: SharedControlContext = Field(default_factory=SharedControlContext, description="Typed cross-agent context shared by Main, IEA, OSA, and tools")
    feedback_context: str = Field(default="", description="Aggregated feedback context from previous rounds")
    handoff_history: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured handoff history from previous rounds",
    )
    active_domains: list[str] = Field(default_factory=list, description="Domains selected by Main Agent")
    main_round_strategy: str = Field(default="", description="High-level round strategy chosen by Main Agent")
    main_retry_scope: str = Field(default="", description="Main-selected retry scope for this round")
    main_investigation_targets: list[str] = Field(default_factory=list, description="What Main wants downstream to inspect or re-check")
    main_uncertainty_flags: list[str] = Field(default_factory=list, description="High-level uncertainty markers carried from Main")
    main_routing_decision: str = Field(default="", description="Explicit routing decision declared by Main")
    main_routing_rationale: str = Field(default="", description="Why Main selected this downstream path")
    main_reuse_contract: Dict[str, Any] = Field(default_factory=dict, description="Explicit reuse contract declared by Main")
    objective_profile: Dict[str, Any] = Field(default_factory=dict, description="Semantic optimization profile")
    forbidden_assumptions: list[str] = Field(default_factory=list, description="Assumptions subagents must not make")
    required_evidence: list[str] = Field(default_factory=list, description="Evidence subagents must collect")
    revision_requests: list[Dict[str, Any]] = Field(default_factory=list, description="Structured revision requests returned by Mediator")
    unified_constraints: Dict[str, Any] = Field(default_factory=dict, description="Structured hard constraints returned by Mediator")


class PlanningRequest(BaseModel):
    operation_intent: OperationIntent = Field(description="Resolved operation intent produced by IEA")
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
    main_requested_domains: list[str] = Field(default_factory=list)
    grounded_requested_domains: list[str] = Field(default_factory=list)
    domain_resolution: str = Field(default="cannot_confirm")
    domain_revision_needed: bool = False
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
