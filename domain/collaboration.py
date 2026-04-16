from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .policy_plan import OperationIntent


class PlanningContext(BaseModel):
    round_index: int = Field(default=1, description="Closed-loop round number")
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Bound planning snapshot identifier")
    snapshot_metadata: Dict[str, Any] = Field(default_factory=dict, description="Snapshot metadata bound to this round")
    memory_context: str = Field(default="", description="Retrieved memory context for the round")
    feedback_context: str = Field(default="", description="Aggregated feedback context from previous rounds")
    handoff_history: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured handoff history from previous rounds",
    )
    active_domains: list[str] = Field(default_factory=list, description="Domains selected by Main Agent")
    main_agent_guidance: str = Field(default="", description="Round-specific guidance injected by Main Agent")
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
