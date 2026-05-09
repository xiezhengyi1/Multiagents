from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator

from .control_plane import (
    AgentConflict,
    AgentContribution,
    ControlSemantics,
    HandoffRecord,
    OpenQuestion,
)


def _json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class FlowSelector(BaseModel):
    supi: str = Field(default="", description="UE identifier")
    app_id: str = Field(default="", description="Application identifier")
    app_name: Optional[str] = Field(default=None, description="Application name")
    flow_id: Optional[str] = Field(default=None, description="Flow identifier")
    target_type: str = Field(default="flow", description="Target scope")
    name: str = Field(default="", description="Flow name")
    service_type: Optional[str] = Field(default=None, description="Service type")
    service_type_id: Optional[int] = Field(default=None, description="Service type identifier")
    bw_ul: Optional[float] = Field(default=None, description="Requested uplink bandwidth in Mbps")
    bw_dl: Optional[float] = Field(default=None, description="Requested downlink bandwidth in Mbps")
    gbr_ul: Optional[float] = Field(default=None, description="Guaranteed uplink bitrate in Mbps")
    gbr_dl: Optional[float] = Field(default=None, description="Guaranteed downlink bitrate in Mbps")
    lat: Optional[float] = Field(default=None, description="Latency requirement in ms")
    loss_req: Optional[float] = Field(default=None, description="Packet loss requirement")
    jitter_req: Optional[float] = Field(default=None, description="Jitter requirement in ms")
    priority: Optional[int] = Field(default=None, description="Priority level")
    description: Optional[str] = Field(default=None, description="Human-readable flow description")
    five_tuple: Optional[List[Any]] = Field(default=None, description="Resolved five tuple")
    current_bw_ul: Optional[float] = Field(default=None, description="Current uplink bandwidth in Mbps")
    current_bw_dl: Optional[float] = Field(default=None, description="Current downlink bandwidth in Mbps")
    resolution_status: str = Field(default="resolved", description="Resolution status")
    resolution_candidates: List[str] = Field(default_factory=list, description="Resolution candidates")


class GroundingEvidenceBundle(BaseModel):
    grounded_supi: str = Field(default="", description="Authoritative SUPI after grounding")
    grounded_apps: List[Dict[str, Any]] = Field(default_factory=list, description="Grounded candidate or selected applications")
    grounded_flows: List[Dict[str, Any]] = Field(default_factory=list, description="Grounded candidate or selected flows")
    grounded_mobility_targets: Dict[str, Any] = Field(default_factory=dict, description="Grounded mobility-target summary")
    evidence_sources: Dict[str, List[str]] = Field(default_factory=dict, description="Evidence grouped by source")


class QosTargetEnvelope(BaseModel):
    flow_id: str = Field(default="", description="Grounded flow identifier")
    app_id: str = Field(default="", description="Grounded application identifier")
    flow_name: str = Field(default="", description="Grounded flow name")
    baseline_priority: Optional[int] = Field(default=None, description="Grounded priority baseline")
    baseline_latency_ms: Optional[float] = Field(default=None, description="Grounded latency baseline in ms")
    baseline_jitter_ms: Optional[float] = Field(default=None, description="Grounded jitter baseline in ms")
    baseline_packet_error_rate: Optional[float] = Field(default=None, description="Grounded packet error rate baseline")
    baseline_max_br_ul_mbps: Optional[float] = Field(default=None, description="Grounded uplink MBR baseline in Mbps")
    baseline_max_br_dl_mbps: Optional[float] = Field(default=None, description="Grounded downlink MBR baseline in Mbps")
    baseline_gbr_ul_mbps: Optional[float] = Field(default=None, description="Grounded uplink GBR baseline in Mbps")
    baseline_gbr_dl_mbps: Optional[float] = Field(default=None, description="Grounded downlink GBR baseline in Mbps")
    strictest_priority: Optional[int] = Field(default=None, description="Smallest priority number OSA may request")
    strictest_latency_ms: Optional[float] = Field(default=None, description="Smallest latency target OSA may request")
    strictest_jitter_ms: Optional[float] = Field(default=None, description="Smallest jitter target OSA may request")
    strictest_packet_error_rate: Optional[float] = Field(default=None, description="Smallest packet error rate OSA may request")
    strictest_max_br_ul_mbps: Optional[float] = Field(default=None, description="Largest uplink MBR OSA may request")
    strictest_max_br_dl_mbps: Optional[float] = Field(default=None, description="Largest downlink MBR OSA may request")
    strictest_gbr_ul_mbps: Optional[float] = Field(default=None, description="Largest uplink GBR OSA may request")
    strictest_gbr_dl_mbps: Optional[float] = Field(default=None, description="Largest downlink GBR OSA may request")
    rationale: List[str] = Field(default_factory=list, description="Deterministic reasons for the envelope")


class OperationIntent(BaseModel):
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Planning snapshot identifier")
    supi: str = Field(default="", description="UE identifier")
    app_id: Optional[str] = Field(default="", description="Application identifier")
    app_name: Optional[str] = Field(default=None, description="Application name")
    operation_type: str = Field(default="modify", description="Requested operation type")
    urgency: str = Field(default="Normal", description="Requested urgency")
    raw_input: str = Field(default="", description="Original user input")
    resolution_status: str = Field(default="", description="Top-level resolution status")
    requested_domains: List[str] = Field(default_factory=list, description="Requested control domains inferred from intent")
    main_requested_domains: List[str] = Field(default_factory=list, description="Domain set explicitly requested by Main")
    grounded_requested_domains: List[str] = Field(default_factory=list, description="Domain set confirmed or revised by IEA")
    domain_revision_needed: bool = Field(default=False, description="Whether IEA revised or could not confirm Main's domain boundary")
    domain_revision_rationale: str = Field(default="", description="Why IEA revised or could not confirm the domain boundary")
    domain_resolution: str = Field(default="confirmed", description="confirmed, narrowed, widened, or cannot_confirm")
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict, description="Evidence supporting each domain decision")
    control_semantics: ControlSemantics = Field(default_factory=ControlSemantics, description="Structured staged control semantics derived from the user intent")
    mobility_intent: Dict[str, Any] = Field(default_factory=dict, description="Mobility / AM policy goals extracted from the user request")
    grounding_evidence: GroundingEvidenceBundle = Field(default_factory=GroundingEvidenceBundle, description="Structured grounding evidence carried forward for traceability")
    flows: List[FlowSelector] = Field(default_factory=list, description="Resolved flow selectors")
    qos_target_envelopes: List[QosTargetEnvelope] = Field(default_factory=list, description="IEA-owned QoS target envelopes derived from grounded baselines")
    agent_contributions: List[AgentContribution] = Field(default_factory=list, description="Structured multi-agent contribution trace")
    agent_conflicts: List[AgentConflict] = Field(default_factory=list, description="Structured multi-agent conflict trace")
    handoff_records: List[HandoffRecord] = Field(default_factory=list, description="Structured handoff trace")
    open_questions: List[OpenQuestion] = Field(default_factory=list, description="Structured unresolved questions")

    @field_validator("domain_evidence", mode="before")
    @classmethod
    def _normalize_domain_evidence(cls, value: Any) -> Dict[str, List[str]]:
        if isinstance(value, dict):
            normalized: Dict[str, List[str]] = {}
            for key, items in value.items():
                if isinstance(items, list):
                    normalized[str(key)] = [str(item) for item in items if str(item or "").strip()]
                elif items not in (None, "", [], {}):
                    normalized[str(key)] = [str(items)]
            return normalized
        if isinstance(value, list):
            items = [str(item) for item in value if str(item or "").strip()]
            return {"general": items} if items else {}
        if value in (None, "", {}, []):
            return {}
        return {"general": [str(value)]}

    @field_validator("mobility_intent", mode="before")
    @classmethod
    def _normalize_mobility_intent(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}


class PolicyDraft(BaseModel):
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    supi: str = Field(default="", description="User SUPI")
    app_id: str = Field(default="", description="Application ID")
    flow_id: Optional[str] = Field(default=None, description="Flow ID")
    target_type: str = Field(default="flow", description="Target scope")
    policy_id: str = Field(default="", description="Unique policy ID")
    policy_type: str = Field(..., description="Policy type such as SmPolicyDecision, UrspRuleRequest, or PcfAmPolicyControlPolicyAssociation")
    resource_keys: List[str] = Field(default_factory=list, description="Normalized resource claims such as selected S-NSSAI or DNN bindings")
    policy_details: Dict[str, Any] = Field(default_factory=dict, description="Raw policy details")

    @field_serializer("policy_details", when_used="always")
    def _serialize_policy_details(self, value: Dict[str, Any]) -> Any:
        return _json_friendly(value)


class PlanningRationale(BaseModel):
    selected_strategy_profile: str = Field(default="", description="Selected strategy or heuristic profile")
    decisive_evidence: List[str] = Field(default_factory=list, description="Evidence items that determined the chosen plan")
    active_constraints: List[str] = Field(default_factory=list, description="Constraints active in the chosen plan")
    explanation: str = Field(default="", description="Planner explanation for the chosen plan")
    rejected_alternatives: List[str] = Field(default_factory=list, description="Alternatives explicitly not selected")
    main_constraints: List[str] = Field(default_factory=list, description="Constraints handed down by Main")
    iea_grounding_basis: List[str] = Field(default_factory=list, description="Grounding basis inherited from IEA")
    osa_decision_basis: List[str] = Field(default_factory=list, description="Planner basis for the current decision")
    unresolved_gaps: List[str] = Field(default_factory=list, description="Unresolved gaps that block full execution")


class PolicyPlanDraft(BaseModel):
    supi: str = Field(default="", description="User SUPI")
    session_id: str = Field(default="", description="Session identifier for deterministic execution")
    snapshot_id: str = Field(default="", description="Snapshot identifier for deterministic execution")
    planning_status: str = Field(default="executable_plan", description="executable_plan, partial_plan, or needs_upstream_reground")
    planning_basis: Dict[str, Any] = Field(default_factory=dict, description="Structured planning basis and planner-owned reasoning inputs")
    constraint_sources: Dict[str, Any] = Field(default_factory=dict, description="Structured constraint sources inherited from Main and Mediator")
    optimizer_result: Dict[str, Any] = Field(default_factory=dict, description="Structured optimizer output and cross-domain verdicts")
    execution_writeback: Dict[str, Any] = Field(default_factory=dict, description="Structured execution writeback patch information")
    planning_rationale: PlanningRationale = Field(default_factory=PlanningRationale, description="Structured rationale for the selected plan")
    all_policies: List[PolicyDraft] = Field(default_factory=list, description="All generated policy drafts")
    partial_policies: List[PolicyDraft] = Field(default_factory=list, description="Partially grounded policy drafts that are not yet executable")
    missing_evidence: List[str] = Field(default_factory=list, description="Evidence missing for full planning")
    blocked_targets: List[str] = Field(default_factory=list, description="Targets blocked by missing evidence or conflicts")
    upstream_requests: List[str] = Field(default_factory=list, description="Structured requests that must be sent upstream")
    planner_conflicts: List[str] = Field(default_factory=list, description="Planner-side conflicts or infeasibility notes")
    agent_contributions: List[AgentContribution] = Field(default_factory=list, description="Structured multi-agent contribution trace")
    agent_conflicts: List[AgentConflict] = Field(default_factory=list, description="Structured multi-agent conflict trace")
    handoff_records: List[HandoffRecord] = Field(default_factory=list, description="Structured handoff trace")
    open_questions: List[OpenQuestion] = Field(default_factory=list, description="Structured unresolved questions")


class PolicyPlan(BaseModel):
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Execution snapshot identifier")
    supi: str = Field(default="", description="UE identifier")
    policies: List[Dict[str, Any]] = Field(default_factory=list, description="Compiled policies")


class AssuranceVerdict(BaseModel):
    policy_id: str = Field(default="", description="Policy identifier")
    flow_id: Optional[str] = Field(default=None, description="Flow identifier")
    status: str = Field(default="unknown", description="satisfied, violated, skipped, or failed")
    reason: str = Field(default="", description="Explanation of the verdict")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Observed metrics")
