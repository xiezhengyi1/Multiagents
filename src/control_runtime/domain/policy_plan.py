from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from .control_plane import (
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
    service_type: Optional[str] = Field(default=None, description="Grounded service type from the UE flow catalog")
    service_type_id: Optional[int] = Field(default=None, description="Grounded service type identifier from the UE flow catalog")
    bw_ul: Optional[float] = Field(default=None, description="Grounded baseline uplink max bitrate in Mbps")
    bw_dl: Optional[float] = Field(default=None, description="Grounded baseline downlink max bitrate in Mbps")
    gbr_ul: Optional[float] = Field(default=None, description="Grounded baseline uplink guaranteed bitrate in Mbps")
    gbr_dl: Optional[float] = Field(default=None, description="Grounded baseline downlink guaranteed bitrate in Mbps")
    lat: Optional[float] = Field(default=None, description="Grounded baseline latency requirement in ms")
    loss_req: Optional[float] = Field(default=None, description="Grounded baseline packet loss requirement")
    jitter_req: Optional[float] = Field(default=None, description="Grounded baseline jitter requirement in ms")
    priority: Optional[int] = Field(default=None, description="Grounded baseline priority level")
    description: Optional[str] = Field(default=None, description="Human-readable flow description")
    five_tuple: Optional[List[Any]] = Field(default=None, description="Resolved five tuple")
    current_slice_snssai: Optional[str] = Field(default=None, description="Grounded current serving slice S-NSSAI")
    current_bw_ul: Optional[float] = Field(default=None, description="Grounded current uplink bandwidth in Mbps")
    current_bw_dl: Optional[float] = Field(default=None, description="Grounded current downlink bandwidth in Mbps")
    resolution_status: str = Field(default="resolved", description="Resolution status")


class GroundingEvidenceBundle(BaseModel):
    grounded_supi: str = Field(default="", description="Authoritative SUPI after grounding")
    grounded_apps: List[Dict[str, Any]] = Field(default_factory=list, description="Grounded candidate or selected applications")
    grounded_flows: List[Dict[str, Any]] = Field(default_factory=list, description="Grounded candidate or selected flows")
    grounded_mobility_targets: Dict[str, Any] = Field(default_factory=dict, description="Grounded mobility-target summary")
    evidence_sources: Dict[str, List[str]] = Field(default_factory=dict, description="Evidence grouped by source")


class QosOperationConstraint(BaseModel):
    flow_id: str = Field(default="", description="Grounded flow identifier")
    app_id: str = Field(default="", description="Grounded application identifier")
    operation_type: str = Field(default="", description="QoS operation type such as slice_migration or qos_reallocation")
    require_slice_change: bool = Field(default=False, description="True when satisfying the user goal requires selecting a different slice")
    source_slice_snssai: Optional[str] = Field(default=None, description="Current/source slice that must be treated as the migration origin")
    excluded_slice_snssais: List[str] = Field(default_factory=list, description="Slices the optimizer must not select for this flow")
    target_slice_preference: str = Field(default="", description="Preference such as lower_latency or higher_throughput")
    no_op_allowed: bool = Field(default=True, description="False when keeping the same slice cannot satisfy the operation")
    rationale: List[str] = Field(default_factory=list, description="Why this constraint is required")

    @model_validator(mode="after")
    def _derive_excluded_source(self) -> "QosOperationConstraint":
        source = str(self.source_slice_snssai or "").strip()
        existing = [str(item or "").strip() for item in self.excluded_slice_snssais if str(item or "").strip()]
        if self.require_slice_change and source and source not in existing:
            existing.append(source)
        self.excluded_slice_snssais = existing
        return self


class SliceMigrationAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(default="not_applicable", description="LLM-selected migration authorization decision")
    authority: str = Field(default="", description="Authoritative subscription evidence source")
    authorized_snssais: List[str] = Field(default_factory=list, description="Authorized S-NSSAI keys from the entitlement source")
    target_snssais: List[str] = Field(default_factory=list, description="Known requested target S-NSSAI keys")
    subscription_change_required: bool = Field(default=False, description="Whether external subscription provisioning is required")
    rationale: List[str] = Field(default_factory=list, description="Evidence-backed decision rationale")


class GroundingDecision(BaseModel):
    """IEA output: facts and operation constraints for Main's active stage only.

    Main owns user scope, requested domains, stage ordering, and semantic
    targets.  The IEA never repeats or edits those decisions; it returns only
    bindings and constraints that are needed by OSA for this stage.
    """

    model_config = ConfigDict(extra="forbid")

    flows: List[FlowSelector] = Field(default_factory=list, description="Resolved flow selectors")
    mobility_intent: Dict[str, Any] = Field(default_factory=dict, description="Grounded mobility / AM-policy state needed for this stage")
    grounding_evidence: GroundingEvidenceBundle = Field(default_factory=GroundingEvidenceBundle, description="Grounding provenance needed for planning and audit")
    qos_operation_constraints: List[QosOperationConstraint] = Field(default_factory=list, description="IEA-owned hard QoS constraints for grounded flows")
    slice_migration_authorization: SliceMigrationAuthorization = Field(default_factory=SliceMigrationAuthorization, description="IEA-selected authorization state for a requested slice migration")
    open_questions: List[OpenQuestion] = Field(default_factory=list, description="Structured unresolved questions")

    @model_validator(mode="after")
    def _validate_grounded_qos_flows(self) -> "GroundingDecision":
        for index, flow in enumerate(self.flows or []):
            resolution_status = str(flow.resolution_status or "").strip().lower()
            if resolution_status != "resolved":
                continue
            flow_id = str(flow.flow_id or "").strip()
            app_id = str(flow.app_id or "").strip()
            if not flow_id or not app_id:
                raise ValueError(
                    "resolved qos flow must include both flow_id and app_id "
                    f"(flows[{index}] has flow_id={flow_id or '<empty>'}, app_id={app_id or '<empty>'})"
                )
        return self

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
    optimizer_result: Dict[str, Any] = Field(default_factory=dict, description="Structured optimizer output and cross-domain verdicts")
    planning_rationale: PlanningRationale = Field(default_factory=PlanningRationale, description="Structured rationale for the selected plan")
    all_policies: List[PolicyDraft] = Field(default_factory=list, description="All generated policy drafts")
    partial_policies: List[PolicyDraft] = Field(default_factory=list, description="Partially grounded policy drafts that are not yet executable")
    missing_evidence: List[str] = Field(default_factory=list, description="Evidence missing for full planning")
    blocked_targets: List[str] = Field(default_factory=list, description="Targets blocked by missing evidence or conflicts")
    upstream_requests: List[str] = Field(default_factory=list, description="Structured requests that must be sent upstream")
    planner_conflicts: List[str] = Field(default_factory=list, description="Planner-side conflicts or infeasibility notes")
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
