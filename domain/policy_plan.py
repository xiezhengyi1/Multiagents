from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer


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


class OperationIntent(BaseModel):
    session_id: str = Field(default="", description="Session identifier")
    snapshot_id: str = Field(default="", description="Planning snapshot identifier")
    supi: str = Field(default="", description="UE identifier")
    app_id: str = Field(default="", description="Application identifier")
    app_name: Optional[str] = Field(default=None, description="Application name")
    operation_type: str = Field(default="modify", description="Requested operation type")
    urgency: str = Field(default="Normal", description="Requested urgency")
    raw_input: str = Field(default="", description="Original user input")
    raw_intent_summary: str = Field(default="", description="Structured intent summary")
    resolution_status: str = Field(default="", description="Top-level resolution status")
    flows: List[FlowSelector] = Field(default_factory=list, description="Resolved flow selectors")


class PolicyDraft(BaseModel):
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    supi: str = Field(default="", description="User SUPI")
    app_id: str = Field(default="", description="Application ID")
    flow_id: Optional[str] = Field(default=None, description="Flow ID")
    target_type: str = Field(default="flow", description="Target scope")
    policy_id: str = Field(default="", description="Unique policy ID")
    policy_type: str = Field(..., description="Policy type")
    policy_details: Dict[str, Any] = Field(default_factory=dict, description="Raw policy details")

    @field_serializer("policy_details", when_used="always")
    def _serialize_policy_details(self, value: Dict[str, Any]) -> Any:
        return _json_friendly(value)


class PolicyPlanDraft(BaseModel):
    supi: str = Field(default="", description="User SUPI")
    session_id: str = Field(default="", description="Session identifier for deterministic execution")
    snapshot_id: str = Field(default="", description="Snapshot identifier for deterministic execution")
    all_policies: List[PolicyDraft] = Field(default_factory=list, description="All generated policy drafts")


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
