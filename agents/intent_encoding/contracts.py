from domain.policy_plan import FlowSelector, OperationIntent
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class FlowIntent(BaseModel):
    name: str = Field(default="", description="Flow name clue from the user request.")
    flow_id: Optional[str] = Field(default=None, description="Resolved existing flow ID when uniquely matched.")
    service_type: Optional[str] = Field(default=None, description="Inherited service type for the resolved flow.")
    service_type_id: Optional[int] = Field(default=None, description="Inherited service type ID for the resolved flow.")
    bw_ul: Optional[float] = Field(default=None, description="Requested target UL bandwidth in Mbps.")
    bw_dl: Optional[float] = Field(default=None, description="Requested target DL bandwidth in Mbps.")
    gbr_ul: Optional[float] = Field(default=None, description="Inherited GBR UL in Mbps.")
    gbr_dl: Optional[float] = Field(default=None, description="Inherited GBR DL in Mbps.")
    lat: Optional[float] = Field(default=None, description="Inherited latency requirement in ms.")
    loss_req: Optional[float] = Field(default=None, description="Inherited packet loss requirement.")
    jitter_req: Optional[float] = Field(default=None, description="Inherited jitter requirement in ms.")
    priority: Optional[int] = Field(default=None, description="Inherited priority for the resolved flow.")
    description: Optional[str] = Field(default=None, description="Human-readable flow description.")
    supi: Optional[str] = Field(default=None, description="User SUPI.")
    resolution_status: str = Field(default="unmatched", description="One of: resolved, ambiguous, unmatched.")
    current_bw_ul: Optional[float] = Field(default=None, description="Current UL bandwidth baseline in Mbps.")
    current_bw_dl: Optional[float] = Field(default=None, description="Current DL bandwidth baseline in Mbps.")
    five_tuple: Optional[List[Any]] = Field(default=None, description="Resolved five tuple: [src_ip, dst_ip, src_port, dst_port, protocol].")
    resolution_candidates: List[str] = Field(default_factory=list, description="Candidate app/flow labels when not uniquely resolved.")


class UserIntent(BaseModel):
    supi: Optional[str] = Field(default=None, description="UE SUPI such as imsi-...")
    app_name: Optional[str] = Field(default=None, description="App name clue from the user request.")
    app_id: Optional[str] = Field(default=None, description="Resolved existing app ID when uniquely matched.")
    operation_type: str = Field(default="modify", description="Operation type: add, modify, delete.")
    flows: List[FlowIntent] = Field(default_factory=list, description="Requested flows to be modified.")
    urgency: str = Field(default="Normal", description="Overall urgency.")
    raw_intent_summary: str = Field(default="", description="Summary of the raw user request.")
    resolution_status: str = Field(default="unmatched", description="One of: resolved, ambiguous, unmatched.")

