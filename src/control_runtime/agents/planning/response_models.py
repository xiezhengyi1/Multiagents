from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class SnssaiSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    sst: int = Field(ge=0, le=255)
    sd: Optional[str] = Field(default=None, min_length=6, max_length=6)

    @field_validator("sd", mode="before")
    @classmethod
    def _normalize_sd(cls, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
            raise ValueError("sd must be a 6-character hexadecimal string")
        return text.lower()


class TrafficDescriptorSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    flow_descs: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("flow_descs", "flowDescs"),
    )
    dnns: List[str] = Field(default_factory=list)
    app_ids: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("app_ids", "appIds"),
    )
    os_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("os_id", "osId"),
    )

    @model_validator(mode="after")
    def _require_descriptor_signal(self) -> "TrafficDescriptorSpec":
        if not self.flow_descs and not self.dnns and not self.app_ids:
            raise ValueError("traffic descriptor requires at least one of flow_descs, dnns, or app_ids")
        return self


class RouteSelectionParameterSetSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dnn: str = Field(min_length=1)
    precedence: Optional[int] = Field(default=None, ge=1)
    snssai: Optional[SnssaiSpec] = None


class SmPolicySpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    flow_id: str = Field(min_length=1, validation_alias=AliasChoices("flow_id", "flowId"))
    app_id: str = Field(min_length=1, validation_alias=AliasChoices("app_id", "appId"))
    priority: int = Field(ge=1, le=15)
    target_latency_ms: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("target_latency_ms", "targetLatencyMs"),
    )
    packet_error_rate: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("packet_error_rate", "packetErrorRate"),
    )
    max_br_ul_mbps: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("max_br_ul_mbps", "maxBrUlMbps"),
    )
    max_br_dl_mbps: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("max_br_dl_mbps", "maxBrDlMbps"),
    )
    gbr_ul_mbps: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("gbr_ul_mbps", "gbrUlMbps"),
    )
    gbr_dl_mbps: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("gbr_dl_mbps", "gbrDlMbps"),
    )
    target_jitter_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("target_jitter_ms", "targetJitterMs"),
    )
    flow_description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("flow_description", "flowDescription"),
    )

    @model_validator(mode="after")
    def _validate_gbr_not_above_maxbr(self) -> "SmPolicySpec":
        errors: List[str] = []
        if self.gbr_ul_mbps is not None and self.gbr_ul_mbps > self.max_br_ul_mbps:
            errors.append(
                "gbr_ul_mbps must not exceed max_br_ul_mbps "
                f"({self.gbr_ul_mbps} > {self.max_br_ul_mbps})"
            )
        if self.gbr_dl_mbps is not None and self.gbr_dl_mbps > self.max_br_dl_mbps:
            errors.append(
                "gbr_dl_mbps must not exceed max_br_dl_mbps "
                f"({self.gbr_dl_mbps} > {self.max_br_dl_mbps})"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self


class AmPolicySpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    triggers: List[str] = Field(min_length=1)
    rfsp: int = Field(ge=1)
    allowed_snssais: List[SnssaiSpec] = Field(
        min_length=1,
        validation_alias=AliasChoices("allowed_snssais", "allowedSnssais"),
    )
    target_snssais: List[SnssaiSpec] = Field(
        min_length=1,
        validation_alias=AliasChoices("target_snssais", "targetSnssais"),
    )
    ue_ambr_ul_mbps: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("ue_ambr_ul_mbps", "ueAmbrUlMbps"),
    )
    ue_ambr_dl_mbps: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("ue_ambr_dl_mbps", "ueAmbrDlMbps"),
    )
    serv_area_res: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("serv_area_res", "servAreaRes"),
    )

    @model_validator(mode="after")
    def _validate_snssai_coverage(self) -> "AmPolicySpec":
        allowed = {(item.sst, item.sd) for item in self.allowed_snssais}
        target = {(item.sst, item.sd) for item in self.target_snssais}
        if not target.issubset(allowed):
            raise ValueError("target_snssais must be a subset of allowed_snssais")
        return self


class UrspPolicySpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    target_type: str = Field(
        default="flow",
        validation_alias=AliasChoices("target_type", "targetType"),
    )
    app_id: str = Field(min_length=1, validation_alias=AliasChoices("app_id", "appId"))
    flow_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("flow_id", "flowId"))
    relat_precedence: int = Field(
        ge=1,
        validation_alias=AliasChoices("relat_precedence", "relatPrecedence"),
    )
    traffic_desc: Optional[TrafficDescriptorSpec] = Field(
        default=None,
        validation_alias=AliasChoices("traffic_desc", "trafficDesc"),
    )
    route_sel_param_sets: List[RouteSelectionParameterSetSpec] = Field(
        min_length=1,
        validation_alias=AliasChoices("route_sel_param_sets", "routeSelParamSets"),
    )

    @field_validator("target_type", mode="before")
    @classmethod
    def _normalize_target_type(cls, value: Any) -> str:
        normalized = str(value or "flow").strip().lower()
        if normalized not in {"flow", "app"}:
            raise ValueError("target_type must be either 'flow' or 'app'")
        return normalized

    @model_validator(mode="after")
    def _validate_target_binding(self) -> "UrspPolicySpec":
        if self.target_type == "flow" and not str(self.flow_id or "").strip():
            raise ValueError("flow-scoped URSP policy requires flow_id")
        if self.target_type == "flow" and self.traffic_desc is None:
            raise ValueError("flow-scoped URSP policy requires traffic_desc")
        return self


class OsaAdvisorOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    planning_status: str = Field(default="executable_plan")
    rationale: str = Field(default="")
    missing_evidence: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("missing_evidence", "missingEvidence"),
    )
    blocked_targets: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("blocked_targets", "blockedTargets"),
    )
    upstream_requests: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("upstream_requests", "upstreamRequests"),
    )
    planner_conflicts: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("planner_conflicts", "plannerConflicts"),
    )
    sm_policies: List[SmPolicySpec] = Field(
        default_factory=list,
        validation_alias=AliasChoices("sm_policies", "smPolicies"),
    )
    am_policy: Optional[AmPolicySpec] = Field(
        default=None,
        validation_alias=AliasChoices("am_policy", "amPolicy"),
    )
    ursp_policies: List[UrspPolicySpec] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ursp_policies", "urspPolicies"),
    )
    partial_policies: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("partial_policies", "partialPolicies"),
    )
    @field_validator("planning_status", mode="before")
    @classmethod
    def _normalize_planning_status(cls, value: Any) -> str:
        normalized = str(value or "executable_plan").strip().lower()
        if normalized not in {"executable_plan", "partial_plan", "needs_upstream_reground"}:
            raise ValueError(
                "planning_status must be one of executable_plan, partial_plan, or needs_upstream_reground"
            )
        return normalized

    @model_validator(mode="after")
    def _validate_policy_collections(self) -> "OsaAdvisorOutput":
        seen_flow_ids: set[str] = set()
        for item in self.sm_policies:
            if item.flow_id in seen_flow_ids:
                raise ValueError(f"duplicate sm_policies flow_id={item.flow_id}")
            seen_flow_ids.add(item.flow_id)
        if self.planning_status == "needs_upstream_reground":
            if self.sm_policies or self.am_policy is not None or self.ursp_policies:
                raise ValueError("needs_upstream_reground must not include executable policy payloads")
            if not self.missing_evidence and not self.upstream_requests and not self.blocked_targets:
                raise ValueError("needs_upstream_reground must explain what is missing or blocked")
        if self.planning_status == "partial_plan":
            if not self.partial_policies and not self.sm_policies and self.am_policy is None and not self.ursp_policies:
                raise ValueError("partial_plan must include partial policy output or executable fragments")
            if not self.missing_evidence and not self.blocked_targets and not self.planner_conflicts:
                raise ValueError("partial_plan must explain what blocks full execution")
        return self


__all__ = [
    "AmPolicySpec",
    "OsaAdvisorOutput",
    "RouteSelectionParameterSetSpec",
    "SmPolicySpec",
    "SnssaiSpec",
    "TrafficDescriptorSpec",
    "UrspPolicySpec",
]
