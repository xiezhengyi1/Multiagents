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
        ge=1.0,
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
    rationale: str = Field(default="")

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
    rationale: str = Field(default="")

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

    rationale: str = Field(default="")
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
    planning_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("planning_metadata", "planningMetadata"),
    )

    @model_validator(mode="after")
    def _require_policy_output(self) -> "OsaAdvisorOutput":
        if not self.sm_policies and self.am_policy is None and not self.ursp_policies:
            raise ValueError("advisor output must contain at least one policy specification")
        seen_flow_ids: set[str] = set()
        for item in self.sm_policies:
            if item.flow_id in seen_flow_ids:
                raise ValueError(f"duplicate sm_policies flow_id={item.flow_id}")
            seen_flow_ids.add(item.flow_id)
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
