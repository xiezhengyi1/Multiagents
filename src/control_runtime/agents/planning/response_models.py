from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from shared.contracts import ContractModel, JsonValue, NonEmptyStr


class SnssaiSpec(ContractModel):
    sst: int = Field(ge=0, le=255)
    sd: Optional[str] = Field(pattern=r"^[0-9a-f]{6}$")


def decode_snssai_key(value: str) -> SnssaiSpec:
    normalized = value.strip()
    if len(normalized) != 8 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("S-NSSAI key must be exactly eight lowercase hexadecimal characters")
    return SnssaiSpec(sst=int(normalized[:2], 16), sd=normalized[2:])


class TrafficDescriptorSpec(ContractModel):
    flow_descs: list[NonEmptyStr]
    dnns: list[NonEmptyStr]
    app_ids: list[NonEmptyStr]
    os_id: Optional[NonEmptyStr]

    @model_validator(mode="after")
    def _require_descriptor_signal(self) -> "TrafficDescriptorSpec":
        if not self.flow_descs and not self.dnns and not self.app_ids:
            raise ValueError("traffic descriptor requires at least one of flow_descs, dnns, or app_ids")
        return self


class RouteSelectionParameterSetSpec(ContractModel):
    dnn: NonEmptyStr
    precedence: Optional[int] = Field(ge=1)
    snssai: Optional[SnssaiSpec]


class SmPolicySpec(ContractModel):
    flow_id: NonEmptyStr
    app_id: NonEmptyStr
    priority: int = Field(ge=1, le=15)
    target_latency_ms: float = Field(ge=1.0)
    packet_error_rate: float = Field(ge=0.0, le=1.0)
    max_br_ul_mbps: float = Field(ge=0.0)
    max_br_dl_mbps: float = Field(ge=0.0)
    gbr_ul_mbps: Optional[float] = Field(ge=0.0)
    gbr_dl_mbps: Optional[float] = Field(ge=0.0)
    target_jitter_ms: Optional[float] = Field(ge=0.0)
    flow_description: Optional[NonEmptyStr]

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


class AmPolicySpec(ContractModel):
    triggers: list[NonEmptyStr] = Field(min_length=1)
    rfsp: int = Field(ge=1)
    allowed_snssais: List[SnssaiSpec] = Field(min_length=1)
    target_snssais: List[SnssaiSpec] = Field(min_length=1)
    ue_ambr_ul_mbps: Optional[float] = Field(ge=0.0)
    ue_ambr_dl_mbps: Optional[float] = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_snssai_coverage(self) -> "AmPolicySpec":
        allowed = {(item.sst, item.sd) for item in self.allowed_snssais}
        target = {(item.sst, item.sd) for item in self.target_snssais}
        if not target.issubset(allowed):
            raise ValueError("target_snssais must be a subset of allowed_snssais")
        return self


class UrspPolicySpec(ContractModel):
    target_type: Literal["flow", "app"]
    app_id: NonEmptyStr
    flow_id: Optional[NonEmptyStr]
    relat_precedence: int = Field(ge=1)
    traffic_desc: Optional[TrafficDescriptorSpec]
    route_sel_param_sets: List[RouteSelectionParameterSetSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_target_binding(self) -> "UrspPolicySpec":
        if self.target_type == "flow" and not str(self.flow_id or "").strip():
            raise ValueError("flow-scoped URSP policy requires flow_id")
        if self.target_type == "flow" and self.traffic_desc is None:
            raise ValueError("flow-scoped URSP policy requires traffic_desc")
        return self


class PartialPolicySpec(ContractModel):
    app_id: str
    flow_id: Optional[NonEmptyStr]
    target_type: Literal["flow", "app", "ue"]
    policy_type: NonEmptyStr
    resource_keys: list[NonEmptyStr]
    policy_details: dict[str, JsonValue]


class OsaAdvisorOutput(ContractModel):
    planning_status: Literal["executable_plan", "partial_plan", "needs_upstream_reground"]
    rationale: NonEmptyStr
    missing_evidence: list[NonEmptyStr]
    blocked_targets: list[NonEmptyStr]
    upstream_requests: list[NonEmptyStr]
    planner_conflicts: list[NonEmptyStr]
    sm_policies: list[SmPolicySpec]
    am_policy: Optional[AmPolicySpec]
    ursp_policies: list[UrspPolicySpec]
    partial_policies: list[PartialPolicySpec]

    @model_validator(mode="after")
    def _validate_policy_collections(self) -> "OsaAdvisorOutput":
        for name, values in (
            ("missing_evidence", self.missing_evidence),
            ("blocked_targets", self.blocked_targets),
            ("upstream_requests", self.upstream_requests),
            ("planner_conflicts", self.planner_conflicts),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        seen_flow_ids: set[str] = set()
        for item in self.sm_policies:
            if item.flow_id in seen_flow_ids:
                raise ValueError(f"duplicate sm_policies flow_id={item.flow_id}")
            seen_flow_ids.add(item.flow_id)
        partial_policy_keys = [
            (item.target_type, item.app_id, item.flow_id, item.policy_type)
            for item in self.partial_policies
        ]
        if len(partial_policy_keys) != len(set(partial_policy_keys)):
            raise ValueError("partial_policies target bindings must be unique")
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
    "decode_snssai_key",
    "OsaAdvisorOutput",
    "PartialPolicySpec",
    "RouteSelectionParameterSetSpec",
    "SmPolicySpec",
    "SnssaiSpec",
    "TrafficDescriptorSpec",
    "UrspPolicySpec",
]
