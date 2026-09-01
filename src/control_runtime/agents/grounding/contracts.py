from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from shared.contracts import (
    AgentMessage,
    CONTRACT_VERSION,
    ContractModel,
    ContractVersion,
    ExecutionScope,
    JsonValue,
    NonEmptyStr,
)

from ..main.contracts import OpenQuestion


VALID_DOMAINS = {"qos", "mobility"}
SM_GROUNDING_TOOLS = {
    "get_sm_ue_context",
    "get_sm_ue_flow_catalog",
    "search_sm_flow_targets",
    "get_ue_flow_catalog",
    "search_flow_targets_by_name",
}
AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}


class GroundingDecisionRequest(ContractModel):
    schema_version: ContractVersion = CONTRACT_VERSION
    scope: ExecutionScope
    user_input: NonEmptyStr
    context: str
    messages: list[AgentMessage]
    allow_user_interaction: bool

    @model_validator(mode="after")
    def validate_scope(self) -> "GroundingDecisionRequest":
        if self.scope.snapshot_id is None:
            raise ValueError("grounding request requires scope.snapshot_id")
        if any(message.role not in {"user", "assistant"} for message in self.messages):
            raise ValueError("grounding request history only accepts user and assistant messages")
        return self


def normalize_requested_domains(requested_domains: Any) -> List[str]:
    if not isinstance(requested_domains, list):
        raise TypeError("main_intent.requested_domains must be a list")
    normalized = [str(item).strip().lower() for item in requested_domains]
    invalid = sorted({item for item in normalized if item not in VALID_DOMAINS})
    if invalid:
        raise ValueError(f"unsupported requested domains: {invalid}")
    return list(dict.fromkeys(normalized))


def normalize_domain_evidence(domain_evidence: Any) -> Dict[str, List[str]]:
    if not isinstance(domain_evidence, dict):
        raise TypeError("main_intent.domain_evidence must be an object")
    normalized: Dict[str, List[str]] = {}
    for key, values in domain_evidence.items():
        if not isinstance(values, list):
            raise TypeError(f"main_intent.domain_evidence[{key!r}] must be a list")
        normalized[str(key).strip().lower()] = [str(item).strip() for item in values]
    return normalized


@dataclass(frozen=True)
class IntentEncodingDirectives:
    requested_domains: List[str]
    domain_evidence: Dict[str, List[str]]
    supi: str
    retry_scope: str
    serving_slice_change_requested: bool

    @classmethod
    def from_context(cls, context: str) -> "IntentEncodingDirectives":
        payload = json.loads(context)
        if not isinstance(payload, dict):
            raise TypeError("IEA context must be a JSON object")
        main_intent = payload.get("main_intent")
        if not isinstance(main_intent, dict):
            raise TypeError("IEA context.main_intent must be an object")
        return cls(
            requested_domains=normalize_requested_domains(main_intent.get("requested_domains")),
            domain_evidence=normalize_domain_evidence(main_intent.get("domain_evidence")),
            supi=str(main_intent.get("supi") or "").strip(),
            retry_scope=str(main_intent.get("retry_scope") or "").strip(),
            serving_slice_change_requested=main_intent.get("serving_slice_change_requested") is True,
        )

    def model_dump(self) -> Dict[str, Any]:
        return {
            "requested_domains": list(self.requested_domains),
            "domain_evidence": dict(self.domain_evidence),
            "supi": self.supi,
            "retry_scope": self.retry_scope,
            "serving_slice_change_requested": self.serving_slice_change_requested,
        }


class FlowCandidateEvidence(BaseModel):
    supi: str = ""
    app_id: str = ""
    app_name: Optional[str] = None
    flow_id: str = ""
    flow_name: str = ""
    service_type: Optional[str] = None
    service_type_id: Optional[int] = None
    score: float = 1.0


class ExplicitFlowTarget(BaseModel):
    flow_name: str = ""
    app_name: Optional[str] = None
    supi: Optional[str] = None


class IntentEvidence(BaseModel):
    user_input: str = ""
    supi: str = ""
    requested_domains: List[str] = Field(default_factory=list)
    serving_slice_change_requested: bool = False
    retry_scope: str = ""
    explicit_app_id: str = ""
    explicit_app_name: str = ""
    explicit_flow_id: str = ""
    explicit_flow_name: str = ""
    explicit_flow_targets: List[ExplicitFlowTarget] = Field(default_factory=list)
    candidate_flows: List[FlowCandidateEvidence] = Field(default_factory=list)
    candidate_apps: List[Dict[str, Any]] = Field(default_factory=list)
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    am_context_summary: Dict[str, Any] = Field(default_factory=dict)
    am_policy_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    subscription_summary: Dict[str, Any] = Field(default_factory=dict)
    catalog_evidence_observed: bool = Field(default=False, exclude=True)
    search_evidence_observed: bool = Field(default=False, exclude=True)
    catalog_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    semantic_candidates: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    am_context_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    subscription_payload: Dict[str, Any] = Field(default_factory=dict, exclude=True)


class FlowSelector(ContractModel):
    supi: NonEmptyStr | None
    app_id: NonEmptyStr | None
    app_name: NonEmptyStr | None
    flow_id: NonEmptyStr | None
    target_type: Literal["flow"]
    name: NonEmptyStr
    service_type: NonEmptyStr | None
    service_type_id: int | None
    bw_ul: float | None
    bw_dl: float | None
    gbr_ul: float | None
    gbr_dl: float | None
    lat: float | None
    loss_req: float | None
    jitter_req: float | None
    priority: int | None
    description: NonEmptyStr | None
    five_tuple: list[JsonValue] | None
    current_slice_snssai: NonEmptyStr | None
    current_bw_ul: float | None
    current_bw_dl: float | None
    resolution_status: Literal["resolved", "unresolved"]

    @model_validator(mode="after")
    def validate_resolution(self) -> "FlowSelector":
        if self.resolution_status == "resolved" and (
            self.supi is None or self.app_id is None or self.flow_id is None
        ):
            raise ValueError("resolved flow requires supi, app_id, and flow_id")
        if self.resolution_status == "unresolved" and self.flow_id is not None:
            raise ValueError("unresolved flow cannot claim a flow_id")
        return self


class GroundedAppEvidence(ContractModel):
    app_id: NonEmptyStr
    app_name: NonEmptyStr


class MobilityIntent(ContractModel):
    association_id: NonEmptyStr
    rfsp: int = Field(ge=0)
    allowed_snssais: list[NonEmptyStr]
    target_snssais: list[NonEmptyStr]
    access_type: Literal["3GPP_ACCESS", "NON_3GPP_ACCESS"] | None
    service_area_restriction: dict[str, JsonValue] | None


class MobilityContextEvidence(ContractModel):
    association_ids: list[NonEmptyStr]
    allowed_snssais: list[dict[str, JsonValue]]
    target_snssais: list[dict[str, JsonValue]]
    mapping_snssais: list[dict[str, JsonValue]]
    current_association_id: NonEmptyStr | None
    access_type: NonEmptyStr | None
    rfsp: int | None
    service_area_restriction: dict[str, JsonValue] | None


class GroundedMobilityEvidence(ContractModel):
    summary: MobilityContextEvidence | None
    subscription_entitlement: dict[str, JsonValue] | None


class GroundingEvidenceBundle(ContractModel):
    grounded_supi: NonEmptyStr | None
    grounded_apps: list[GroundedAppEvidence]
    grounded_flows: list[FlowCandidateEvidence]
    mobility: GroundedMobilityEvidence
    evidence_sources: dict[NonEmptyStr, list[NonEmptyStr]]


class QosOperationConstraint(ContractModel):
    flow_id: NonEmptyStr
    app_id: NonEmptyStr
    operation_type: Literal["qos_reallocation", "slice_migration"]
    require_slice_change: bool
    source_slice_snssai: NonEmptyStr | None
    excluded_slice_snssais: list[NonEmptyStr]
    target_slice_preference: str
    no_op_allowed: bool
    rationale: list[NonEmptyStr]

    @field_validator("excluded_slice_snssais")
    @classmethod
    def validate_unique_exclusions(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("excluded_slice_snssais must be unique")
        return values


SliceMigrationDecisionKind = Literal[
    "not_applicable",
    "migration_authorized",
    "blocked_by_subscription_entitlement",
    "blocked_requires_subscription_provisioning",
    "evidence_missing",
]


class SliceMigrationAuthorization(ContractModel):
    decision: SliceMigrationDecisionKind
    authority: NonEmptyStr | None
    authorized_snssais: list[NonEmptyStr]
    target_snssais: list[NonEmptyStr]
    subscription_change_required: bool
    rationale: list[NonEmptyStr]


class GroundedFlowBinding(ContractModel):
    app_id: NonEmptyStr | None
    flow_id: NonEmptyStr | None
    name: NonEmptyStr
    resolution_status: Literal["resolved", "unresolved"]

    @model_validator(mode="after")
    def validate_resolution(self) -> "GroundedFlowBinding":
        if self.resolution_status == "resolved" and (self.app_id is None or self.flow_id is None):
            raise ValueError("resolved binding requires app_id and flow_id")
        if self.resolution_status == "unresolved" and (self.app_id is not None or self.flow_id is not None):
            raise ValueError("unresolved binding cannot claim app_id or flow_id")
        return self


class QosOperationDecision(ContractModel):
    flow_id: NonEmptyStr
    app_id: NonEmptyStr
    operation_type: Literal["qos_reallocation", "slice_migration"]
    require_slice_change: bool
    target_slice_preference: str
    no_op_allowed: bool
    rationale: list[NonEmptyStr]


class SliceMigrationDecision(ContractModel):
    decision: SliceMigrationDecisionKind
    target_snssais: list[NonEmptyStr]
    subscription_change_required: bool
    rationale: list[NonEmptyStr]


class GroundingAdvisorOutput(ContractModel):
    flows: list[GroundedFlowBinding]
    mobility_intent: MobilityIntent | None
    qos_operation_constraints: list[QosOperationDecision]
    slice_migration_authorization: SliceMigrationDecision
    open_questions: list[OpenQuestion]


class GroundingDecision(ContractModel):
    schema_version: ContractVersion = CONTRACT_VERSION
    scope: ExecutionScope
    flows: list[FlowSelector]
    mobility_intent: MobilityIntent | None
    grounding_evidence: GroundingEvidenceBundle
    qos_operation_constraints: list[QosOperationConstraint]
    slice_migration_authorization: SliceMigrationAuthorization
    open_questions: list[OpenQuestion]

    @model_validator(mode="after")
    def _validate_grounded_qos_flows(self) -> "GroundingDecision":
        for index, flow in enumerate(self.flows):
            if flow.resolution_status != "resolved":
                continue
            if flow.flow_id is None or flow.app_id is None:
                raise ValueError(f"resolved qos flow at flows[{index}] is missing binding identity")
        resolved_ids = [flow.flow_id for flow in self.flows if flow.flow_id is not None]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("grounding decision flow_id values must be unique")
        return self


__all__ = [
    "GroundingDecisionRequest",
    "AM_GROUNDING_TOOLS",
    "ExplicitFlowTarget",
    "FlowCandidateEvidence",
    "FlowSelector",
    "GroundedAppEvidence",
    "GroundedFlowBinding",
    "GroundedMobilityEvidence",
    "GroundingAdvisorOutput",
    "GroundingDecision",
    "GroundingEvidenceBundle",
    "IntentEncodingDirectives",
    "IntentEvidence",
    "MobilityContextEvidence",
    "MobilityIntent",
    "QosOperationConstraint",
    "QosOperationDecision",
    "SliceMigrationAuthorization",
    "SliceMigrationDecision",
    "SliceMigrationDecisionKind",
    "SM_GROUNDING_TOOLS",
    "VALID_DOMAINS",
    "normalize_domain_evidence",
    "normalize_requested_domains",
]
