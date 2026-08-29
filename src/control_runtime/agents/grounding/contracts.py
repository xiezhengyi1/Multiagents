from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class FlowSelector(BaseModel):
    supi: str = Field(default="")
    app_id: str = Field(default="")
    app_name: Optional[str] = Field(default=None)
    flow_id: Optional[str] = Field(default=None)
    target_type: str = Field(default="flow")
    name: str = Field(default="")
    service_type: Optional[str] = Field(default=None)
    service_type_id: Optional[int] = Field(default=None)
    bw_ul: Optional[float] = Field(default=None)
    bw_dl: Optional[float] = Field(default=None)
    gbr_ul: Optional[float] = Field(default=None)
    gbr_dl: Optional[float] = Field(default=None)
    lat: Optional[float] = Field(default=None)
    loss_req: Optional[float] = Field(default=None)
    jitter_req: Optional[float] = Field(default=None)
    priority: Optional[int] = Field(default=None)
    description: Optional[str] = Field(default=None)
    five_tuple: Optional[List[Any]] = Field(default=None)
    current_slice_snssai: Optional[str] = Field(default=None)
    current_bw_ul: Optional[float] = Field(default=None)
    current_bw_dl: Optional[float] = Field(default=None)
    resolution_status: str = Field(default="resolved")


class GroundingEvidenceBundle(BaseModel):
    grounded_supi: str = Field(default="")
    grounded_apps: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_flows: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_mobility_targets: Dict[str, Any] = Field(default_factory=dict)
    evidence_sources: Dict[str, List[str]] = Field(default_factory=dict)


class QosOperationConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_id: str = Field(default="")
    app_id: str = Field(default="")
    operation_type: str = Field(default="")
    require_slice_change: bool = Field(default=False)
    source_slice_snssai: Optional[str] = Field(default=None)
    excluded_slice_snssais: List[str] = Field(default_factory=list)
    target_slice_preference: str = Field(default="")
    no_op_allowed: bool = Field(default=True)
    rationale: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_excluded_source(self) -> "QosOperationConstraint":
        source = str(self.source_slice_snssai or "").strip()
        excluded = [str(item or "").strip() for item in self.excluded_slice_snssais if str(item or "").strip()]
        if self.require_slice_change and source and source not in excluded:
            excluded.append(source)
        self.excluded_slice_snssais = excluded
        return self


SliceMigrationDecisionKind = Literal[
    "not_applicable",
    "migration_authorized",
    "blocked_by_subscription_entitlement",
    "blocked_requires_subscription_provisioning",
    "evidence_missing",
]


class SliceMigrationAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: SliceMigrationDecisionKind = Field(default="not_applicable")
    authority: str = Field(default="")
    authorized_snssais: List[str] = Field(default_factory=list)
    target_snssais: List[str] = Field(default_factory=list)
    subscription_change_required: bool = Field(default=False)
    rationale: List[str] = Field(default_factory=list)


class GroundedFlowBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_id: str = Field(default="")
    flow_id: str = Field(default="")
    name: str = Field(default="")
    resolution_status: Literal["resolved", "unresolved"] = Field(default="resolved")


class QosOperationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_id: str = Field(default="")
    app_id: str = Field(default="")
    operation_type: Literal["qos_reallocation", "slice_migration"] = Field(default="qos_reallocation")
    require_slice_change: bool = Field(default=False)
    target_slice_preference: str = Field(default="")
    no_op_allowed: bool = Field(default=True)
    rationale: List[str] = Field(default_factory=list)


class SliceMigrationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: SliceMigrationDecisionKind = Field(default="not_applicable")
    target_snssais: List[str] = Field(default_factory=list)
    subscription_change_required: bool = Field(default=False)
    rationale: List[str] = Field(default_factory=list)


class GroundingAdvisorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: List[GroundedFlowBinding]
    mobility_intent: Dict[str, Any] = Field(default_factory=dict)
    qos_operation_constraints: List[QosOperationDecision] = Field(default_factory=list)
    slice_migration_authorization: SliceMigrationDecision = Field(default_factory=SliceMigrationDecision)
    open_questions: List[OpenQuestion] = Field(default_factory=list)


class GroundingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: List[FlowSelector] = Field(default_factory=list)
    mobility_intent: Dict[str, Any] = Field(default_factory=dict)
    grounding_evidence: GroundingEvidenceBundle = Field(default_factory=GroundingEvidenceBundle)
    qos_operation_constraints: List[QosOperationConstraint] = Field(default_factory=list)
    slice_migration_authorization: SliceMigrationAuthorization = Field(default_factory=SliceMigrationAuthorization)
    open_questions: List[OpenQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_grounded_qos_flows(self) -> "GroundingDecision":
        for index, flow in enumerate(self.flows):
            if str(flow.resolution_status or "").strip().lower() != "resolved":
                continue
            if str(flow.flow_id or "").strip() and str(flow.app_id or "").strip():
                continue
            raise ValueError(
                "resolved qos flow must include both flow_id and app_id "
                f"(flows[{index}] has flow_id={str(flow.flow_id or '').strip() or '<empty>'}, "
                f"app_id={str(flow.app_id or '').strip() or '<empty>'})"
            )
        return self


__all__ = [
    "AM_GROUNDING_TOOLS",
    "ExplicitFlowTarget",
    "FlowCandidateEvidence",
    "FlowSelector",
    "GroundedFlowBinding",
    "GroundingAdvisorOutput",
    "GroundingDecision",
    "GroundingEvidenceBundle",
    "IntentEncodingDirectives",
    "IntentEvidence",
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
