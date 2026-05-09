from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from ...domain.policy_plan import FlowSelector


class SingleAgentRoundDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    requested_domains: List[str] = Field(default_factory=list)
    domain_evidence: Dict[str, List[Any]] = Field(default_factory=dict)
    selected_app_id: str = ""
    selected_flow_id: str = ""
    operation_type: str = "modify"
    supi: str = ""
    intent: str = ""
    raw_intent_summary: str = ""
    rationale: str = ""
    mobility_intent: Dict[str, Any] = Field(default_factory=dict)
    objective_profile_hint: str = ""
    flows: List[FlowSelector] = Field(default_factory=list)
    sm_policies: List[Dict[str, Any]] = Field(default_factory=list, validation_alias=AliasChoices("sm_policies", "smPolicies"))
    am_policy: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("am_policy", "amPolicy"))
    ursp_policies: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ursp_policies", "urspPolicies", "ursp_rules"),
    )
    planning_metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices("planning_metadata", "planningMetadata"))

    @field_validator("requested_domains", mode="before")
    @classmethod
    def _normalize_requested_domains(cls, value: Any) -> List[str]:
        normalized: List[str] = []
        for item in value or []:
            text = str(item or "").strip().lower()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @field_validator("domain_evidence", mode="before")
    @classmethod
    def _normalize_domain_evidence(cls, value: Any) -> Dict[str, List[Any]]:
        normalized: Dict[str, List[Any]] = {}
        if not isinstance(value, dict):
            return normalized
        for key, items in value.items():
            bucket: List[Any] = []
            if isinstance(items, list):
                for item in items:
                    if item in (None, "", {}, []):
                        continue
                    if isinstance(item, str):
                        text = item.strip()
                        if text:
                            bucket.append(text)
                        continue
                    bucket.append(item)
            elif items not in (None, "", {}, []):
                bucket = [items]
            if bucket:
                normalized[str(key or "").strip().lower()] = bucket
        return normalized

    @field_validator("mobility_intent", "planning_metadata", mode="before")
    @classmethod
    def _normalize_dict_fields(cls, value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @field_validator("sm_policies", "ursp_policies", mode="before")
    @classmethod
    def _normalize_policy_lists(cls, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    @field_validator("am_policy", mode="before")
    @classmethod
    def _normalize_am_policy(cls, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, dict) and not value:
            return None
        return dict(value) if isinstance(value, dict) else None

    @field_validator("flows", mode="before")
    @classmethod
    def _normalize_flows(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        normalized: List[Any] = []
        for item in value:
            if isinstance(item, dict):
                candidate = dict(item)
                if "name" not in candidate and "flow_name" in candidate:
                    candidate["name"] = candidate.get("flow_name")
                if "target_type" not in candidate:
                    candidate["target_type"] = "flow"
                normalized.append(candidate)
                continue
            text = str(item or "").strip()
            if text:
                normalized.append(
                    {
                        "flow_id": text,
                        "name": text,
                        "target_type": "flow",
                    }
                )
        return normalized

    @model_validator(mode="after")
    def _validate_domains_and_policies(self) -> "SingleAgentRoundDecision":
        allowed = {"qos", "mobility"}
        if not self.requested_domains:
            raise ValueError("requested_domains must not be empty")
        if any(item not in allowed for item in self.requested_domains):
            raise ValueError("requested_domains contains unsupported values")
        if not str(self.supi or "").strip():
            raise ValueError("grounded supi is required")
        profile_name = str(self.objective_profile_hint or "").strip().lower()
        if not profile_name:
            raise ValueError("objective_profile_hint is required")
        if profile_name not in {"balanced", "latency", "throughput", "stability"}:
            raise ValueError("objective_profile_hint contains unsupported value")

        has_qos = "qos" in self.requested_domains
        has_mobility = "mobility" in self.requested_domains

        if has_qos and not self.flows:
            raise ValueError("qos intent requires non-empty flows")
        if not has_qos and self.flows:
            raise ValueError("non-qos intent must not include qos flows")
        if has_qos and not self.sm_policies:
            raise ValueError("qos intent requires sm_policies")
        if not has_qos and self.sm_policies:
            raise ValueError("non-qos intent must not include sm_policies")
        if has_mobility and self.am_policy is None:
            raise ValueError("mobility intent requires am_policy")
        if not has_mobility and self.am_policy is not None:
            raise ValueError("non-mobility intent must not include am_policy")
        return self


__all__ = ["SingleAgentRoundDecision"]
