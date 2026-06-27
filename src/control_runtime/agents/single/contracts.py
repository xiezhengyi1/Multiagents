from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class SingleAgentRoundDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    requested_domains: List[str] = Field(default_factory=list)
    supi: str = ""
    objective_profile_hint: str = "balanced"
    sm_policies: List[Dict[str, Any]] = Field(default_factory=list, validation_alias=AliasChoices("sm_policies", "smPolicies"))
    am_policy: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("am_policy", "amPolicy"))
    ursp_policies: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ursp_policies", "urspPolicies", "ursp_rules"),
    )

    @field_validator("requested_domains", mode="before")
    @classmethod
    def _normalize_requested_domains(cls, value: Any) -> List[str]:
        normalized: List[str] = []
        for item in value or []:
            text = str(item or "").strip().lower()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

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

    @model_validator(mode="after")
    def _validate_domains_and_policies(self) -> "SingleAgentRoundDecision":
        allowed = {"qos", "mobility"}
        if not self.requested_domains:
            inferred_domains: List[str] = []
            if self.sm_policies:
                inferred_domains.append("qos")
            if self.am_policy is not None:
                inferred_domains.append("mobility")
            self.requested_domains = inferred_domains
        if not self.requested_domains:
            raise ValueError("final product must contain sm_policies and/or am_policy")
        if any(item not in allowed for item in self.requested_domains):
            raise ValueError("requested_domains contains unsupported values")
        if not str(self.supi or "").strip():
            raise ValueError("grounded supi is required")
        profile_name = str(self.objective_profile_hint or "").strip().lower()
        if not profile_name:
            self.objective_profile_hint = "balanced"
            profile_name = "balanced"
        if profile_name not in {"balanced", "latency", "throughput", "stability"}:
            raise ValueError("objective_profile_hint contains unsupported value")

        has_qos = "qos" in self.requested_domains
        has_mobility = "mobility" in self.requested_domains

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
