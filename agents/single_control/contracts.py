from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

from domain.policy_plan import FlowSelector


class SingleAgentIntentDecision(BaseModel):
    requested_domains: List[str] = Field(default_factory=list)
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    selected_app_id: str = ""
    selected_flow_id: str = ""
    operation_type: str = "modify"
    raw_intent_summary: str = ""
    rationale: str = ""
    mobility_intent: Dict[str, Any] = Field(default_factory=dict)
    objective_profile_hint: str = ""
    flows: List[FlowSelector] = Field(default_factory=list)

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
    def _normalize_domain_evidence(cls, value: Any) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        if not isinstance(value, dict):
            return normalized
        for key, items in value.items():
            values = [str(item or "").strip() for item in (items or []) if str(item or "").strip()]
            if values:
                normalized[str(key or "").strip().lower()] = values
        return normalized

    @field_validator("mobility_intent", mode="before")
    @classmethod
    def _normalize_mobility_intent(cls, value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @model_validator(mode="after")
    def _validate_domains_and_flows(self) -> "SingleAgentIntentDecision":
        allowed = {"qos", "mobility"}
        if not self.requested_domains:
            raise ValueError("requested_domains must not be empty")
        if any(item not in allowed for item in self.requested_domains):
            raise ValueError("requested_domains contains unsupported values")
        if "qos" in self.requested_domains and not self.flows:
            raise ValueError("qos intent requires non-empty flows")
        if self.requested_domains == ["mobility"] and self.flows:
            raise ValueError("mobility-only intent must not include qos flows")
        return self


__all__ = ["SingleAgentIntentDecision"]
