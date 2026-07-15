"""Typed boundary for the compact Main -> IEA routing payload."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


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
    normalized = [
        str(item or "").strip().lower()
        for item in (requested_domains or [])
        if str(item or "").strip()
    ]
    return list(dict.fromkeys(item for item in normalized if item in VALID_DOMAINS))


def normalize_domain_evidence(domain_evidence: Any) -> Dict[str, List[str]]:
    if not isinstance(domain_evidence, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, values in domain_evidence.items():
        items = [str(item or "").strip() for item in (values or []) if str(item or "").strip()]
        if items:
            normalized[str(key).strip().lower()] = items
    return normalized


def uses_sm_grounding(requested_domains: List[str] | None) -> bool:
    normalized = normalize_requested_domains(requested_domains)
    return not normalized or "qos" in normalized


def uses_am_grounding(requested_domains: List[str] | None) -> bool:
    normalized = normalize_requested_domains(requested_domains)
    return not normalized or "mobility" in normalized


@dataclass(frozen=True)
class IntentEncodingDirectives:
    """Only the routing fields IEA may consume from Main's compact JSON."""

    requested_domains: List[str]
    domain_evidence: Dict[str, List[str]]
    supi: str
    retry_scope: str

    @classmethod
    def from_context(cls, context: str) -> "IntentEncodingDirectives":
        try:
            payload = json.loads(str(context or "").strip())
        except (TypeError, ValueError):
            payload = {}
        main_intent = payload.get("main_intent") if isinstance(payload, dict) else {}
        if not isinstance(main_intent, dict):
            main_intent = {}
        return cls(
            requested_domains=normalize_requested_domains(main_intent.get("requested_domains")),
            domain_evidence=normalize_domain_evidence(main_intent.get("domain_evidence")),
            supi=str(main_intent.get("supi") or "").strip(),
            retry_scope=str(main_intent.get("retry_scope") or "").strip(),
        )

    def model_dump(self) -> Dict[str, Any]:
        return {
            "requested_domains": list(self.requested_domains),
            "domain_evidence": dict(self.domain_evidence),
            "supi": self.supi,
            "retry_scope": self.retry_scope,
        }


__all__ = [
    "AM_GROUNDING_TOOLS",
    "SM_GROUNDING_TOOLS",
    "VALID_DOMAINS",
    "IntentEncodingDirectives",
    "normalize_domain_evidence",
    "normalize_requested_domains",
    "uses_am_grounding",
    "uses_sm_grounding",
]
