from __future__ import annotations

import re
from typing import Any, Dict, List

from .contracts import IntentAdvisorDecision, IntentEvidence

VALID_DOMAINS = {"qos", "mobility"}
SM_GROUNDING_TOOLS = {
    "get_sm_ue_context",
    "get_sm_ue_flow_catalog",
    "search_sm_flow_targets",
    "get_ue_flow_catalog",
    "search_flow_targets_by_name",
}
AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}


def uses_sm_grounding(requested_domains: List[str] | None) -> bool:
    normalized = {
        str(item or "").strip().lower()
        for item in (requested_domains or [])
        if str(item or "").strip()
    }
    return not normalized or "qos" in normalized


def uses_am_grounding(requested_domains: List[str] | None) -> bool:
    normalized = {
        str(item or "").strip().lower()
        for item in (requested_domains or [])
        if str(item or "").strip()
    }
    return not normalized or "mobility" in normalized


def normalize_requested_domains(requested_domains: Any) -> List[str]:
    normalized = [
        str(item or "").strip().lower()
        for item in (requested_domains or [])
        if str(item or "").strip()
    ]
    valid = [item for item in normalized if item in VALID_DOMAINS]
    return list(dict.fromkeys(valid))


def normalize_domain_evidence(domain_evidence: Any) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    if not isinstance(domain_evidence, dict):
        return normalized
    for key, values in domain_evidence.items():
        items = [str(item or "").strip() for item in (values or []) if str(item or "").strip()]
        if items:
            normalized[str(key).strip().lower()] = items
    return normalized


def extract_requested_supis(*texts: str) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for text in texts:
        for match in re.findall(r"(?i)\bimsi-\d{5,}\b", str(text or "")):
            normalized = str(match).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return ordered


def merge_catalog_payloads(*payloads: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    app_catalog: List[Dict[str, Any]] = []
    flow_catalog: List[Dict[str, Any]] = []
    seen_apps: set[tuple[str, str]] = set()
    seen_flows: set[tuple[str, str, str]] = set()

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if key in {"app_catalog", "flow_catalog"}:
                continue
            merged[key] = value
        for item in payload.get("app_catalog") or []:
            if not isinstance(item, dict):
                continue
            app_id = str(item.get("app_id") or "").strip()
            app_name = str(item.get("app_name") or "").strip()
            dedupe_key = (app_id, app_name)
            if not any(dedupe_key) or dedupe_key in seen_apps:
                continue
            seen_apps.add(dedupe_key)
            app_catalog.append(dict(item))
        for item in payload.get("flow_catalog") or []:
            if not isinstance(item, dict):
                continue
            dedupe_key = (
                str(item.get("supi") or "").strip(),
                str(item.get("app_id") or "").strip(),
                str(item.get("flow_id") or "").strip(),
            )
            if not any(dedupe_key) or dedupe_key in seen_flows:
                continue
            seen_flows.add(dedupe_key)
            flow_catalog.append(dict(item))

    if app_catalog:
        merged["app_catalog"] = app_catalog
    if flow_catalog:
        merged["flow_catalog"] = flow_catalog
    return merged


def merge_candidate_dicts(*candidate_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in candidate_groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            dedupe_key = (
                str(item.get("supi") or "").strip(),
                str(item.get("app_id") or "").strip(),
                str(item.get("flow_id") or "").strip(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(dict(item))
    return merged


def classify_domain_resolution(
    *,
    main_requested_domains: List[str],
    grounded_requested_domains: List[str],
    decision: IntentAdvisorDecision,
) -> tuple[str, bool]:
    explicit_resolution = str(decision.domain_resolution or "").strip().lower()
    if explicit_resolution in {"confirmed", "narrowed", "widened", "cannot_confirm"}:
        revision_needed = bool(decision.domain_revision_needed) or explicit_resolution != "confirmed"
        return explicit_resolution, revision_needed
    main_set = {
        str(item or "").strip().lower()
        for item in (main_requested_domains or [])
        if str(item or "").strip()
    }
    grounded_set = {
        str(item or "").strip().lower()
        for item in (grounded_requested_domains or [])
        if str(item or "").strip()
    }
    if not grounded_set:
        return "cannot_confirm", True
    if grounded_set == main_set:
        return "confirmed", bool(decision.domain_revision_needed)
    if grounded_set.issubset(main_set):
        return "narrowed", True
    if main_set.issubset(grounded_set):
        return "widened", True
    return "cannot_confirm", True


def mobility_request_mentions_specific_targets(user_input: str) -> bool:
    text = str(user_input or "").strip().lower()
    if not text:
        return False
    patterns = (
        r"\bassociation\b",
        r"\brfsp\b",
        r"\bnssai\b",
        r"\bs-nssai\b",
        r"\ballowed\s*nssai\b",
        r"\btarget\s*nssai\b",
        r"\bservice[-\s]?area\b",
        r"\bpresence\s*area\b",
        r"\baccess\s*type\b",
        r"\b3gpp_access\b",
        r"\bnon_3gpp_access\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def flow_id_is_grounded(*, flow_id: str, evidence: IntentEvidence) -> bool:
    normalized_flow_id = str(flow_id or "").strip()
    if not normalized_flow_id:
        return False
    if any(str(item.flow_id or "").strip() == normalized_flow_id for item in (evidence.candidate_flows or [])):
        return True
    catalog_payload = evidence.catalog_payload or {}
    flow_catalog = catalog_payload.get("flow_catalog") or []
    if any(str(item.get("flow_id") or "").strip() == normalized_flow_id for item in flow_catalog if isinstance(item, dict)):
        return True
    semantic_candidates = evidence.semantic_candidates or []
    return any(
        str(item.get("flow_id") or "").strip() == normalized_flow_id
        for item in semantic_candidates
        if isinstance(item, dict)
    )


def flow_name_is_grounded(*, flow_name: str, evidence: IntentEvidence) -> bool:
    normalized_flow_name = str(flow_name or "").strip()
    if not normalized_flow_name:
        return False
    if any(str(item.flow_name or "").strip() == normalized_flow_name for item in (evidence.candidate_flows or [])):
        return True
    catalog_payload = evidence.catalog_payload or {}
    flow_catalog = catalog_payload.get("flow_catalog") or []
    if any(str(item.get("flow_name") or "").strip() == normalized_flow_name for item in flow_catalog if isinstance(item, dict)):
        return True
    semantic_candidates = evidence.semantic_candidates or []
    return any(
        str(item.get("flow_name") or "").strip() == normalized_flow_name
        for item in semantic_candidates
        if isinstance(item, dict)
    )
