from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from .ue_store import (
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    list_ue_contexts,
)


def _score(query: str, value: object) -> float:
    normalized_query = query.strip().casefold()
    normalized_value = str(value).strip().casefold()
    if not normalized_query or not normalized_value:
        return 0.0
    if normalized_query in normalized_value:
        return 1.0
    return SequenceMatcher(None, normalized_query, normalized_value).ratio()


def search_flow_targets_by_semantic(
    *,
    supi: str = "",
    app_name: str = "",
    flow_name: str = "",
    snapshot_id: str = "",
    limit: int = 5,
    min_score: float = 0.35,
) -> dict[str, Any]:
    if not app_name.strip() and not flow_name.strip():
        raise ValueError("app_name or flow_name is required")
    catalogs = (
        [get_ue_flow_catalog_by_supi(supi, snapshot_id=snapshot_id)]
        if supi.strip()
        else [
            get_ue_flow_catalog_by_supi(item["supi"], snapshot_id=snapshot_id)
            for item in list_ue_contexts(snapshot_id=snapshot_id)
        ]
    )
    candidates = []
    for catalog in catalogs:
        for flow in catalog["flow_catalog"]:
            app_score = _score(app_name, flow["app_name"]) if app_name.strip() else 0.0
            flow_score = _score(flow_name, flow["name"]) if flow_name.strip() else 0.0
            score = (
                min(app_score, flow_score)
                if app_name.strip() and flow_name.strip()
                else max(app_score, flow_score)
            )
            if score >= min_score:
                candidates.append({**flow, "match_score": round(score, 4)})
    candidates.sort(key=lambda item: (-item["match_score"], item["flow_id"]))
    return {
        "query": {"supi": supi, "app_name": app_name, "flow_name": flow_name},
        "candidate_count": len(candidates),
        "candidates": candidates[:limit],
    }


def search_am_policy_targets_by_context(
    *,
    supi: str,
    association_id: str = "",
    allowed_snssai: str = "",
    target_snssai: str = "",
    service_area: str = "",
    rfsp: str = "",
    access_type: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    context = get_ue_context_by_supi(supi)["am_policy_context"]
    searchable = " ".join(
        str(value)
        for value in (
            context["association_id"],
            context["allowed_snssais"],
            context["target_snssais"],
            context["service_area_restriction"],
            context["rfsp"],
            context["access_type"],
        )
    ).casefold()
    queries = [
        value.casefold()
        for value in (
            association_id,
            allowed_snssai,
            target_snssai,
            service_area,
            rfsp,
            access_type,
        )
        if value.strip()
    ]
    matches = [context] if all(query in searchable for query in queries) else []
    return {
        "query": {
            "supi": supi,
            "association_id": association_id,
            "allowed_snssai": allowed_snssai,
            "target_snssai": target_snssai,
            "service_area": service_area,
            "rfsp": rfsp,
            "access_type": access_type,
        },
        "candidate_count": len(matches),
        "candidates": matches[:limit],
    }


__all__ = ["search_am_policy_targets_by_context", "search_flow_targets_by_semantic"]
