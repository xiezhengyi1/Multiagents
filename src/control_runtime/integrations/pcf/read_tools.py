from __future__ import annotations

from typing import Any

from ..storage import (
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    search_flow_targets_by_semantic,
)


def get_sm_ue_context(supi: str, snapshot_id: str = "") -> dict[str, Any]:
    return get_ue_context_by_supi(supi, snapshot_id=snapshot_id)


def get_sm_ue_flow_catalog(supi: str, snapshot_id: str = "") -> dict[str, Any]:
    return get_ue_flow_catalog_by_supi(supi, snapshot_id=snapshot_id)


def search_sm_flow_targets(
    supi: str, query: str, snapshot_id: str = ""
) -> dict[str, Any]:
    return search_flow_targets_by_semantic(
        supi=supi,
        app_name=query,
        snapshot_id=snapshot_id,
    )


def get_am_policy_context(supi: str, snapshot_id: str = "") -> dict[str, Any]:
    return get_ue_context_by_supi(supi, snapshot_id=snapshot_id)["am_policy_context"]


def search_am_policy_targets(
    supi: str, query: str, snapshot_id: str = ""
) -> dict[str, Any]:
    context = get_am_policy_context(supi, snapshot_id=snapshot_id)
    searchable = " ".join(
        str(context[field])
        for field in (
            "association_id",
            "allowed_snssais",
            "target_snssais",
            "serving_gnb",
            "rnti",
        )
    ).casefold()
    return {
        "supi": supi,
        "query": query,
        "matches": [context] if query.casefold() in searchable else [],
        "source_snapshot_id": context["source_snapshot_id"],
    }


def get_ue_slice_subscription(supi: str, snapshot_id: str = "") -> dict[str, Any]:
    return get_ue_context_by_supi(supi, snapshot_id=snapshot_id)["subscription"]


get_ue_context = get_sm_ue_context
get_ue_flow_catalog = get_sm_ue_flow_catalog
search_flow_targets_by_name = search_sm_flow_targets


__all__ = [
    "get_am_policy_context",
    "get_sm_ue_context",
    "get_sm_ue_flow_catalog",
    "get_ue_context",
    "get_ue_flow_catalog",
    "get_ue_slice_subscription",
    "search_am_policy_targets",
    "search_flow_targets_by_name",
    "search_sm_flow_targets",
]
