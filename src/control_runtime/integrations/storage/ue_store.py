from __future__ import annotations

from typing import Any

from ..testbed import TestbedClient, TestbedConfig
from ..testbed.projection import (
    build_am_context,
    build_flow_catalog,
    project_subscription,
)
from .session_store import get_latest_snapshot_data, get_snapshot_data_by_id


def _snapshot(snapshot_id: str) -> dict[str, Any]:
    return (
        get_snapshot_data_by_id(snapshot_id)
        if snapshot_id.strip()
        else get_latest_snapshot_data()
    )


def _subscription(supi: str) -> dict[str, Any]:
    client = TestbedClient.create(TestbedConfig.from_env())
    return project_subscription(
        supi=supi,
        plmn_id=client.config.plmn_id,
        subscriber=client.subscriber(supi),
    )


def get_ue_context_by_supi(supi: str, snapshot_id: str = "") -> dict[str, Any]:
    normalized_supi = supi.strip()
    if not normalized_supi:
        raise ValueError("supi is required")
    native = _snapshot(snapshot_id)
    flow_catalog = build_flow_catalog(native, normalized_supi)
    subscription = _subscription(normalized_supi)
    am_context = build_am_context(native, subscription, normalized_supi)
    sessions = [item for item in native["sessions"] if item["supi"] == normalized_supi]
    return {
        "supi": normalized_supi,
        "snapshot_id": native["snapshot_id"],
        "app_catalog": flow_catalog["app_catalog"],
        "flow_catalog": flow_catalog["flow_catalog"],
        "sessions": sessions,
        "subscription": subscription,
        "am_policy_context": am_context,
        "access_mobility_context": {
            "accessType": am_context["access_type"],
            "servingGnb": am_context["serving_gnb"],
            "rnti": am_context["rnti"],
        },
    }


def get_ue_flow_catalog_by_supi(supi: str, snapshot_id: str = "") -> dict[str, Any]:
    return build_flow_catalog(_snapshot(snapshot_id), supi.strip())


def list_ue_contexts(snapshot_id: str = "") -> list[dict[str, Any]]:
    native = _snapshot(snapshot_id)
    supis = sorted({flow["supi"] for flow in native["flows"]})
    return [
        get_ue_context_by_supi(supi, snapshot_id=native["snapshot_id"])
        for supi in supis
    ]


def list_am_policy_associations_by_supi(
    supi: str, snapshot_id: str = ""
) -> list[dict[str, Any]]:
    context = get_ue_context_by_supi(supi, snapshot_id=snapshot_id)
    return list(context["am_policy_context"]["associations"])


__all__ = [
    "get_ue_context_by_supi",
    "get_ue_flow_catalog_by_supi",
    "list_am_policy_associations_by_supi",
    "list_ue_contexts",
]
