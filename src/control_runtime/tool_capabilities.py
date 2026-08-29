"""Read-only 6G tool capability aliases for explicit trace construction."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

CONTROL_RUNTIME_CAPABILITY_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "get_sm_ue_context": ("sm_ue_context",),
        "get_sm_ue_flow_catalog": ("sm_flow_catalog",),
        "get_ue_flow_catalog": ("sm_flow_catalog",),
        "search_sm_flow_targets": ("sm_flow_target_resolution",),
        "search_flow_targets_by_name": ("sm_flow_target_resolution",),
        "get_am_policy_context": ("am_policy_context",),
        "search_am_policy_targets": ("am_policy_target_resolution",),
        "get_ue_slice_subscription": ("ue_slice_subscription_entitlement",),
        "preview_qos_optimizer": ("optimizer_counterfactual", "qos_runtime_evidence"),
        "preview_optimizer": ("optimizer_counterfactual", "qos_runtime_evidence"),
        "fetch_qos_network_status": ("qos_runtime_evidence",),
        "fetch_network_status": ("qos_runtime_evidence",),
        "inspect_mobility_ue_policies": ("ue_policy_context", "mobility_policy_context"),
        "inspect_ue_policies": ("ue_policy_context", "mobility_policy_context"),
    }
)


__all__ = ["CONTROL_RUNTIME_CAPABILITY_ALIASES"]
