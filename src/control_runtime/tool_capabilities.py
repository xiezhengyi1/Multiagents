"""6G-specific tool capability aliases.

These were previously hardcoded in ``agent_runtime/messages.py`` and
``agent_runtime/tooling.py`` (duplicated). They are now registered at import
time via ``agent_runtime.messages.register_capability_aliases`` so the generic
runtime library stays free of domain-specific knowledge.
"""

from __future__ import annotations

from typing import Dict, List

CONTROL_RUNTIME_CAPABILITY_ALIASES: Dict[str, List[str]] = {
    "get_sm_ue_context": ["sm_ue_context"],
    "get_sm_ue_flow_catalog": ["sm_flow_catalog"],
    "get_ue_flow_catalog": ["sm_flow_catalog"],
    "search_sm_flow_targets": ["sm_flow_target_resolution"],
    "search_flow_targets_by_name": ["sm_flow_target_resolution"],
    "get_am_policy_context": ["am_policy_context"],
    "search_am_policy_targets": ["am_policy_target_resolution"],
    "preview_qos_optimizer": ["optimizer_counterfactual", "qos_runtime_evidence"],
    "preview_optimizer": ["optimizer_counterfactual", "qos_runtime_evidence"],
    "fetch_qos_network_status": ["qos_runtime_evidence"],
    "fetch_network_status": ["qos_runtime_evidence"],
    "inspect_mobility_ue_policies": ["ue_policy_context", "mobility_policy_context"],
    "inspect_ue_policies": ["ue_policy_context", "mobility_policy_context"],
}


def register() -> None:
    """Inject 6G-specific capability aliases into the generic runtime."""
    from agent_runtime.messages import register_capability_aliases

    register_capability_aliases(CONTROL_RUNTIME_CAPABILITY_ALIASES)


# Auto-register on first import of this module (side-effect is intentional —
# this is the application-layer injection point for domain-specific tool names).
register()


__all__ = ["CONTROL_RUNTIME_CAPABILITY_ALIASES", "register"]
