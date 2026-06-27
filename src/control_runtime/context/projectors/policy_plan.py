from __future__ import annotations

from typing import Any

from ...domain.policy_plan import PolicyPlanDraft
from .base import BaseProjector, field, json_mapping, without_empty_values


class PolicyPlanDraftProjector(BaseProjector):
    model = PolicyPlanDraft
    visible = (
        field("supi"),
        field("session_id"),
        field("snapshot_id"),
        field("planning_status"),
        field("missing_evidence"),
        field("blocked_targets"),
        field("upstream_requests"),
        field("planner_conflicts"),
        field("planning_rationale"),
    )

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        projected = super().project(raw)
        policies = [
            without_empty_values(
                {
                    "policy_id": policy.get("policy_id"),
                    "policy_type": policy.get("policy_type"),
                    "supi": policy.get("supi"),
                    "app_id": policy.get("app_id"),
                    "flow_id": policy.get("flow_id"),
                    "target_type": policy.get("target_type"),
                    "resource_keys": policy.get("resource_keys"),
                }
            )
            for policy in (raw.get("all_policies") or [])
            if isinstance(policy, dict)
        ]
        if policies:
            projected["policies"] = policies
        return projected
