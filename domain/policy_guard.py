from __future__ import annotations

from typing import Any, Dict

from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest


class PolicyGuard:
    SUPPORTED_POLICY_TYPES = {"SmPolicyDecision", "UrspRuleRequest"}

    def validate_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(policy, dict):
            raise ValueError("compiled policy must be a JSON object")

        supi = str(policy.get("supi") or "").strip()
        app_id = str(policy.get("app_id") or "").strip()
        policy_id = str(policy.get("policy_id") or "").strip()
        policy_type = str(policy.get("policy_type") or "").strip()
        target_type = str(policy.get("target_type") or "").strip()
        flow_id = str(policy.get("flow_id") or "").strip()
        policy_details = policy.get("policy_details")

        if not supi:
            raise ValueError(f"policy {policy_id or '<unknown>'} missing supi")
        if not app_id:
            raise ValueError(f"policy {policy_id or '<unknown>'} missing app_id")
        if not policy_id:
            raise ValueError("policy_id is required")
        if policy_type not in self.SUPPORTED_POLICY_TYPES:
            raise ValueError(f"policy {policy_id} has unsupported policy_type={policy_type}")
        if not target_type:
            raise ValueError(f"policy {policy_id} missing target_type")
        if not isinstance(policy_details, dict):
            raise ValueError(f"policy {policy_id} missing policy_details object")

        nested_policy_id = str(policy_details.get("policy_id") or "").strip()
        if nested_policy_id and nested_policy_id != policy_id:
            raise ValueError(f"policy {policy_id} policy_details.policy_id does not match top-level policy_id")

        nested_target_type = str(policy_details.get("target_type") or "").strip()
        if nested_target_type and nested_target_type != target_type:
            raise ValueError(f"policy {policy_id} policy_details.target_type does not match top-level target_type")

        nested_flow_id = str(policy_details.get("flow_id") or "").strip()
        if nested_flow_id and flow_id and nested_flow_id != flow_id:
            raise ValueError(f"policy {policy_id} policy_details.flow_id does not match top-level flow_id")

        if target_type == "flow" and not flow_id:
            raise ValueError(f"flow-scoped policy {policy_id} missing flow_id")

        expected_prefix = "smp-" if policy_type == "SmPolicyDecision" else "ursp-"
        if not policy_id.startswith(expected_prefix):
            raise ValueError(f"policy {policy_id} must start with {expected_prefix}")

        if target_type == "flow":
            expected_policy_id = f"{expected_prefix}{app_id}-{flow_id}"
            if policy_id != expected_policy_id:
                raise ValueError(f"policy {policy_id} does not match canonical id {expected_policy_id}")

        if policy_type == "SmPolicyDecision":
            pcc_rules = policy_details.get("pccRules")
            qos_decs = policy_details.get("qosDecs")
            if not isinstance(pcc_rules, dict) or not pcc_rules:
                raise ValueError(f"policy {policy_id} missing non-empty pccRules")
            if not isinstance(qos_decs, dict) or not qos_decs:
                raise ValueError(f"policy {policy_id} missing non-empty qosDecs")
            SmPolicyDecision.model_validate(policy_details)
        else:
            route_sets = policy_details.get("routeSelParamSets")
            if not isinstance(route_sets, list) or not route_sets:
                raise ValueError(f"policy {policy_id} missing non-empty routeSelParamSets")
            if target_type == "flow" and not isinstance(policy_details.get("trafficDesc"), dict):
                raise ValueError(f"flow-scoped policy {policy_id} missing trafficDesc")
            UrspRuleRequest.model_validate(policy_details)

        return policy
