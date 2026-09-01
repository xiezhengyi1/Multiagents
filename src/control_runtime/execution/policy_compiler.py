from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest
from shared.contracts import ContractModel, ExecutionScope, JsonValue
from ..agents.planning.contracts import PolicyPlanDraft


def _json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class CompiledStrategyPlan(ContractModel):
    scope: ExecutionScope
    policies: list[dict[str, JsonValue]]


class PolicyCompiler:
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"
    SUPPORTED_POLICY_TYPES = {"SmPolicyDecision", "UrspRuleRequest", AM_POLICY_TYPE}

    @staticmethod
    def json_friendly(value: Any) -> Any:
        return _json_friendly(value)

    @staticmethod
    def strip_rule_prefix(candidate: Any) -> Optional[str]:
        text = str(candidate or "").strip()
        if not text:
            return None
        for prefix in ("pcc-", "qos-", "sess-", "smp-", "ursp-"):
            if text.startswith(prefix) and len(text) > len(prefix):
                return text[len(prefix) :]
        return text

    @classmethod
    def extract_flow_id(cls, policy_details: Any) -> Optional[str]:
        data: Dict[str, Any] = {}
        if isinstance(policy_details, dict):
            data = policy_details
        elif hasattr(policy_details, "model_dump"):
            dumped = policy_details.model_dump(mode="json")
            if isinstance(dumped, dict):
                data = dumped

        flow_id = cls.strip_rule_prefix(data.get("flow_id") or data.get("flowId"))
        if flow_id:
            return str(flow_id)

        qos_decs = data.get("qos_decs", data.get("qosDecs"))
        if isinstance(qos_decs, dict) and qos_decs:
            first_qos = next(iter(qos_decs.values()))
            if isinstance(first_qos, dict):
                qos_id = cls.strip_rule_prefix(first_qos.get("qosId") or first_qos.get("qos_id"))
                if qos_id:
                    return str(qos_id)
            first_qos_key = cls.strip_rule_prefix(next(iter(qos_decs.keys())))
            if first_qos_key:
                return str(first_qos_key)

        pcc_rules = data.get("pcc_rules", data.get("pccRules"))
        if isinstance(pcc_rules, dict) and pcc_rules:
            first_rule = next(iter(pcc_rules.values()))
            if isinstance(first_rule, dict):
                rule_id = cls.strip_rule_prefix(
                    first_rule.get("flow_id") or first_rule.get("flowId") or first_rule.get("pccRuleId")
                )
                if rule_id:
                    return str(rule_id)
            first_key = cls.strip_rule_prefix(next(iter(pcc_rules.keys())))
            if first_key:
                return str(first_key)
        return None

    @classmethod
    def coerce_strategy_output(cls, strategy_output: PolicyPlanDraft) -> CompiledStrategyPlan:
        return CompiledStrategyPlan(
            scope=strategy_output.scope,
            policies=[item.model_dump(mode="json") for item in strategy_output.all_policies],
        )

    @classmethod
    def compile_policy(cls, policy: Dict[str, Any]) -> Dict[str, Any]:
        policy_id = str(policy.get("policy_id") or "").strip()
        policy_type = str(policy.get("policy_type") or "").strip()
        app_id = str(policy.get("app_id") or "").strip()
        target_type = str(policy.get("target_type") or "").strip()
        policy_details = _json_friendly(policy.get("policy_details"))
        if not isinstance(policy_details, dict):
            raise ValueError(f"policy {policy_id or '<unknown>'} missing policy_details object")
        supi = str(policy.get("supi") or "").strip()
        flow_id = str(policy.get("flow_id") or "").strip()
        if not policy_id:
            raise ValueError("policy_id is required")
        if not policy_type:
            raise ValueError(f"policy {policy_id} missing policy_type")
        if not supi:
            raise ValueError(f"policy {policy_id} missing supi")
        if not target_type:
            raise ValueError(f"policy {policy_id} missing target_type")
        if policy_type != cls.AM_POLICY_TYPE and not flow_id:
            raise ValueError(f"policy {policy_id} missing flow_id")
        if policy_type != cls.AM_POLICY_TYPE and not app_id:
            raise ValueError(f"policy {policy_id} missing app_id")
        compiled = _json_friendly(
            {
                **policy,
                "supi": supi,
                "app_id": app_id,
                "policy_id": policy_id,
                "policy_type": policy_type,
                "target_type": target_type,
                "flow_id": flow_id,
                "policy_details": policy_details,
            }
        )
        if not isinstance(compiled, dict):
            raise ValueError("compiled policy must be a JSON object")
        cls._validate_compiled_policy(compiled)
        return compiled

    @classmethod
    def _validate_compiled_policy(cls, policy: Dict[str, Any]) -> None:
        supi = str(policy.get("supi") or "").strip()
        app_id = str(policy.get("app_id") or "").strip()
        policy_id = str(policy.get("policy_id") or "").strip()
        policy_type = str(policy.get("policy_type") or "").strip()
        target_type = str(policy.get("target_type") or "").strip()
        flow_id = str(policy.get("flow_id") or "").strip()
        policy_details = policy.get("policy_details")
        if not supi:
            raise ValueError(f"policy {policy_id or '<unknown>'} missing supi")
        if policy_type != cls.AM_POLICY_TYPE and not app_id:
            raise ValueError(f"policy {policy_id or '<unknown>'} missing app_id")
        if not policy_id:
            raise ValueError("policy_id is required")
        if policy_type not in cls.SUPPORTED_POLICY_TYPES:
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
        if policy_type == cls.AM_POLICY_TYPE:
            if target_type != "ue":
                raise ValueError(f"AM policy {policy_id} must use target_type=ue")
            request = policy_details.get("request")
            if not isinstance(request, dict):
                raise ValueError(f"AM policy {policy_id} missing request object")
            request_supi = str(request.get("supi") or "").strip()
            if request_supi and request_supi != supi:
                raise ValueError(f"AM policy {policy_id} request.supi does not match top-level supi")
            PcfAmPolicyControlPolicyAssociation.model_validate(policy_details)
            return
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
            return
        route_sets = policy_details.get("routeSelParamSets")
        if not isinstance(route_sets, list) or not route_sets:
            raise ValueError(f"policy {policy_id} missing non-empty routeSelParamSets")
        if target_type == "flow" and not isinstance(policy_details.get("trafficDesc"), dict):
            raise ValueError(f"flow-scoped policy {policy_id} missing trafficDesc")
        UrspRuleRequest.model_validate(policy_details)

    @classmethod
    def compile_plan(cls, strategy_output: PolicyPlanDraft) -> CompiledStrategyPlan:
        plan = cls.coerce_strategy_output(strategy_output)
        compiled_policies = [cls.compile_policy(policy) for policy in plan.policies]
        if not compiled_policies:
            raise ValueError("strategy_output contains no policies")
        return CompiledStrategyPlan(
            scope=plan.scope,
            policies=compiled_policies,
        )

__all__ = ["CompiledStrategyPlan", "PolicyCompiler"]
