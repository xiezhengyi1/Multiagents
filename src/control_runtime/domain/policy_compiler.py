from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


def _json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class CompiledStrategyPlan:
    supi: str
    session_id: str
    snapshot_id: str
    policies: List[Dict[str, Any]]


class PolicyCompiler:
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

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
    def coerce_strategy_output(cls, strategy_output: Any) -> CompiledStrategyPlan:
        payload = _json_friendly(strategy_output)
        if not isinstance(payload, dict):
            raise ValueError("strategy_output must be a JSON object")

        policies = payload.get("all_policies")
        if not isinstance(policies, list):
            raise ValueError("strategy_output.all_policies must be a list")

        return CompiledStrategyPlan(
            supi=str(payload.get("supi") or "").strip(),
            session_id=str(payload.get("session_id") or "").strip(),
            snapshot_id=str(payload.get("snapshot_id") or "").strip(),
            policies=[_json_friendly(item) for item in policies if isinstance(item, dict)],
        )

    @classmethod
    def compile_policy(cls, policy: Dict[str, Any], top_level_supi: str) -> Dict[str, Any]:
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

        return _json_friendly(
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

    @classmethod
    def compile_plan(cls, strategy_output: Any) -> CompiledStrategyPlan:
        plan = cls.coerce_strategy_output(strategy_output)
        compiled_policies = [cls.compile_policy(policy, plan.supi) for policy in plan.policies]
        if not compiled_policies:
            raise ValueError("strategy_output contains no policies")
        return CompiledStrategyPlan(
            supi=plan.supi,
            session_id=plan.session_id,
            snapshot_id=plan.snapshot_id,
            policies=compiled_policies,
        )

    @staticmethod
    def _merge_bucket(existing_bucket: Any, group_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = _json_friendly(existing_bucket) if isinstance(existing_bucket, dict) else {}
        normalized[group_key] = payload
        return normalized

    @classmethod
    def merge_policies_into_context(cls, existing_ctx: Dict[str, Any], policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        sm_policy_data = _json_friendly(existing_ctx.get("smPolicyData")) if isinstance(existing_ctx, dict) else {}
        pcc_rules = _json_friendly(existing_ctx.get("pccRules")) if isinstance(existing_ctx, dict) else {}
        qos_decs = _json_friendly(existing_ctx.get("qosDecs")) if isinstance(existing_ctx, dict) else {}
        sess_rules = _json_friendly(existing_ctx.get("sessRules")) if isinstance(existing_ctx, dict) else {}
        traff_cont_decs = _json_friendly(existing_ctx.get("traffContDecs")) if isinstance(existing_ctx, dict) else {}
        chg_decs = _json_friendly(existing_ctx.get("chgDecs")) if isinstance(existing_ctx, dict) else {}
        ursp_rules = _json_friendly(existing_ctx.get("urspRules")) if isinstance(existing_ctx, dict) else {}

        sm_policy_data = sm_policy_data if isinstance(sm_policy_data, dict) else {}
        pcc_rules = pcc_rules if isinstance(pcc_rules, dict) else {}
        qos_decs = qos_decs if isinstance(qos_decs, dict) else {}
        sess_rules = sess_rules if isinstance(sess_rules, dict) else {}
        traff_cont_decs = traff_cont_decs if isinstance(traff_cont_decs, dict) else {}
        chg_decs = chg_decs if isinstance(chg_decs, dict) else {}
        ursp_rules = ursp_rules if isinstance(ursp_rules, dict) else {}

        for policy in policies:
            policy_details = _json_friendly(policy.get("policy_details"))
            if not isinstance(policy_details, dict):
                continue

            policy_id = str(policy.get("policy_id") or policy_details.get("policy_id") or "").strip()
            if not policy_id:
                continue

            policy_type = str(policy.get("policy_type") or "").strip()
            if policy_type == "SmPolicyDecision":
                sm_policy_data[policy_id] = {
                    "policy_id": policy_id,
                    "policy_type": policy_type,
                    "policy_details": policy_details,
                }
                if isinstance(policy_details.get("pccRules"), dict):
                    pcc_rules = cls._merge_bucket(pcc_rules, policy_id, policy_details["pccRules"])
                if isinstance(policy_details.get("qosDecs"), dict):
                    qos_decs = cls._merge_bucket(qos_decs, policy_id, policy_details["qosDecs"])
                if isinstance(policy_details.get("sessRules"), dict):
                    sess_rules = cls._merge_bucket(sess_rules, policy_id, policy_details["sessRules"])
                if isinstance(policy_details.get("traffContDecs"), dict):
                    traff_cont_decs = cls._merge_bucket(traff_cont_decs, policy_id, policy_details["traffContDecs"])
                if isinstance(policy_details.get("chgDecs"), dict):
                    chg_decs = cls._merge_bucket(chg_decs, policy_id, policy_details["chgDecs"])
            elif policy_type == "UrspRuleRequest":
                ursp_rules[policy_id] = policy_details
            elif policy_type == cls.AM_POLICY_TYPE:
                continue

        return {
            "sm_policy_data": sm_policy_data,
            "pcc_rules": pcc_rules,
            "qos_decs": qos_decs,
            "sess_rules": sess_rules,
            "traff_cont_decs": traff_cont_decs,
            "chg_decs": chg_decs,
            "ursp_rules": ursp_rules,
        }
