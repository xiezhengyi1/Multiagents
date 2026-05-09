from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import re
from typing import Any, Dict, List

from pydantic import BaseModel

from ...domain.policy_plan import PolicyDraft, PolicyPlanDraft
from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest


def json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_app_id(app_id: Any) -> str:
    text = str(app_id or "").strip()
    if not text:
        return ""
    if text.startswith(("app_", "app-")):
        return f"app-{text[4:].replace('_', '-')}"
    if re.fullmatch(r"app\d+", text, flags=re.IGNORECASE):
        return f"app-{text[3:]}"
    return text.replace("_", "-")


def _require_policy_id(policy_id: Any, *, policy_type: str) -> str:
    normalized = str(policy_id or "").strip()
    if not normalized:
        raise ValueError(f"{policy_type} is missing policy_id")
    return normalized


def _require_supi(supi: Any) -> str:
    normalized = str(supi or "").strip()
    if not normalized:
        raise ValueError("PolicyPlanDraft is missing authoritative supi")
    return normalized


def _normalize_sm_policy_details(details: Dict[str, Any], *, flow_id: str, app_id: str) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("SmPolicyDecision is missing policy_details")
    if not flow_id:
        raise ValueError("SmPolicyDecision is missing flow_id")
    if not app_id:
        raise ValueError("SmPolicyDecision is missing app_id")
    return json_friendly(SmPolicyDecision.model_validate(data))


def _normalize_ursp_policy_details(details: Dict[str, Any], *, target_type: str, flow_id: str | None) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("UrspRuleRequest is missing policy_details")
    if target_type == "flow" and not str(flow_id or "").strip():
        raise ValueError("flow-scoped URSP policy is missing flow_id")
    return json_friendly(UrspRuleRequest.model_validate(data))


def _normalize_am_policy_details(details: Dict[str, Any], *, supi: str) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("PcfAmPolicyControlPolicyAssociation is missing policy_details")
    request = data.get("request")
    if not isinstance(request, dict):
        raise ValueError("PcfAmPolicyControlPolicyAssociation requires a request object")
    request_supi = str(request.get("supi") or "").strip()
    if request_supi != supi:
        raise ValueError("AM policy request.supi must match authoritative supi")
    return json_friendly(PcfAmPolicyControlPolicyAssociation.model_validate(data))


def normalize_policy_plan_draft(draft: PolicyPlanDraft) -> PolicyPlanDraft:
    base_supi = _require_supi(draft.supi)
    normalized_policies: List[PolicyDraft] = []
    for index, policy in enumerate(draft.all_policies, start=1):
        policy_type = str(policy.policy_type or "").strip()
        if not policy_type:
            raise ValueError(f"Policy #{index} is missing policy_type")
        supi = _require_supi(policy.supi or base_supi)
        app_id = normalize_app_id(policy.app_id)
        flow_id = str(policy.flow_id or "").strip() or None
        target_type = str(policy.target_type or "").strip().lower()
        if not target_type:
            raise ValueError(f"Policy #{index} is missing target_type")

        if policy_type == "SmPolicyDecision":
            if target_type != "flow":
                raise ValueError("SmPolicyDecision target_type must be flow")
            norm_details = _normalize_sm_policy_details(policy.policy_details, flow_id=flow_id or "", app_id=app_id)
        elif policy_type == "UrspRuleRequest":
            if target_type not in {"flow", "app"}:
                raise ValueError("UrspRuleRequest target_type must be flow or app")
            norm_details = _normalize_ursp_policy_details(policy.policy_details, target_type=target_type, flow_id=flow_id)
        elif policy_type == "PcfAmPolicyControlPolicyAssociation":
            if target_type != "ue":
                raise ValueError("PcfAmPolicyControlPolicyAssociation target_type must be ue")
            if flow_id is not None:
                raise ValueError("PcfAmPolicyControlPolicyAssociation must not include flow_id")
            if app_id:
                raise ValueError("PcfAmPolicyControlPolicyAssociation must not include app_id")
            norm_details = _normalize_am_policy_details(policy.policy_details, supi=supi)
        else:
            raise ValueError(f"Unsupported policy_type: {policy_type}")

        normalized_policies.append(
            PolicyDraft(
                recommended_actions=[],
                supi=supi,
                app_id=app_id,
                flow_id=flow_id,
                target_type=target_type,
                policy_id=_require_policy_id(policy.policy_id, policy_type=policy_type),
                policy_type=policy_type,
                resource_keys=[str(item or "").strip() for item in (policy.resource_keys or []) if str(item or "").strip()],
                policy_details=norm_details,
            )
        )

    normalized_partial_policies: List[PolicyDraft] = []
    for index, policy in enumerate(draft.partial_policies, start=1):
        policy_type = str(policy.policy_type or "").strip()
        if not policy_type:
            raise ValueError(f"partial_policies[{index}] is missing policy_type")
        normalized_partial_policies.append(
            PolicyDraft(
                recommended_actions=[],
                supi=_require_supi(policy.supi or base_supi),
                app_id=normalize_app_id(policy.app_id),
                flow_id=str(policy.flow_id or "").strip() or None,
                target_type=str(policy.target_type or "").strip().lower() or "flow",
                policy_id=_require_policy_id(policy.policy_id, policy_type=policy_type),
                policy_type=policy_type,
                resource_keys=[str(item or "").strip() for item in (policy.resource_keys or []) if str(item or "").strip()],
                policy_details=json_friendly(policy.policy_details),
            )
        )

    return PolicyPlanDraft(
        supi=base_supi,
        session_id=str(draft.session_id or "").strip(),
        snapshot_id=str(draft.snapshot_id or "").strip(),
        planning_status=str(draft.planning_status or "executable_plan").strip(),
        planning_basis=json_friendly(draft.planning_basis),
        constraint_sources=json_friendly(draft.constraint_sources),
        optimizer_result=json_friendly(draft.optimizer_result),
        execution_writeback=json_friendly(draft.execution_writeback),
        all_policies=normalized_policies,
        partial_policies=normalized_partial_policies,
        missing_evidence=[str(item) for item in (draft.missing_evidence or []) if str(item or "").strip()],
        blocked_targets=[str(item) for item in (draft.blocked_targets or []) if str(item or "").strip()],
        upstream_requests=[str(item) for item in (draft.upstream_requests or []) if str(item or "").strip()],
        planner_conflicts=[str(item) for item in (draft.planner_conflicts or []) if str(item or "").strip()],
        agent_contributions=[item.model_copy(deep=True) for item in draft.agent_contributions],
        agent_conflicts=[item.model_copy(deep=True) for item in draft.agent_conflicts],
        handoff_records=[item.model_copy(deep=True) for item in draft.handoff_records],
        open_questions=[item.model_copy(deep=True) for item in draft.open_questions],
        planning_rationale=draft.planning_rationale.model_copy(deep=True),
    )


__all__ = [
    "json_friendly",
    "normalize_app_id",
    "normalize_policy_plan_draft",
]
