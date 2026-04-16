from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from domain.control_plane import ControlDomain, DomainProposal, DomainStatus, DomainVerdict, GlobalControlIntent
from domain.policy_plan import PolicyPlanDraft


AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"


def enforce_global_intent(user_input: str, intent: GlobalControlIntent) -> GlobalControlIntent:
    return intent


def split_domain_proposals(policy_plan: PolicyPlanDraft) -> Tuple[Optional[DomainProposal], Optional[DomainProposal]]:
    qos_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies if item.policy_type != AM_POLICY_TYPE]
    mobility_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies if item.policy_type == AM_POLICY_TYPE]

    qos_proposal = None
    if qos_policies:
        qos_proposal = DomainProposal(
            domain=ControlDomain.QOS,
            status=DomainStatus.READY,
            rationale="QoS planning generated via IEA + OSA chain.",
            evidence={"policy_count": len(qos_policies)},
            payload={"policy_plan": policy_plan.model_dump(mode="json")},
            policy_drafts=qos_policies,
        )

    mobility_proposal = None
    if mobility_policies:
        mobility_proposal = DomainProposal(
            domain=ControlDomain.MOBILITY,
            status=DomainStatus.READY,
            rationale="AM policy planning generated via IEA + OSA chain.",
            evidence={"policy_count": len(mobility_policies)},
            payload={"policy_plan": policy_plan.model_dump(mode="json")},
            policy_drafts=mobility_policies,
        )
    return qos_proposal, mobility_proposal


def build_conflict_request_payload(
    *,
    policy_plan: PolicyPlanDraft,
    ue_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies]
    domains = {
        "mobility" if item.policy_type == AM_POLICY_TYPE else "qos"
        for item in policy_plan.all_policies
    }
    return {
        "candidate_policies": candidate_policies,
        "resource_view": {},
        "conflict_scope": {"domains": sorted(domains)},
        "upstream_context": {
            "ue_context": ue_context or {},
            "planner_cross_domain_verdicts": list((policy_plan.planning_metadata or {}).get("optimizer_cross_domain_verdicts") or []),
        },
    }


def verdicts_from_conflict_result(conflict_result: Any, domains: List[ControlDomain]) -> List[DomainVerdict]:
    mediator_status = str(getattr(conflict_result, "mediator_status", "") or "").strip().lower()
    legacy_status = str(getattr(conflict_result, "status", "") or "").strip().lower()
    if mediator_status == "approved" and legacy_status in {"no_conflict", "resolved"}:
        return []
    if mediator_status == "reject":
        status = DomainStatus.REJECTED
    elif mediator_status == "revise" or legacy_status == "unresolved":
        status = DomainStatus.NEEDS_REVISION
    else:
        status = DomainStatus.APPROVED
    verdicts: List[DomainVerdict] = []
    for domain in domains:
        verdicts.append(
            DomainVerdict(
                domain=domain,
                status=status,
                summary=str(getattr(conflict_result, "reason_summary", "") or "").strip(),
                hard_conflicts=[json.dumps(item, ensure_ascii=False) for item in getattr(conflict_result, "conflicts", [])],
                metrics={"affected_policy_ids": getattr(conflict_result, "affected_policy_ids", [])},
            )
        )
    return verdicts


def build_domain_feedback(
    report: Any,
    *,
    dispatch_receipts: List[Dict[str, Any]],
    assurance_verdicts: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    qos_receipts = [item for item in dispatch_receipts if str(item.get("policy_type") or "").strip() != AM_POLICY_TYPE]
    mobility_receipts = [item for item in dispatch_receipts if str(item.get("policy_type") or "").strip() == AM_POLICY_TYPE]
    qos_verdicts = [item for item in assurance_verdicts if str(item.get("policy_type") or "").strip() != AM_POLICY_TYPE]
    mobility_verdicts = [item for item in assurance_verdicts if str(item.get("policy_type") or "").strip() == AM_POLICY_TYPE]

    qos_feedback = {
        "execution_status": "Success" if qos_receipts and all(item.get("status") == "success" for item in qos_receipts) else report.execution_status,
        "dispatch_receipts": qos_receipts,
        "assurance_verdicts": qos_verdicts,
        "failure_scope": str(report.failure_scope or "").strip(),
    }
    mobility_feedback = {
        "status": "success" if mobility_receipts and all(item.get("status") == "success" for item in mobility_receipts) else "failed",
        "dispatch_receipts": mobility_receipts,
        "assurance_verdicts": mobility_verdicts,
        "failure_scope": str(report.failure_scope or "").strip(),
    }
    if mobility_feedback["status"] != "success" and report.feedback_payload:
        mobility_feedback["error"] = str(report.feedback_payload.get("error") or report.violation_details or "").strip()
    return qos_feedback, mobility_feedback
