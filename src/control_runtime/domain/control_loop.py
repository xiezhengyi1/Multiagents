from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .control_plane import ControlDomain, DomainProposal, DomainStatus, DomainVerdict
from .policy_plan import PolicyPlanDraft


AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

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


def _build_resource_view(snapshot_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    snapshot = snapshot_data if isinstance(snapshot_data, dict) else {}
    slices = snapshot.get("slices") if isinstance(snapshot.get("slices"), list) else []
    apps = snapshot.get("apps") if isinstance(snapshot.get("apps"), list) else []

    valid_slice_keys: List[str] = []
    valid_snssai_keys: List[str] = []
    slice_capacity_by_snssai: Dict[str, Dict[str, float]] = {}
    flow_allocations: Dict[str, Dict[str, Any]] = {}

    for slice_item in slices:
        if not isinstance(slice_item, dict):
            continue
        snssai = str(slice_item.get("snssai") or "").strip()
        if not snssai:
            continue
        capacity = slice_item.get("capacity") if isinstance(slice_item.get("capacity"), dict) else {}
        load = slice_item.get("load") if isinstance(slice_item.get("load"), dict) else {}
        total_ul = float(capacity.get("total_bandwidth_ul", 0.0) or 0.0)
        total_dl = float(capacity.get("total_bandwidth_dl", 0.0) or 0.0)
        guaranteed_ul = float(capacity.get("guaranteed_bandwidth_ul", 0.0) or 0.0)
        guaranteed_dl = float(capacity.get("guaranteed_bandwidth_dl", 0.0) or 0.0)
        current_ul = float(load.get("current_bandwidth_ul", 0.0) or 0.0)
        current_dl = float(load.get("current_bandwidth_dl", 0.0) or 0.0)
        valid_slice_keys.append(f"slice:{snssai}")
        snssai_payload = {"sst": int(snssai[:2], 16), "sd": snssai[2:]} if len(snssai) >= 8 else None
        if snssai_payload is not None:
            valid_snssai_keys.append(f"snssai:{json.dumps(snssai_payload, sort_keys=True, ensure_ascii=False)}")
        slice_capacity_by_snssai[snssai] = {
            "total_ul": total_ul,
            "total_dl": total_dl,
            "guaranteed_ul": guaranteed_ul,
            "guaranteed_dl": guaranteed_dl,
            "current_ul": current_ul,
            "current_dl": current_dl,
            "remaining_ul": total_ul - current_ul,
            "remaining_dl": total_dl - current_dl,
        }

    for app in apps:
        if not isinstance(app, dict):
            continue
        for flow in app.get("flows", []) or []:
            if not isinstance(flow, dict):
                continue
            flow_id = str(flow.get("id") or flow.get("flow_id") or "").strip()
            if not flow_id:
                continue
            allocation = flow.get("allocation") if isinstance(flow.get("allocation"), dict) else {}
            flow_allocations[flow_id] = {
                "current_slice_snssai": str(allocation.get("current_slice_snssai") or "").strip(),
                "allocated_bandwidth_ul": float(allocation.get("allocated_bandwidth_ul", 0.0) or 0.0),
                "allocated_bandwidth_dl": float(allocation.get("allocated_bandwidth_dl", 0.0) or 0.0),
            }

    return {
        "valid_slice_keys": sorted(set(valid_slice_keys)),
        "valid_snssai_keys": sorted(set(valid_snssai_keys)),
        "slice_capacity_by_snssai": slice_capacity_by_snssai,
        "flow_allocations": flow_allocations,
        "exclusive_resource_keys": [],
    }


def build_conflict_request_payload(
    *,
    policy_plan: PolicyPlanDraft,
    ue_context: Optional[Dict[str, Any]],
    snapshot_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidate_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies]
    domains = {
        "mobility" if item.policy_type == AM_POLICY_TYPE else "qos"
        for item in policy_plan.all_policies
    }
    return {
        "candidate_policies": candidate_policies,
        "resource_view": _build_resource_view(snapshot_data),
        "conflict_scope": {"domains": sorted(domains)},
        "upstream_context": {
            "ue_context": ue_context or {},
            "planner_cross_domain_verdicts": list((policy_plan.optimizer_result or {}).get("cross_domain_verdicts") or []),
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
        "committed_snapshot_id": str(getattr(report, "committed_snapshot_id", "") or "").strip(),
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
