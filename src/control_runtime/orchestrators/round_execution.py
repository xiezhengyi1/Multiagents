from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from ..agents.dispatch import PolicyDispatchAgent
from ..agents.dispatch.contracts import FeedbackReport
from ..diagnostics.diagnosis import AssuranceDiagnosisTool
from ..diagnostics.diagnosis.contracts import AssuranceDiagnosisRequest
from ..diagnostics.mediation import ConflictResolutionTool
from ..diagnostics.mediation.contracts import ConflictResolutionRequest
from ..domain.control_loop import (
    build_conflict_request_payload,
    build_domain_feedback,
    split_domain_proposals,
    verdicts_from_conflict_result,
)
from ..domain.control_plane import ControlDomain, DomainStatus, DomainVerdict, GlobalControlIntent, UnifiedControlPlan
from ..domain.policy_plan import OperationIntent, PolicyPlanDraft
from ..integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi
from .main_control_support import ControlRoundTrace, parse_pda_metrics


AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"


@dataclass
class RoundExecutionArtifacts:
    completed: bool
    report: Optional[FeedbackReport]
    diagnosis: Dict[str, Any]
    qos_feedback: Dict[str, Any]
    mobility_feedback: Dict[str, Any]
    unified_plan: UnifiedControlPlan
    trace: ControlRoundTrace
    mediator_decision_payload: Dict[str, Any]
    domain_verdict_payloads: List[Dict[str, Any]]


def execution_order(policy_plan: Optional[PolicyPlanDraft]) -> List[ControlDomain]:
    if policy_plan is None:
        return []
    order: List[ControlDomain] = []
    for item in policy_plan.all_policies:
        domain = ControlDomain.MOBILITY if item.policy_type == AM_POLICY_TYPE else ControlDomain.QOS
        if domain not in order:
            order.append(domain)
    return order


def approved_policies(report: Optional[FeedbackReport]) -> List[Dict[str, Any]]:
    if report is None:
        return []
    try:
        payload = json.loads(report.performance_metrics or "{}")
    except Exception:
        return []
    receipts = payload.get("dispatch_results", []) if isinstance(payload, dict) else []
    approved: List[Dict[str, Any]] = []
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("status") != "success":
            continue
        approved.append(
            {
                "domain": "mobility" if receipt.get("policy_type") == AM_POLICY_TYPE else "qos",
                "policy_id": receipt.get("policy_id"),
                "policy_type": receipt.get("policy_type"),
            }
        )
    return approved


def execute_planned_round(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    global_intent: GlobalControlIntent,
    operation_intent: OperationIntent,
    policy_plan: PolicyPlanDraft,
    cr_tool: ConflictResolutionTool,
    pd_agent: PolicyDispatchAgent,
    ad_tool: AssuranceDiagnosisTool,
) -> RoundExecutionArtifacts:
    snapshot_data = get_snapshot_data_by_id(snapshot_id) or {}
    if not snapshot_data:
        raise LookupError(f"bound snapshot not found: snapshot_id={snapshot_id}")
    qos_proposal, mobility_proposal = split_domain_proposals(policy_plan)
    initial_verdicts: List[DomainVerdict] = []
    if qos_proposal:
        initial_verdicts.append(DomainVerdict(domain=ControlDomain.QOS, status=DomainStatus.APPROVED, summary="QoS proposal ready."))
    if mobility_proposal:
        initial_verdicts.append(DomainVerdict(domain=ControlDomain.MOBILITY, status=DomainStatus.APPROVED, summary="AM policy proposal ready."))

    conflict_result = cr_tool.run(
        ConflictResolutionRequest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            **build_conflict_request_payload(
                policy_plan=policy_plan,
                ue_context=get_ue_context_by_supi(global_intent.supi or operation_intent.supi or "", snapshot_id=snapshot_id),
                snapshot_data=snapshot_data,
            ),
        )
    )
    conflict_verdicts = verdicts_from_conflict_result(conflict_result, execution_order(policy_plan))
    domain_verdicts = initial_verdicts + conflict_verdicts
    mediator_decision = conflict_result.to_mediator_decision()

    report: Optional[FeedbackReport] = None
    dispatch_receipts: List[Dict[str, Any]] = []
    assurance_verdicts: List[Dict[str, Any]] = []
    qos_feedback: Dict[str, Any] = {}
    mobility_feedback: Dict[str, Any] = {}

    if str(conflict_result.mediator_status or "").strip().lower() == "approved":
        report = pd_agent.execute_and_evaluate(policy_plan)
        dispatch_receipts, assurance_verdicts = parse_pda_metrics(report)
        qos_feedback, mobility_feedback = build_domain_feedback(
            report,
            dispatch_receipts=dispatch_receipts,
            assurance_verdicts=assurance_verdicts,
        )
        if qos_proposal is None:
            qos_feedback = {}
        if mobility_proposal is None:
            mobility_feedback = {}
        completed = report.execution_status == "Success"
    else:
        completed = False
        qos_feedback = {
            "execution_status": "Failed",
            "violation_details": str(conflict_result.reason_summary or "mediator blocked execution"),
            "revision_requests": conflict_result.revision_requests,
        }
        mobility_feedback = {
            "status": "failed",
            "error": str(conflict_result.reason_summary or "mediator blocked execution"),
            "revision_requests": conflict_result.revision_requests,
        }

    diagnosis_request = AssuranceDiagnosisRequest(
        execution_feedback={
            "pda": report.model_dump(mode="json") if report is not None else {},
            "qos": qos_feedback,
            "mobility": mobility_feedback,
        },
        dispatch_receipts=dispatch_receipts,
        assurance_verdicts=assurance_verdicts,
        telemetry_snapshot=snapshot_data,
        session_id=session_id,
        snapshot_id=snapshot_id,
        upstream_context={
            "global_intent": global_intent.model_dump(mode="json"),
            "operation_intent": operation_intent.model_dump(mode="json"),
            "conflict_result": conflict_result.model_dump(mode="json"),
        },
    )
    diagnosis = ad_tool.run(diagnosis_request).model_dump(mode="json")

    unified_plan = UnifiedControlPlan(
        session_id=session_id,
        snapshot_id=snapshot_id,
        supi=global_intent.supi or operation_intent.supi,
        global_intent=global_intent,
        qos_proposal=qos_proposal,
        mobility_proposal=mobility_proposal,
        domain_verdicts=domain_verdicts,
        mediator_decision=mediator_decision,
        unified_constraints=mediator_decision.unified_constraints,
        execution_order=execution_order(policy_plan),
        approved_policies=approved_policies(report),
        blocked_domains=[
            verdict.domain
            for verdict in domain_verdicts
            if verdict.status in {DomainStatus.REJECTED, DomainStatus.NEEDS_REVISION, DomainStatus.INCOMPLETE_CONTEXT, DomainStatus.FAILED}
        ],
        objective_breakdown=policy_plan.planning_metadata or global_intent.objective_profile.model_dump(mode="json"),
        control_churn_count=len(approved_policies(report)),
    )
    trace = ControlRoundTrace(
        round_index=round_index,
        global_intent=global_intent.model_dump(mode="json"),
        operation_intent=operation_intent.model_dump(mode="json"),
        policy_plan=policy_plan.model_dump(mode="json"),
        domain_verdicts=[item.model_dump(mode="json") for item in domain_verdicts],
        pda_feedback=report.model_dump(mode="json") if report is not None else {},
        qos_feedback=qos_feedback,
        mobility_feedback=mobility_feedback,
        diagnosis=diagnosis,
    )
    return RoundExecutionArtifacts(
        completed=completed,
        report=report,
        diagnosis=diagnosis,
        qos_feedback=qos_feedback,
        mobility_feedback=mobility_feedback,
        unified_plan=unified_plan,
        trace=trace,
        mediator_decision_payload=mediator_decision.model_dump(mode="json"),
        domain_verdict_payloads=[item.model_dump(mode="json") for item in domain_verdicts],
    )


__all__ = ["RoundExecutionArtifacts", "approved_policies", "execution_order", "execute_planned_round"]
