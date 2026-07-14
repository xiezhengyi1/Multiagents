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
from ..domain.collaboration import ExecutionReentryRequest, PlanningBlockerReport
from ..domain.control_loop import (
    build_conflict_request_payload,
    build_domain_feedback,
    split_domain_proposals,
    verdicts_from_conflict_result,
)
from ..domain.control_plane import ControlDomain, DomainStatus, DomainVerdict, GlobalControlIntent, UnifiedControlPlan
from ..domain.policy_plan import OperationIntent, PolicyPlanDraft
from ..integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi
from ..context import ControlRoundTrace, parse_pda_metrics


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
    planning_blocker: Optional[PlanningBlockerReport] = None
    execution_reentry: Optional[ExecutionReentryRequest] = None


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


def _global_intent_payload(global_intent: Optional[GlobalControlIntent]) -> Dict[str, Any]:
    return global_intent.model_dump(mode="json") if global_intent is not None else {}


def _operation_intent_payload(operation_intent: Optional[OperationIntent]) -> Dict[str, Any]:
    return operation_intent.model_dump(mode="json") if operation_intent is not None else {}


def _effective_supi(
    global_intent: Optional[GlobalControlIntent],
    operation_intent: Optional[OperationIntent],
    policy_plan: PolicyPlanDraft,
) -> str:
    if global_intent is not None and str(global_intent.supi or "").strip():
        return str(global_intent.supi or "").strip()
    if operation_intent is not None and str(operation_intent.supi or "").strip():
        return str(operation_intent.supi or "").strip()
    return str(policy_plan.supi or "").strip()


def _requested_domains(operation_intent: Optional[OperationIntent], policy_plan: PolicyPlanDraft) -> List[str]:
    if operation_intent is not None:
        domains = [
            str(item or "").strip().lower()
            for item in (operation_intent.requested_domains or [])
            if str(item or "").strip()
        ]
        if domains:
            return domains
    inferred: List[str] = []
    for item in policy_plan.all_policies or policy_plan.partial_policies or []:
        domain = "mobility" if item.policy_type == AM_POLICY_TYPE else "qos"
        if domain not in inferred:
            inferred.append(domain)
    return inferred


def _open_questions(operation_intent: Optional[OperationIntent], policy_plan: PolicyPlanDraft) -> List[Any]:
    if operation_intent is None:
        return list(policy_plan.open_questions)
    return list(operation_intent.open_questions + policy_plan.open_questions)


def _objective_breakdown(policy_plan: PolicyPlanDraft, global_intent: Optional[GlobalControlIntent]) -> Dict[str, Any]:
    if policy_plan.optimizer_result:
        return dict(policy_plan.optimizer_result)
    if global_intent is not None:
        return global_intent.objective_profile.model_dump(mode="json")
    return {}


def _planning_source_agent(global_intent: Optional[GlobalControlIntent]) -> str:
    return "optimization_strategy" if global_intent is not None else "single_control"


def _agent_contributions(
    *,
    global_intent: Optional[GlobalControlIntent],
    operation_intent: Optional[OperationIntent],
    policy_plan: PolicyPlanDraft,
) -> List[Dict[str, Any]]:
    if global_intent is None:
        return [
            {
                "agent": "single_control",
                "summary": str(policy_plan.planning_rationale.explanation or "").strip(),
                "payload": {"policy_plan": policy_plan.model_dump(mode="json")},
            }
        ]
    operation_payload = _operation_intent_payload(operation_intent)
    return [
        {"agent": "main_control", "summary": str(global_intent.routing_rationale or "").strip(), "payload": global_intent.model_dump(mode="json")},
        {
            "agent": "intent_encoding",
            "summary": str((operation_intent.domain_resolution if operation_intent is not None else "") or "").strip(),
            "payload": operation_payload,
        },
        {"agent": "optimization_strategy", "summary": str(policy_plan.planning_rationale.explanation or "").strip(), "payload": policy_plan.model_dump(mode="json")},
    ]


def _agent_conflicts(
    *,
    global_intent: Optional[GlobalControlIntent],
    operation_intent: Optional[OperationIntent],
) -> List[Dict[str, Any]]:
    if global_intent is None or operation_intent is None:
        return []
    if str(operation_intent.domain_resolution or "confirmed").strip() == "confirmed":
        return []
    return [
        {
            "agents": ["main_control", "intent_encoding"],
            "summary": str(operation_intent.domain_resolution or "").strip(),
            "impact": str(operation_intent.domain_resolution or "").strip(),
        }
    ]


def _handoff_records(
    *,
    global_intent: Optional[GlobalControlIntent],
    operation_intent: Optional[OperationIntent],
    policy_plan: PolicyPlanDraft,
) -> List[Dict[str, Any]]:
    if global_intent is None:
        return [
            {
                "source_agent": "single_control",
                "target_agent": "policy_dispatch",
                "artifact_type": "PolicyPlanDraft",
                "summary": str(policy_plan.planning_rationale.explanation or "").strip(),
            }
        ]
    return [
        {"source_agent": "main_control", "target_agent": "intent_encoding", "artifact_type": "GlobalControlIntent", "summary": str(global_intent.routing_decision or "").strip()},
        {"source_agent": "intent_encoding", "target_agent": "optimization_strategy", "artifact_type": "OperationIntent", "summary": str((operation_intent.domain_resolution if operation_intent is not None else "") or "").strip()},
    ]


def execute_planned_round(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    policy_plan: PolicyPlanDraft,
    cr_tool: ConflictResolutionTool,
    pd_agent: PolicyDispatchAgent,
    ad_tool: AssuranceDiagnosisTool,
    operation_intent: Optional[OperationIntent] = None,
    global_intent: Optional[GlobalControlIntent] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
) -> RoundExecutionArtifacts:
    snapshot_data = get_snapshot_data_by_id(snapshot_id) or {}
    if not snapshot_data:
        raise LookupError(f"bound snapshot not found: snapshot_id={snapshot_id}")
    planning_status = str(policy_plan.planning_status or "").strip().lower()
    if planning_status != "executable_plan":
        blocker = PlanningBlockerReport(
            round_index=round_index,
            source_agent=_planning_source_agent(global_intent),
            planning_status=planning_status or "needs_upstream_reground",
            missing_evidence=list(policy_plan.missing_evidence or []),
            blocked_targets=list(policy_plan.blocked_targets or []),
            upstream_requests=list(policy_plan.upstream_requests or []),
            planner_conflicts=list(policy_plan.planner_conflicts or []),
            recommended_consumers=[],
            summary=str(policy_plan.planning_rationale.explanation or "").strip(),
        )
        diagnosis = {
            "root_cause_category": "planning_blocked",
            "root_cause": "; ".join(
                policy_plan.upstream_requests
                or policy_plan.missing_evidence
                or policy_plan.blocked_targets
                or policy_plan.planner_conflicts
            ),
            "reason_summary": policy_plan.planning_rationale.explanation,
            "recommended_actions": list(
                policy_plan.upstream_requests
                or policy_plan.missing_evidence
                or policy_plan.blocked_targets
                or policy_plan.planner_conflicts
            ),
        }
        unified_plan = UnifiedControlPlan(
            session_id=session_id,
            snapshot_id=snapshot_id,
            supi=_effective_supi(global_intent, operation_intent, policy_plan),
            global_intent=global_intent,
            domain_verdicts=[],
            blocked_domains=[ControlDomain(item) for item in _requested_domains(operation_intent, policy_plan) if item in {ControlDomain.QOS.value, ControlDomain.MOBILITY.value}],
            objective_breakdown=_objective_breakdown(policy_plan, global_intent),
            open_questions=_open_questions(operation_intent, policy_plan),
        )
        trace = ControlRoundTrace(
            round_index=round_index,
            global_intent=_global_intent_payload(global_intent),
            operation_intent=_operation_intent_payload(operation_intent),
            policy_plan=policy_plan.model_dump(mode="json"),
            domain_verdicts=[],
            pda_feedback={},
            qos_feedback={},
            mobility_feedback={},
            diagnosis=diagnosis,
            planning_blocker=blocker.model_dump(mode="json"),
        )
        return RoundExecutionArtifacts(
            completed=False,
            report=None,
            diagnosis=diagnosis,
            qos_feedback={},
            mobility_feedback={},
            unified_plan=unified_plan,
            trace=trace,
            mediator_decision_payload={},
            domain_verdict_payloads=[],
            planning_blocker=blocker,
        )
    qos_proposal, mobility_proposal = split_domain_proposals(policy_plan)
    initial_verdicts: List[DomainVerdict] = []
    if qos_proposal:
        initial_verdicts.append(DomainVerdict(domain=ControlDomain.QOS, status=DomainStatus.PROPOSED, summary="QoS proposal ready for mediation."))
    if mobility_proposal:
        initial_verdicts.append(DomainVerdict(domain=ControlDomain.MOBILITY, status=DomainStatus.PROPOSED, summary="AM policy proposal ready for mediation."))

    conflict_result = cr_tool.run(
        ConflictResolutionRequest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            **build_conflict_request_payload(
                policy_plan=policy_plan,
                ue_context=get_ue_context_by_supi(_effective_supi(global_intent, operation_intent, policy_plan), snapshot_id=snapshot_id),
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
        report = pd_agent.execute_and_evaluate(policy_plan, trace_metadata=trace_metadata)
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
            "global_intent": _global_intent_payload(global_intent),
            "operation_intent": _operation_intent_payload(operation_intent),
            "conflict_result": conflict_result.model_dump(mode="json"),
        },
    )
    diagnosis = ad_tool.run(diagnosis_request).model_dump(mode="json")
    execution_reentry = None
    if report is not None and report.execution_status != "Success":
        feedback_payload = report.feedback_payload if isinstance(report.feedback_payload, dict) else {}
        execution_reentry = ExecutionReentryRequest(
            round_index=round_index,
            source_agent="policy_dispatch",
            recommended_consumers=[],
            target_bindings_at_risk=list(feedback_payload.get("target_bindings_at_risk") or []),
            policy_objects_at_risk=list(feedback_payload.get("policy_objects_at_risk") or []),
            reason_by_domain=dict(feedback_payload.get("reason_by_domain") or {}),
            failure_scope=str(report.failure_scope or "none"),
            failures=list(feedback_payload.get("failures") or []),
            summary=str(report.violation_details or diagnosis.get("reason_summary") or "").strip(),
        )

    unified_plan = UnifiedControlPlan(
        session_id=session_id,
        snapshot_id=snapshot_id,
        supi=_effective_supi(global_intent, operation_intent, policy_plan),
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
        objective_breakdown=_objective_breakdown(policy_plan, global_intent),
        control_churn_count=len(approved_policies(report)),
        agent_contributions=_agent_contributions(global_intent=global_intent, operation_intent=operation_intent, policy_plan=policy_plan),
        agent_conflicts=_agent_conflicts(global_intent=global_intent, operation_intent=operation_intent),
        handoff_records=_handoff_records(global_intent=global_intent, operation_intent=operation_intent, policy_plan=policy_plan),
        open_questions=[
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in _open_questions(operation_intent, policy_plan)
        ],
    )
    trace = ControlRoundTrace(
        round_index=round_index,
        global_intent=_global_intent_payload(global_intent),
        operation_intent=_operation_intent_payload(operation_intent),
        policy_plan=policy_plan.model_dump(mode="json"),
        domain_verdicts=[item.model_dump(mode="json") for item in domain_verdicts],
        pda_feedback=report.model_dump(mode="json") if report is not None else {},
        qos_feedback=qos_feedback,
        mobility_feedback=mobility_feedback,
        diagnosis=diagnosis,
        execution_reentry=execution_reentry.model_dump(mode="json") if execution_reentry is not None else {},
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
        execution_reentry=execution_reentry,
    )


__all__ = ["RoundExecutionArtifacts", "approved_policies", "execution_order", "execute_planned_round"]
