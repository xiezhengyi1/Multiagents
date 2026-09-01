"""Mediation, dispatch, assurance, and trace assembly for one control round."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from shared.contracts import ExecutionScope

from ..agents.dispatch.agent import PolicyDispatchAgent
from ..agents.dispatch.contracts import (
    DispatchFailureFeedback,
    FeedbackReport,
    PartialIntentFeedback,
)
from ..agents.grounding.contracts import GroundingDecision
from ..agents.main.contracts import ControlDomain, GlobalControlIntent
from ..agents.planning.contracts import PolicyPlanDraft
from ..diagnostics.diagnosis import AssuranceDiagnosisTool
from ..diagnostics.diagnosis.contracts import (
    AssuranceDiagnosisRequest,
    DiagnosisExecutionFeedback,
    DiagnosisUpstreamContext,
)
from ..diagnostics.mediation import ConflictResolutionTool
from ..diagnostics.mediation.contracts import (
    ConflictResolutionRequest,
    ConflictResolutionResult,
    ConflictResourceView,
    ConflictScope,
    ConflictUpstreamContext,
    DomainStatus,
    DomainVerdict,
    MediatorDecision,
    UnifiedConstraintSet,
)
from ..execution.contracts import CompileFailureMetrics
from ..integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi
from .contracts import (
    ControlRoundTrace,
    DomainProposal,
    ExecutionReentryRequest,
    PlanningBlockerReport,
    UnifiedControlPlan,
)


AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"


@dataclass
class RoundExecutionArtifacts:
    """Stable result of a fully planned control round."""

    completed: bool
    report: Optional[FeedbackReport]
    diagnosis: Dict[str, Any]
    qos_feedback: Dict[str, Any]
    mobility_feedback: Dict[str, Any]
    unified_plan: UnifiedControlPlan
    trace: ControlRoundTrace
    planning_blocker: Optional[PlanningBlockerReport] = None
    execution_reentry: Optional[ExecutionReentryRequest] = None

@dataclass(frozen=True)
class _MediationResult:
    qos_proposal: Optional[DomainProposal]
    mobility_proposal: Optional[DomainProposal]
    domain_verdicts: List[DomainVerdict]
    mediator_decision: MediatorDecision
    conflict_result: ConflictResolutionResult


@dataclass(frozen=True)
class _DispatchResult:
    completed: bool
    report: Optional[FeedbackReport]
    dispatch_receipts: List[Dict[str, Any]]
    assurance_verdicts: List[Dict[str, Any]]
    qos_feedback: Dict[str, Any]
    mobility_feedback: Dict[str, Any]
    approved_policies: List[Dict[str, Any]]


@dataclass(frozen=True)
class _AssuranceResult:
    diagnosis: Dict[str, Any]
    execution_reentry: Optional[ExecutionReentryRequest]


def _split_domain_proposals(policy_plan: PolicyPlanDraft) -> tuple[Optional[DomainProposal], Optional[DomainProposal]]:
    qos_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies if item.policy_type != AM_POLICY_TYPE]
    mobility_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies if item.policy_type == AM_POLICY_TYPE]
    qos_proposal = (
        DomainProposal(
            domain=ControlDomain.QOS,
            status=DomainStatus.READY,
            rationale="QoS planning generated via IEA + OSA chain.",
            evidence={"policy_count": len(qos_policies)},
            payload={"policy_plan": policy_plan.model_dump(mode="json")},
            policy_drafts=qos_policies,
        )
        if qos_policies
        else None
    )
    mobility_proposal = (
        DomainProposal(
            domain=ControlDomain.MOBILITY,
            status=DomainStatus.READY,
            rationale="AM policy planning generated via IEA + OSA chain.",
            evidence={"policy_count": len(mobility_policies)},
            payload={"policy_plan": policy_plan.model_dump(mode="json")},
            policy_drafts=mobility_policies,
        )
        if mobility_policies
        else None
    )
    return qos_proposal, mobility_proposal


def _build_resource_view(snapshot_data: Dict[str, Any]) -> ConflictResourceView:
    snapshot = snapshot_data
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
        if len(snssai) >= 8:
            valid_snssai_keys.append(
                f"snssai:{json.dumps({'sst': int(snssai[:2], 16), 'sd': snssai[2:]}, sort_keys=True, ensure_ascii=False)}"
            )
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
    return ConflictResourceView(
        valid_slice_keys=sorted(set(valid_slice_keys)),
        valid_snssai_keys=sorted(set(valid_snssai_keys)),
        slice_capacity_by_snssai=slice_capacity_by_snssai,
        flow_allocations=flow_allocations,
        exclusive_resource_keys=[],
    )


def _effective_supi(
    global_intent: Optional[GlobalControlIntent],
    grounding_decision: Optional[GroundingDecision],
    policy_plan: PolicyPlanDraft,
) -> str:
    if global_intent is not None and global_intent.scope.supi is not None:
        return global_intent.scope.supi
    if grounding_decision is not None:
        for flow in grounding_decision.flows or []:
            if str(flow.supi or "").strip():
                return str(flow.supi).strip()
    return policy_plan.scope.supi or ""


def _execution_order(policy_plan: Optional[PolicyPlanDraft]) -> List[ControlDomain]:
    if policy_plan is None:
        return []
    order: List[ControlDomain] = []
    for item in policy_plan.all_policies:
        domain = ControlDomain.MOBILITY if item.policy_type == AM_POLICY_TYPE else ControlDomain.QOS
        if domain not in order:
            order.append(domain)
    return order


def _build_conflict_resolution_request(
    *,
    policy_plan: PolicyPlanDraft,
    ue_context: Dict[str, Any],
    snapshot_data: Dict[str, Any],
) -> ConflictResolutionRequest:
    candidate_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies]
    domains = {
        ControlDomain.MOBILITY if item.policy_type == AM_POLICY_TYPE else ControlDomain.QOS
        for item in policy_plan.all_policies
    }
    return ConflictResolutionRequest(
        scope=policy_plan.scope,
        candidate_policies=candidate_policies,
        resource_view=_build_resource_view(snapshot_data),
        conflict_scope=ConflictScope(domains=sorted(domains, key=lambda item: item.value)),
        upstream_context=ConflictUpstreamContext(
            ue_context=ue_context,
            planner_cross_domain_verdicts=policy_plan.optimizer_result.cross_domain_verdicts,
        ),
    )


def _verdicts_from_conflict_result(
    conflict_result: ConflictResolutionResult,
    domains: List[ControlDomain],
) -> List[DomainVerdict]:
    mediator_status = conflict_result.mediator_status
    if mediator_status == "approved":
        return []
    status = (
        DomainStatus.REJECTED
        if mediator_status == "reject"
        else DomainStatus.NEEDS_REVISION
        if mediator_status == "revise"
        else DomainStatus.APPROVED
    )
    return [
        DomainVerdict(
            domain=domain,
            status=status,
            summary=conflict_result.reason_summary,
            hard_conflicts=[item.model_dump_json() for item in conflict_result.conflicts],
            soft_conflicts=[],
            infeasible_reasons=[],
            metrics={"affected_policy_ids": conflict_result.affected_policy_ids},
        )
        for domain in domains
    ]


def _mediate_policy_plan(
    *,
    scope: ExecutionScope,
    snapshot_data: Dict[str, Any],
    policy_plan: PolicyPlanDraft,
    cr_tool: ConflictResolutionTool,
    grounding_decision: Optional[GroundingDecision],
    global_intent: Optional[GlobalControlIntent],
) -> _MediationResult:
    qos_proposal, mobility_proposal = _split_domain_proposals(policy_plan)
    initial_verdicts: List[DomainVerdict] = []
    if qos_proposal:
        initial_verdicts.append(
            DomainVerdict(
                domain=ControlDomain.QOS,
                status=DomainStatus.PROPOSED,
                summary="QoS proposal ready for mediation.",
                hard_conflicts=[],
                soft_conflicts=[],
                infeasible_reasons=[],
                metrics={},
            )
        )
    if mobility_proposal:
        initial_verdicts.append(
            DomainVerdict(
                domain=ControlDomain.MOBILITY,
                status=DomainStatus.PROPOSED,
                summary="AM policy proposal ready for mediation.",
                hard_conflicts=[],
                soft_conflicts=[],
                infeasible_reasons=[],
                metrics={},
            )
        )
    if scope.snapshot_id is None:
        raise ValueError("mediation requires scope.snapshot_id")
    conflict_result = cr_tool.run(
        _build_conflict_resolution_request(
            policy_plan=policy_plan,
            ue_context=get_ue_context_by_supi(
                _effective_supi(global_intent, grounding_decision, policy_plan),
                snapshot_id=scope.snapshot_id,
            )
            or {},
            snapshot_data=snapshot_data,
        )
    )
    domain_verdicts = initial_verdicts + _verdicts_from_conflict_result(conflict_result, _execution_order(policy_plan))
    return _MediationResult(
        qos_proposal=qos_proposal,
        mobility_proposal=mobility_proposal,
        domain_verdicts=domain_verdicts,
        mediator_decision=conflict_result.to_mediator_decision(),
        conflict_result=conflict_result,
    )


def _build_domain_feedback(
    report: FeedbackReport,
    *,
    dispatch_receipts: List[Dict[str, Any]],
    assurance_verdicts: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    qos_receipts = [item for item in dispatch_receipts if str(item.get("policy_type") or "").strip() != AM_POLICY_TYPE]
    mobility_receipts = [item for item in dispatch_receipts if str(item.get("policy_type") or "").strip() == AM_POLICY_TYPE]
    qos_verdicts = [item for item in assurance_verdicts if str(item.get("policy_type") or "").strip() != AM_POLICY_TYPE]
    mobility_verdicts = [item for item in assurance_verdicts if str(item.get("policy_type") or "").strip() == AM_POLICY_TYPE]
    qos_feedback = {
        "execution_status": "Success" if qos_receipts and all(item.get("status") == "success" for item in qos_receipts) else report.execution_status,
        "intent_status": str(report.intent_status or "satisfied"),
        "dispatch_receipts": qos_receipts,
        "assurance_verdicts": qos_verdicts,
        "failure_scope": str(report.failure_scope or "").strip(),
    }
    mobility_feedback = {
        "status": "success" if mobility_receipts and all(item.get("status") == "success" for item in mobility_receipts) else "failed",
        "intent_status": str(report.intent_status or "satisfied"),
        "dispatch_receipts": mobility_receipts,
        "assurance_verdicts": mobility_verdicts,
        "failure_scope": str(report.failure_scope or "").strip(),
    }
    if mobility_feedback["status"] != "success" and isinstance(
        report.feedback_payload,
        DispatchFailureFeedback,
    ):
        mobility_feedback["error"] = report.feedback_payload.reason
    return qos_feedback, mobility_feedback


def _parse_pda_metrics(report: FeedbackReport) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics = report.performance_metrics
    if isinstance(metrics, CompileFailureMetrics):
        return [], [], []
    dispatch_receipts = metrics.dispatch_results
    assurance_verdicts = metrics.assurance_results
    approved_policies = [
        {
            "domain": "mobility" if receipt.get("policy_type") == AM_POLICY_TYPE else "qos",
            "policy_id": receipt.get("policy_id"),
            "policy_type": receipt.get("policy_type"),
        }
        for receipt in dispatch_receipts
        if receipt.get("status") == "success"
    ]
    return dispatch_receipts, assurance_verdicts, approved_policies


def _dispatch_mediated_plan(
    *,
    policy_plan: PolicyPlanDraft,
    planning_status: str,
    pd_agent: PolicyDispatchAgent,
    conflict_result: ConflictResolutionResult,
    qos_proposal: Optional[DomainProposal],
    mobility_proposal: Optional[DomainProposal],
    trace_metadata: Optional[Dict[str, Any]],
) -> _DispatchResult:
    report: Optional[FeedbackReport] = None
    dispatch_receipts: List[Dict[str, Any]] = []
    assurance_verdicts: List[Dict[str, Any]] = []
    qos_feedback: Dict[str, Any] = {}
    mobility_feedback: Dict[str, Any] = {}

    if str(conflict_result.mediator_status or "").strip().lower() == "approved":
        report = pd_agent.execute_and_evaluate(policy_plan, trace_metadata=trace_metadata)
        if planning_status == "partial_plan" and report.execution_status == "Success":
            intent_blockers = list(
                policy_plan.blocked_targets
                or policy_plan.planner_conflicts
                or policy_plan.missing_evidence
            )
            partial_summary = (
                "Policy dispatch succeeded, but the requested intent is only partially satisfied: "
                + "; ".join(intent_blockers or ["a constrained best-effort policy was delivered"])
            )
            prior_details = str(report.violation_details or "").strip()
            if prior_details and prior_details.lower() != "none":
                partial_summary = f"{prior_details}; {partial_summary}"
            feedback_payload = PartialIntentFeedback(
                intent_blockers=intent_blockers,
                upstream_requests=list(policy_plan.upstream_requests),
            )
            report = FeedbackReport.model_validate(
                {
                    **report.model_dump(mode="json"),
                    "intent_status": "partially_satisfied",
                    "violation_details": partial_summary,
                    "feedback_payload": feedback_payload,
                }
            )
        dispatch_receipts, assurance_verdicts, approved_policies = _parse_pda_metrics(report)
        qos_feedback, mobility_feedback = _build_domain_feedback(
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
        approved_policies = []
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

    return _DispatchResult(
        completed=completed,
        report=report,
        dispatch_receipts=dispatch_receipts,
        assurance_verdicts=assurance_verdicts,
        qos_feedback=qos_feedback,
        mobility_feedback=mobility_feedback,
        approved_policies=approved_policies,
    )


def _diagnose_round(
    *,
    scope: ExecutionScope,
    snapshot_data: Dict[str, Any],
    policy_plan: PolicyPlanDraft,
    global_intent: Optional[GlobalControlIntent],
    grounding_decision: Optional[GroundingDecision],
    conflict_result: ConflictResolutionResult,
    report: Optional[FeedbackReport],
    qos_feedback: Dict[str, Any],
    mobility_feedback: Dict[str, Any],
    dispatch_receipts: List[Dict[str, Any]],
    assurance_verdicts: List[Dict[str, Any]],
    ad_tool: AssuranceDiagnosisTool,
) -> _AssuranceResult:
    diagnosis_request = AssuranceDiagnosisRequest(
        scope=scope,
        execution_feedback=DiagnosisExecutionFeedback(
            pda=report.model_dump(mode="json") if report is not None else {},
            qos=qos_feedback,
            mobility=mobility_feedback,
        ),
        dispatch_receipts=dispatch_receipts,
        assurance_verdicts=assurance_verdicts,
        telemetry_snapshot=snapshot_data,
        upstream_context=DiagnosisUpstreamContext(
            global_intent=global_intent.model_dump(mode="json") if global_intent is not None else {},
            grounding_decision=grounding_decision.model_dump(mode="json") if grounding_decision is not None else {},
            conflict_result=conflict_result.model_dump(mode="json"),
        ),
    )
    diagnosis = ad_tool.run(diagnosis_request).model_dump(mode="json")
    if report is not None and report.intent_status == "partially_satisfied":
        partial_summary = str(report.violation_details or "").strip()
        diagnosis.update(
            {
                "root_cause_category": "intent_partially_satisfied",
                "root_cause": "; ".join(policy_plan.blocked_targets or policy_plan.planner_conflicts),
                "reason_summary": partial_summary,
                "recommended_actions": list(policy_plan.upstream_requests or []),
            }
        )
    execution_reentry = None
    if report is not None and report.execution_status != "Success":
        feedback_payload = report.feedback_payload
        if not isinstance(feedback_payload, DispatchFailureFeedback):
            raise TypeError("non-successful FeedbackReport requires DispatchFailureFeedback")
        if not feedback_payload.retry_forbidden:
            execution_reentry = ExecutionReentryRequest(
                round_index=scope.round_index,
                source_agent="policy_dispatch",
                recommended_consumers=[],
                target_bindings_at_risk=feedback_payload.target_bindings_at_risk,
                policy_objects_at_risk=feedback_payload.policy_objects_at_risk,
                reason_by_domain=feedback_payload.reason_by_domain,
                failure_scope=report.failure_scope,
                failures=feedback_payload.failures,
                summary=str(report.violation_details or diagnosis.get("reason_summary") or "").strip(),
            )
    return _AssuranceResult(diagnosis=diagnosis, execution_reentry=execution_reentry)


def _requested_domains(global_intent: Optional[GlobalControlIntent], policy_plan: PolicyPlanDraft) -> List[str]:
    if global_intent is not None and global_intent.requested_domains:
        return [item.value for item in global_intent.requested_domains]
    inferred: List[str] = []
    for item in policy_plan.all_policies or policy_plan.partial_policies or []:
        domain = "mobility" if item.policy_type == AM_POLICY_TYPE else "qos"
        if domain not in inferred:
            inferred.append(domain)
    return inferred


def _open_questions(grounding_decision: Optional[GroundingDecision], policy_plan: PolicyPlanDraft) -> List[Any]:
    if grounding_decision is None:
        return list(policy_plan.open_questions)
    return list(grounding_decision.open_questions + policy_plan.open_questions)


def _objective_breakdown(policy_plan: PolicyPlanDraft, global_intent: Optional[GlobalControlIntent]) -> Dict[str, Any]:
    if policy_plan.optimizer_result.objective_breakdown:
        return dict(policy_plan.optimizer_result.objective_breakdown)
    if global_intent is not None:
        return global_intent.objective_profile.model_dump(mode="json")
    return {}


def _build_planning_blocked_artifacts(
    *,
    scope: ExecutionScope,
    policy_plan: PolicyPlanDraft,
    grounding_decision: Optional[GroundingDecision],
    global_intent: Optional[GlobalControlIntent],
    planning_status: str,
) -> RoundExecutionArtifacts:
    snapshot_id = scope.snapshot_id
    round_index = scope.round_index
    if snapshot_id is None:
        raise ValueError("planning artifacts require scope.snapshot_id")
    blocker = PlanningBlockerReport(
        round_index=round_index,
        source_agent="optimization_strategy" if global_intent is not None else "single_control",
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
        scope=scope.model_copy(
            update={"supi": _effective_supi(global_intent, grounding_decision, policy_plan) or None}
        ),
        global_intent=global_intent,
        qos_proposal=None,
        mobility_proposal=None,
        domain_verdicts=[],
        mediator_decision=None,
        unified_constraints=UnifiedConstraintSet(hard_constraints=[], evidence=[]),
        execution_order=[],
        approved_policies=[],
        blocked_domains=[
            ControlDomain(item)
            for item in _requested_domains(global_intent, policy_plan)
            if item in {ControlDomain.QOS.value, ControlDomain.MOBILITY.value}
        ],
        objective_breakdown=_objective_breakdown(policy_plan, global_intent),
        control_churn_count=0,
        agent_contributions=[],
        handoff_records=[],
        open_questions=_open_questions(grounding_decision, policy_plan),
    )
    return RoundExecutionArtifacts(
        completed=False,
        report=None,
        diagnosis=diagnosis,
        qos_feedback={},
        mobility_feedback={},
        unified_plan=unified_plan,
        trace=ControlRoundTrace(
            scope=scope,
            round_index=round_index,
            global_intent=global_intent,
            grounding_decision=grounding_decision,
            policy_plan=policy_plan,
            domain_verdicts=[],
            qos_feedback={},
            mobility_feedback={},
            diagnosis=diagnosis,
            planning_blocker=blocker,
        ),
        planning_blocker=blocker,
    )


def _assemble_round_artifacts(
    *,
    scope: ExecutionScope,
    policy_plan: PolicyPlanDraft,
    global_intent: Optional[GlobalControlIntent],
    grounding_decision: Optional[GroundingDecision],
    mediation: _MediationResult,
    dispatch: _DispatchResult,
    assurance: _AssuranceResult,
) -> RoundExecutionArtifacts:
    round_index = scope.round_index
    if global_intent is None:
        agent_contributions = [
            {
                "agent": "single_control",
                "summary": str(policy_plan.planning_rationale.explanation or "").strip(),
                "payload": {"policy_plan": policy_plan.model_dump(mode="json")},
            }
        ]
        handoff_records = [
            {
                "source_agent": "single_control",
                "target_agent": "policy_dispatch",
                "artifact_type": "PolicyPlanDraft",
                "summary": str(policy_plan.planning_rationale.explanation or "").strip(),
            }
        ]
    else:
        agent_contributions = [
            {
                "agent": "main_control",
                "summary": str(global_intent.routing_rationale or "").strip(),
                "payload": global_intent.model_dump(mode="json"),
            },
            {
                "agent": "intent_encoding",
                "summary": f"grounded_flows={len(grounding_decision.flows) if grounding_decision is not None else 0}",
                "payload": grounding_decision.model_dump(mode="json") if grounding_decision is not None else {},
            },
            {
                "agent": "optimization_strategy",
                "summary": str(policy_plan.planning_rationale.explanation or "").strip(),
                "payload": policy_plan.model_dump(mode="json"),
            },
        ]
        handoff_records = [
            {
                "source_agent": "main_control",
                "target_agent": "intent_encoding",
                "artifact_type": "GlobalControlIntent",
                "summary": str(global_intent.routing_decision or "").strip(),
            },
            {
                "source_agent": "intent_encoding",
                "target_agent": "optimization_strategy",
                "artifact_type": "GroundingDecision",
                "summary": f"grounded_flows={len(grounding_decision.flows) if grounding_decision is not None else 0}",
            },
        ]
    unified_plan = UnifiedControlPlan(
        scope=scope.model_copy(
            update={"supi": _effective_supi(global_intent, grounding_decision, policy_plan) or None}
        ),
        global_intent=global_intent,
        qos_proposal=mediation.qos_proposal,
        mobility_proposal=mediation.mobility_proposal,
        domain_verdicts=mediation.domain_verdicts,
        mediator_decision=mediation.mediator_decision,
        unified_constraints=mediation.mediator_decision.unified_constraints,
        execution_order=_execution_order(policy_plan),
        approved_policies=dispatch.approved_policies,
        blocked_domains=[
            verdict.domain
            for verdict in mediation.domain_verdicts
            if verdict.status
            in {
                DomainStatus.REJECTED,
                DomainStatus.NEEDS_REVISION,
                DomainStatus.INCOMPLETE_CONTEXT,
                DomainStatus.FAILED,
            }
        ],
        objective_breakdown=_objective_breakdown(policy_plan, global_intent),
        control_churn_count=len(dispatch.approved_policies),
        agent_contributions=agent_contributions,
        handoff_records=handoff_records,
        open_questions=[
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in _open_questions(grounding_decision, policy_plan)
        ],
    )
    return RoundExecutionArtifacts(
        completed=dispatch.completed,
        report=dispatch.report,
        diagnosis=assurance.diagnosis,
        qos_feedback=dispatch.qos_feedback,
        mobility_feedback=dispatch.mobility_feedback,
        unified_plan=unified_plan,
        trace=ControlRoundTrace(
            scope=scope,
            round_index=round_index,
            global_intent=global_intent,
            grounding_decision=grounding_decision,
            policy_plan=policy_plan,
            domain_verdicts=mediation.domain_verdicts,
            dispatch_report=dispatch.report,
            qos_feedback=dispatch.qos_feedback,
            mobility_feedback=dispatch.mobility_feedback,
            diagnosis=assurance.diagnosis,
            mediator_decision=mediation.mediator_decision,
            execution_reentry=assurance.execution_reentry,
        ),
        execution_reentry=assurance.execution_reentry,
    )


def execute_planned_round(
    *,
    scope: ExecutionScope,
    policy_plan: PolicyPlanDraft,
    cr_tool: ConflictResolutionTool,
    pd_agent: PolicyDispatchAgent,
    ad_tool: AssuranceDiagnosisTool,
    grounding_decision: Optional[GroundingDecision] = None,
    global_intent: Optional[GlobalControlIntent] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
) -> RoundExecutionArtifacts:
    if scope.snapshot_id is None:
        raise ValueError("planned round execution requires scope.snapshot_id")
    if scope.round_index < 1:
        raise ValueError("planned round execution requires scope.round_index >= 1")
    if policy_plan.scope != scope:
        raise ValueError("policy plan scope must match round execution scope")
    snapshot_id = scope.snapshot_id
    snapshot_data = get_snapshot_data_by_id(snapshot_id) or {}
    if not snapshot_data:
        raise LookupError(f"bound snapshot not found: snapshot_id={snapshot_id}")
    planning_status = policy_plan.planning_status
    dispatchable_partial_plan = planning_status == "partial_plan" and bool(policy_plan.all_policies)
    if planning_status != "executable_plan" and not dispatchable_partial_plan:
        return _build_planning_blocked_artifacts(
            scope=scope,
            policy_plan=policy_plan,
            grounding_decision=grounding_decision,
            global_intent=global_intent,
            planning_status=planning_status,
        )
    mediation = _mediate_policy_plan(
        scope=scope,
        snapshot_data=snapshot_data,
        policy_plan=policy_plan,
        cr_tool=cr_tool,
        grounding_decision=grounding_decision,
        global_intent=global_intent,
    )
    dispatch = _dispatch_mediated_plan(
        policy_plan=policy_plan,
        planning_status=planning_status,
        pd_agent=pd_agent,
        conflict_result=mediation.conflict_result,
        qos_proposal=mediation.qos_proposal,
        mobility_proposal=mediation.mobility_proposal,
        trace_metadata=trace_metadata,
    )
    assurance = _diagnose_round(
        scope=scope,
        snapshot_data=snapshot_data,
        policy_plan=policy_plan,
        global_intent=global_intent,
        grounding_decision=grounding_decision,
        conflict_result=mediation.conflict_result,
        report=dispatch.report,
        qos_feedback=dispatch.qos_feedback,
        mobility_feedback=dispatch.mobility_feedback,
        dispatch_receipts=dispatch.dispatch_receipts,
        assurance_verdicts=dispatch.assurance_verdicts,
        ad_tool=ad_tool,
    )
    return _assemble_round_artifacts(
        scope=scope,
        policy_plan=policy_plan,
        global_intent=global_intent,
        grounding_decision=grounding_decision,
        mediation=mediation,
        dispatch=dispatch,
        assurance=assurance,
    )


__all__ = ["RoundExecutionArtifacts", "execute_planned_round"]
