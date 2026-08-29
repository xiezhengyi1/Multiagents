"""Mediation, dispatch, assurance, and trace assembly for one control round."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Dict, List, Optional

from ..agents.dispatch.agent import PolicyDispatchAgent
from ..agents.dispatch.contracts import FeedbackReport
from ..agents.grounding.contracts import GroundingDecision
from ..agents.main.contracts import ControlDomain, GlobalControlIntent
from ..agents.planning.contracts import PolicyPlanDraft
from ..diagnostics.diagnosis import AssuranceDiagnosisTool
from ..diagnostics.diagnosis.contracts import AssuranceDiagnosisRequest
from ..diagnostics.mediation import ConflictResolutionTool
from ..diagnostics.mediation.contracts import ConflictResolutionRequest, DomainStatus, DomainVerdict
from ..integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi
from .contracts import (
    ControlRoundTrace,
    DomainProposal,
    ExecutionReentryRequest,
    PlanningBlockerReport,
    UnifiedControlPlan,
)
from .round_transitions import build_reentry_report_payload


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

    def retry_trace(
        self,
        *,
        prefer_execution_reentry: bool,
        fallback_consumer: str = "",
    ) -> ControlRoundTrace:
        """Return the typed trace with the retry payload for this orchestrator path."""

        retry_forbidden = bool(
            self.report is not None
            and isinstance(self.report.feedback_payload, dict)
            and self.report.feedback_payload.get("retry_forbidden")
        )
        if self.report is not None and (retry_forbidden or not prefer_execution_reentry):
            pda_feedback = self.report.model_dump(mode="json")
        elif prefer_execution_reentry and self.execution_reentry is not None:
            pda_feedback = build_reentry_report_payload(self.execution_reentry)
        elif prefer_execution_reentry and self.planning_blocker is not None:
            pda_feedback = {
                "execution_status": "Failed",
                "violation_details": self.planning_blocker.summary,
                "feedback_payload": {
                    "missing_evidence": list(self.planning_blocker.missing_evidence or []),
                    "blocked_targets": list(self.planning_blocker.blocked_targets or []),
                    "upstream_requests": list(self.planning_blocker.upstream_requests or []),
                    "planner_conflicts": list(self.planning_blocker.planner_conflicts or []),
                },
            }
        elif self.report is not None:
            pda_feedback = self.report.model_dump(mode="json")
        else:
            pda_feedback = {
                "execution_status": "Failed",
                "violation_details": self.diagnosis.get("reason_summary") or "round execution failed",
            }
            if fallback_consumer:
                pda_feedback["correction_suggestion"] = "; ".join(self.diagnosis.get("recommended_actions") or [])
                pda_feedback["recommended_consumer"] = fallback_consumer
        return replace(self.trace, pda_feedback=pda_feedback)


@dataclass(frozen=True)
class _MediationResult:
    qos_proposal: Optional[DomainProposal]
    mobility_proposal: Optional[DomainProposal]
    domain_verdicts: List[DomainVerdict]
    mediator_decision: Any
    conflict_result: Any


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
    return {
        "valid_slice_keys": sorted(set(valid_slice_keys)),
        "valid_snssai_keys": sorted(set(valid_snssai_keys)),
        "slice_capacity_by_snssai": slice_capacity_by_snssai,
        "flow_allocations": flow_allocations,
        "exclusive_resource_keys": [],
    }


def _effective_supi(
    global_intent: Optional[GlobalControlIntent],
    grounding_decision: Optional[GroundingDecision],
    policy_plan: PolicyPlanDraft,
) -> str:
    if global_intent is not None and str(global_intent.supi or "").strip():
        return str(global_intent.supi or "").strip()
    if grounding_decision is not None:
        for flow in grounding_decision.flows or []:
            if str(flow.supi or "").strip():
                return str(flow.supi).strip()
    return str(policy_plan.supi or "").strip()


def _execution_order(policy_plan: Optional[PolicyPlanDraft]) -> List[ControlDomain]:
    if policy_plan is None:
        return []
    order: List[ControlDomain] = []
    for item in policy_plan.all_policies:
        domain = ControlDomain.MOBILITY if item.policy_type == AM_POLICY_TYPE else ControlDomain.QOS
        if domain not in order:
            order.append(domain)
    return order


def _build_conflict_request_payload(
    *,
    policy_plan: PolicyPlanDraft,
    ue_context: Optional[Dict[str, Any]],
    snapshot_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidate_policies = [item.model_dump(mode="json") for item in policy_plan.all_policies]
    domains = {"mobility" if item.policy_type == AM_POLICY_TYPE else "qos" for item in policy_plan.all_policies}
    return {
        "candidate_policies": candidate_policies,
        "resource_view": _build_resource_view(snapshot_data),
        "conflict_scope": {"domains": sorted(domains)},
        "upstream_context": {
            "ue_context": ue_context or {},
            "planner_cross_domain_verdicts": list((policy_plan.optimizer_result or {}).get("cross_domain_verdicts") or []),
        },
    }


def _verdicts_from_conflict_result(conflict_result: Any, domains: List[ControlDomain]) -> List[DomainVerdict]:
    mediator_status = str(getattr(conflict_result, "mediator_status", "") or "").strip().lower()
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
            summary=str(getattr(conflict_result, "reason_summary", "") or "").strip(),
            hard_conflicts=[json.dumps(item, ensure_ascii=False) for item in getattr(conflict_result, "conflicts", [])],
            metrics={"affected_policy_ids": getattr(conflict_result, "affected_policy_ids", [])},
        )
        for domain in domains
    ]


def _mediate_policy_plan(
    *,
    session_id: str,
    snapshot_id: str,
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
            )
        )
    if mobility_proposal:
        initial_verdicts.append(
            DomainVerdict(
                domain=ControlDomain.MOBILITY,
                status=DomainStatus.PROPOSED,
                summary="AM policy proposal ready for mediation.",
            )
        )
    conflict_result = cr_tool.run(
        ConflictResolutionRequest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            **_build_conflict_request_payload(
                policy_plan=policy_plan,
                ue_context=get_ue_context_by_supi(
                    _effective_supi(global_intent, grounding_decision, policy_plan),
                    snapshot_id=snapshot_id,
                ),
                snapshot_data=snapshot_data,
            ),
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
    if mobility_feedback["status"] != "success" and report.feedback_payload:
        mobility_feedback["error"] = str(report.feedback_payload.get("error") or report.violation_details or "").strip()
    return qos_feedback, mobility_feedback


def _parse_pda_metrics(report: FeedbackReport) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_metrics = str(report.performance_metrics or "").strip()
    if not raw_metrics:
        return [], [], []
    payload = json.loads(raw_metrics)
    if not isinstance(payload, dict):
        raise TypeError("performance_metrics must be a JSON object")
    dispatch_receipts = payload.get("dispatch_results", []) if isinstance(payload.get("dispatch_results"), list) else []
    assurance_verdicts = payload.get("assurance_results", []) if isinstance(payload.get("assurance_results"), list) else []
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
    conflict_result: Any,
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
            feedback_payload = dict(report.feedback_payload or {})
            feedback_payload.update(
                {
                    "intent_status": "partially_satisfied",
                    "intent_blockers": intent_blockers,
                    "upstream_requests": list(policy_plan.upstream_requests or []),
                    "planning_status": "partial_plan",
                }
            )
            report = report.model_copy(
                update={
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
    session_id: str,
    snapshot_id: str,
    round_index: int,
    snapshot_data: Dict[str, Any],
    policy_plan: PolicyPlanDraft,
    global_intent: Optional[GlobalControlIntent],
    grounding_decision: Optional[GroundingDecision],
    conflict_result: Any,
    report: Optional[FeedbackReport],
    qos_feedback: Dict[str, Any],
    mobility_feedback: Dict[str, Any],
    dispatch_receipts: List[Dict[str, Any]],
    assurance_verdicts: List[Dict[str, Any]],
    ad_tool: AssuranceDiagnosisTool,
) -> _AssuranceResult:
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
            "global_intent": global_intent.model_dump(mode="json") if global_intent is not None else {},
            "grounding_decision": grounding_decision.model_dump(mode="json") if grounding_decision is not None else {},
            "conflict_result": conflict_result.model_dump(mode="json"),
        },
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
                "intent_status": "partially_satisfied",
            }
        )
    execution_reentry = None
    if report is not None and report.execution_status != "Success":
        feedback_payload = report.feedback_payload if isinstance(report.feedback_payload, dict) else {}
        if not bool(feedback_payload.get("retry_forbidden")):
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
    if policy_plan.optimizer_result:
        return dict(policy_plan.optimizer_result)
    if global_intent is not None:
        return global_intent.objective_profile.model_dump(mode="json")
    return {}


def _build_planning_blocked_artifacts(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    policy_plan: PolicyPlanDraft,
    grounding_decision: Optional[GroundingDecision],
    global_intent: Optional[GlobalControlIntent],
    planning_status: str,
) -> RoundExecutionArtifacts:
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
        session_id=session_id,
        snapshot_id=snapshot_id,
        supi=_effective_supi(global_intent, grounding_decision, policy_plan),
        global_intent=global_intent,
        domain_verdicts=[],
        blocked_domains=[
            ControlDomain(item)
            for item in _requested_domains(global_intent, policy_plan)
            if item in {ControlDomain.QOS.value, ControlDomain.MOBILITY.value}
        ],
        objective_breakdown=_objective_breakdown(policy_plan, global_intent),
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
            round_index=round_index,
            global_intent=global_intent.model_dump(mode="json") if global_intent is not None else {},
            grounding_decision=grounding_decision.model_dump(mode="json") if grounding_decision is not None else {},
            policy_plan=policy_plan.model_dump(mode="json"),
            domain_verdicts=[],
            pda_feedback={},
            qos_feedback={},
            mobility_feedback={},
            diagnosis=diagnosis,
            mediator_decision={},
            planning_blocker=blocker.model_dump(mode="json"),
        ),
        planning_blocker=blocker,
    )


def _assemble_round_artifacts(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    policy_plan: PolicyPlanDraft,
    global_intent: Optional[GlobalControlIntent],
    grounding_decision: Optional[GroundingDecision],
    mediation: _MediationResult,
    dispatch: _DispatchResult,
    assurance: _AssuranceResult,
) -> RoundExecutionArtifacts:
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
        session_id=session_id,
        snapshot_id=snapshot_id,
        supi=_effective_supi(global_intent, grounding_decision, policy_plan),
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
            round_index=round_index,
            global_intent=global_intent.model_dump(mode="json") if global_intent is not None else {},
            grounding_decision=grounding_decision.model_dump(mode="json") if grounding_decision is not None else {},
            policy_plan=policy_plan.model_dump(mode="json"),
            domain_verdicts=[item.model_dump(mode="json") for item in mediation.domain_verdicts],
            pda_feedback=dispatch.report.model_dump(mode="json") if dispatch.report is not None else {},
            qos_feedback=dispatch.qos_feedback,
            mobility_feedback=dispatch.mobility_feedback,
            diagnosis=assurance.diagnosis,
            mediator_decision=mediation.mediator_decision.model_dump(mode="json"),
            execution_reentry=(
                assurance.execution_reentry.model_dump(mode="json")
                if assurance.execution_reentry is not None
                else {}
            ),
        ),
        execution_reentry=assurance.execution_reentry,
    )


def execute_planned_round(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    policy_plan: PolicyPlanDraft,
    cr_tool: ConflictResolutionTool,
    pd_agent: PolicyDispatchAgent,
    ad_tool: AssuranceDiagnosisTool,
    grounding_decision: Optional[GroundingDecision] = None,
    global_intent: Optional[GlobalControlIntent] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
) -> RoundExecutionArtifacts:
    snapshot_data = get_snapshot_data_by_id(snapshot_id) or {}
    if not snapshot_data:
        raise LookupError(f"bound snapshot not found: snapshot_id={snapshot_id}")
    planning_status = str(policy_plan.planning_status or "").strip().lower()
    dispatchable_partial_plan = planning_status == "partial_plan" and bool(policy_plan.all_policies)
    if planning_status != "executable_plan" and not dispatchable_partial_plan:
        return _build_planning_blocked_artifacts(
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_index=round_index,
            policy_plan=policy_plan,
            grounding_decision=grounding_decision,
            global_intent=global_intent,
            planning_status=planning_status,
        )
    mediation = _mediate_policy_plan(
        session_id=session_id,
        snapshot_id=snapshot_id,
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
        session_id=session_id,
        snapshot_id=snapshot_id,
        round_index=round_index,
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
        session_id=session_id,
        snapshot_id=snapshot_id,
        round_index=round_index,
        policy_plan=policy_plan,
        global_intent=global_intent,
        grounding_decision=grounding_decision,
        mediation=mediation,
        dispatch=dispatch,
        assurance=assurance,
    )


__all__ = ["RoundExecutionArtifacts", "execute_planned_round"]
