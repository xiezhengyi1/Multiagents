from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.assurance_diagnosis import AssuranceDiagnosisTool
from agents.conflict_resolution import ConflictResolutionTool
from agents.policy_dispatch import PolicyDispatchAgent
from agents.single_control import SingleControlAgent
from agents.tools.db_tool import (
    create_session_context,
    get_latest_snapshot_data,
    get_latest_snapshot_metadata,
    get_ue_context_by_supi,
    update_session_context,
)
from domain.collaboration import PlanningRequest
from domain.control_loop import (
    build_conflict_request_payload,
    build_domain_feedback,
    split_domain_proposals,
    verdicts_from_conflict_result,
)
from domain.control_plane import ControlDomain, DomainStatus, DomainVerdict, UnifiedControlPlan
from workflows.main_control_support import (
    ControlRoundResult,
    ControlRoundTrace,
    build_feedback_context,
    build_main_context,
    build_planning_context,
    parse_pda_metrics,
)
from utils.logger import log_event


class SingleAgentOrchestrator:
    def __init__(
        self,
        *,
        single_agent: Optional[SingleControlAgent] = None,
        pd_agent: Optional[PolicyDispatchAgent] = None,
        cr_tool: Optional[ConflictResolutionTool] = None,
        ad_tool: Optional[AssuranceDiagnosisTool] = None,
        max_rounds: int = 1,
        use_local_model: bool = False,
        rag_enabled: bool = True,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.single_agent = single_agent or SingleControlAgent(use_local_model=use_local_model, rag_enabled=rag_enabled)
        self.pd_agent = pd_agent or PolicyDispatchAgent(use_local_model=use_local_model)
        self.cr_tool = cr_tool or ConflictResolutionTool()
        self.ad_tool = ad_tool or AssuranceDiagnosisTool()
        self.max_rounds = max_rounds

    @staticmethod
    def _execution_order(policy_plan: Any) -> List[ControlDomain]:
        if policy_plan is None:
            return []
        order: List[ControlDomain] = []
        for item in policy_plan.all_policies:
            domain = ControlDomain.MOBILITY if item.policy_type == "PcfAmPolicyControlPolicyAssociation" else ControlDomain.QOS
            if domain not in order:
                order.append(domain)
        return order

    @staticmethod
    def _approved_policies(report: Any) -> List[Dict[str, Any]]:
        if report is None:
            return []
        try:
            payload = json.loads(report.performance_metrics or "{}")
        except json.JSONDecodeError:
            return []
        receipts = payload.get("dispatch_results", []) if isinstance(payload, dict) else []
        approved: List[Dict[str, Any]] = []
        for receipt in receipts:
            if not isinstance(receipt, dict) or receipt.get("status") != "success":
                continue
            approved.append(
                {
                    "domain": "mobility" if receipt.get("policy_type") == "PcfAmPolicyControlPolicyAssociation" else "qos",
                    "policy_id": receipt.get("policy_id"),
                    "policy_type": receipt.get("policy_type"),
                }
            )
        return approved

    def run(
        self,
        user_input: str,
        *,
        scenario_id: str = "",
        scenario_tags: Optional[List[str]] = None,
    ) -> ControlRoundResult:
        snapshot_metadata = get_latest_snapshot_metadata() or {}
        snapshot_id = str(snapshot_metadata.get("snapshot_id") or "").strip()
        session_id = create_session_context(current_step="single_control", intent_data={"raw_input": user_input}) or ""
        if not session_id:
            raise RuntimeError("failed to create session_context")
        update_session_context(session_id, current_step="single_control", status="active")

        feedback_context = ""
        previous_diagnosis: Dict[str, Any] = {}
        previous_report_payload: Dict[str, Any] = {}
        previous_mediator_decision: Optional[Dict[str, Any]] = None
        latest_result: Optional[ControlRoundResult] = None
        round_traces: List[Dict[str, Any]] = []
        completed = False

        for round_index in range(1, self.max_rounds + 1):
            log_event(
                self.single_agent.logger,
                "single_control_round_start",
                session_id=session_id,
                round_index=round_index,
                retry_count=max(0, round_index - 1),
            )
            intent_context = build_main_context(
                snapshot_id,
                round_index=round_index,
                feedback_context=feedback_context,
                previous_diagnosis=previous_diagnosis,
                previous_execution_feedback=previous_report_payload,
            )
            global_intent, operation_intent = self.single_agent.analyze_operation_intent(
                user_input=user_input,
                context=intent_context,
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            planning_request = PlanningRequest(
                operation_intent=operation_intent,
                context=build_planning_context(
                    global_intent,
                    session_id,
                    snapshot_id,
                    round_index=round_index,
                    feedback_context=feedback_context,
                    handoff_history=round_traces,
                    revision_requests=(previous_mediator_decision or {}).get("revision_requests") if isinstance(previous_mediator_decision, dict) else None,
                    unified_constraints=(previous_mediator_decision or {}).get("unified_constraints") if isinstance(previous_mediator_decision, dict) else None,
                ),
            )
            policy_plan = self.single_agent.generate_strategy(planning_request)

            qos_proposal, mobility_proposal = split_domain_proposals(policy_plan)
            initial_verdicts: List[DomainVerdict] = []
            if qos_proposal:
                initial_verdicts.append(DomainVerdict(domain=ControlDomain.QOS, status=DomainStatus.APPROVED, summary="QoS proposal ready."))
            if mobility_proposal:
                initial_verdicts.append(DomainVerdict(domain=ControlDomain.MOBILITY, status=DomainStatus.APPROVED, summary="AM policy proposal ready."))

            conflict_result = self.cr_tool.run(
                __import__("agents.conflict_resolution.contracts", fromlist=["ConflictResolutionRequest"]).ConflictResolutionRequest(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    **build_conflict_request_payload(
                        policy_plan=policy_plan,
                        ue_context=get_ue_context_by_supi(global_intent.supi or operation_intent.supi or ""),
                    ),
                )
            )
            conflict_verdicts = verdicts_from_conflict_result(conflict_result, self._execution_order(policy_plan))
            domain_verdicts = initial_verdicts + conflict_verdicts
            mediator_decision = conflict_result.to_mediator_decision()

            report = None
            dispatch_receipts: List[Dict[str, Any]] = []
            assurance_verdicts: List[Dict[str, Any]] = []
            qos_feedback: Dict[str, Any] = {}
            mobility_feedback: Dict[str, Any] = {}

            if str(conflict_result.mediator_status or "").strip().lower() == "approved":
                report = self.pd_agent.execute_and_evaluate(policy_plan)
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

            diagnosis_request = __import__("agents.assurance_diagnosis.contracts", fromlist=["AssuranceDiagnosisRequest"]).AssuranceDiagnosisRequest(
                execution_feedback={
                    "pda": report.model_dump(mode="json") if report is not None else {},
                    "qos": qos_feedback,
                    "mobility": mobility_feedback,
                },
                dispatch_receipts=dispatch_receipts,
                assurance_verdicts=assurance_verdicts,
                telemetry_snapshot=get_latest_snapshot_data() or {},
                session_id=session_id,
                snapshot_id=snapshot_id,
                upstream_context={
                    "global_intent": global_intent.model_dump(mode="json"),
                    "operation_intent": operation_intent.model_dump(mode="json"),
                    "conflict_result": conflict_result.model_dump(mode="json"),
                },
            )
            diagnosis = self.ad_tool.run(diagnosis_request).model_dump(mode="json")

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
                execution_order=self._execution_order(policy_plan),
                approved_policies=self._approved_policies(report),
                blocked_domains=[
                    verdict.domain
                    for verdict in domain_verdicts
                    if verdict.status in {DomainStatus.REJECTED, DomainStatus.NEEDS_REVISION, DomainStatus.INCOMPLETE_CONTEXT, DomainStatus.FAILED}
                ],
                objective_breakdown=policy_plan.planning_metadata or global_intent.objective_profile.model_dump(mode="json"),
                control_churn_count=len(self._approved_policies(report)),
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
            round_traces.append(json.loads(json.dumps(trace, default=lambda obj: obj.__dict__, ensure_ascii=False)))

            latest_result = ControlRoundResult(
                session_id=session_id,
                snapshot_id=snapshot_id,
                completed=completed,
                global_intent=global_intent.model_dump(mode="json"),
                unified_plan=unified_plan.model_dump(mode="json"),
                qos_feedback=qos_feedback,
                mobility_feedback=mobility_feedback,
                diagnosis=diagnosis,
                round_count=round_index,
                retry_count=max(0, round_index - 1),
                round_traces=round_traces,
            )
            if completed:
                break

            previous_diagnosis = diagnosis
            previous_mediator_decision = mediator_decision.model_dump(mode="json")
            report_payload = report.model_dump(mode="json") if report is not None else {
                "execution_status": "Failed",
                "violation_details": diagnosis.get("reason_summary") or "round execution failed",
                "correction_suggestion": "; ".join(diagnosis.get("recommended_actions") or []),
                "recommended_consumer": "single_control",
            }
            previous_report_payload = dict(report_payload)
            feedback_context = build_feedback_context(
                feedback_context,
                pda_feedback=report_payload,
                diagnosis=diagnosis,
                domain_verdicts=[item.model_dump(mode="json") for item in domain_verdicts],
                mediator_decision=mediator_decision.model_dump(mode="json"),
                round_index=round_index,
            )

        update_session_context(
            session_id,
            current_step="completed" if completed else "failed",
            current_snapshot_id=str(
                (
                    latest_result.qos_feedback.get("committed_snapshot_id")
                    if latest_result is not None and isinstance(latest_result.qos_feedback, dict)
                    else ""
                )
                or snapshot_id
            ).strip(),
            status="completed" if completed else "failed",
        )
        if latest_result is None:
            raise RuntimeError("single agent orchestrator produced no result")
        return latest_result


__all__ = ["SingleAgentOrchestrator"]
