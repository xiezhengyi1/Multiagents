from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.MemoryManager import MemoryManager
from agents.assurance_diagnosis import AssuranceDiagnosisAgent
from agents.assurance_diagnosis.contracts import AssuranceDiagnosisRequest
from agents.conflict_resolution import ConflictResolutionAgent
from agents.conflict_resolution.contracts import ConflictResolutionRequest
from agents.intent_encoding import IntentEncodingAgent
from agents.main_control import MainControlAgent
from agents.optimization_strategy import OptimizationStrategyAgent
from agents.policy_dispatch import PolicyDispatchAgent
from agents.policy_dispatch.contracts import FeedbackReport
from agents.tools.db_tool import (
    create_session_context,
    get_latest_snapshot_data,
    get_latest_snapshot_metadata,
    get_ue_context_by_supi,
    update_session_context,
)
from domain.control_plane import ControlDomain, DomainStatus, DomainVerdict, UnifiedControlPlan
from domain.control_loop import (
    build_conflict_request_payload,
    build_domain_feedback,
    enforce_global_intent,
    split_domain_proposals,
    verdicts_from_conflict_result,
)
from domain.collaboration import PlanningRequest
from domain.policy_plan import OperationIntent, PolicyPlanDraft
from workflows.main_control_support import (
    ControlRoundResult,
    ControlRoundTrace,
    build_feedback_context,
    build_main_context,
    build_planning_context,
    parse_pda_metrics,
)


class MainControlOrchestrator:
    def __init__(
        self,
        *,
        main_agent: Optional[MainControlAgent] = None,
        ie_agent: Optional[IntentEncodingAgent] = None,
        os_agent: Optional[OptimizationStrategyAgent] = None,
        pd_agent: Optional[PolicyDispatchAgent] = None,
        cr_agent: Optional[ConflictResolutionAgent] = None,
        ad_agent: Optional[AssuranceDiagnosisAgent] = None,
        memory_manager: Optional[MemoryManager] = None,
        max_rounds: int = 3,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.main_agent = main_agent or MainControlAgent()
        self.ie_agent = ie_agent or IntentEncodingAgent()
        self.os_agent = os_agent or OptimizationStrategyAgent()
        self.pd_agent = pd_agent or PolicyDispatchAgent()
        self.cr_agent = cr_agent or ConflictResolutionAgent()
        self.ad_agent = ad_agent or AssuranceDiagnosisAgent()
        self.memory_manager = memory_manager or MemoryManager(
            short_term_limit=max(20, max_rounds * 8),
            enable_llm_summarization=False,
        )
        self.max_rounds = max_rounds

    @staticmethod
    def _apply_trace_metadata(agent: Any, *, scenario_id: str = "", scenario_tags: Optional[List[str]] = None) -> None:
        if not scenario_id and not scenario_tags:
            return
        setattr(
            agent,
            "_pending_trace_metadata",
            {
                "scenario_id": str(scenario_id or "").strip(),
                "scenario_tags": [str(item).strip() for item in (scenario_tags or []) if str(item).strip()],
            },
        )

    @staticmethod
    def _clear_trace_metadata(agent: Any) -> None:
        if hasattr(agent, "_pending_trace_metadata"):
            delattr(agent, "_pending_trace_metadata")

    def _remember(self, role: str, payload: Any) -> None:
        if isinstance(payload, str):
            content = payload
        elif hasattr(payload, "model_dump"):
            content = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)
        else:
            content = json.dumps(payload, ensure_ascii=False)
        if role == "IEA":
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                supi = str(parsed.get("supi") or "").strip()
                if supi:
                    self.memory_manager.bind_supi(supi)
        self.memory_manager.add_memory(role, content)

    def _build_memory_context(self, user_input: str) -> str:
        bundle = self.memory_manager.retrieve(user_input)
        short_term = bundle.get("short_term", []) if isinstance(bundle, dict) else []
        long_term = bundle.get("long_term", []) if isinstance(bundle, dict) else []
        blocks: List[str] = []
        if short_term:
            blocks.append("[Memory][Short-Term]\n" + "\n".join(f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in short_term[-5:] if isinstance(item, dict)))
        if long_term:
            blocks.append("[Memory][Long-Term]\n" + "\n".join(str(item) for item in long_term))
        return "\n\n".join(block for block in blocks if block.strip())

    @staticmethod
    def _execution_order(policy_plan: Optional[PolicyPlanDraft]) -> List[ControlDomain]:
        if policy_plan is None:
            return []
        order: List[ControlDomain] = []
        for item in policy_plan.all_policies:
            domain = ControlDomain.MOBILITY if item.policy_type == "PcfAmPolicyControlPolicyAssociation" else ControlDomain.QOS
            if domain not in order:
                order.append(domain)
        return order

    @staticmethod
    def _approved_policies(report: Optional[FeedbackReport]) -> List[Dict[str, Any]]:
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

    def _build_ie_context(
        self,
        *,
        global_intent: Dict[str, Any],
        round_index: int,
        diagnosis: Dict[str, Any],
        feedback_context: str,
    ) -> str:
        return json.dumps(
            {
                "round_index": round_index,
                "main_intent": global_intent,
                "guidance": global_intent.get("prompt_injections", {}).get("intent_encoding", ""),
                "previous_diagnosis": diagnosis,
                "feedback_context": feedback_context,
            },
            ensure_ascii=False,
        )

    def run(
        self,
        user_input: str,
        *,
        scenario_id: str = "",
        scenario_tags: Optional[List[str]] = None,
    ) -> ControlRoundResult:
        snapshot_metadata = get_latest_snapshot_metadata() or {}
        snapshot_id = str(snapshot_metadata.get("snapshot_id") or "").strip()
        session_id = create_session_context(current_step="main_control", intent_data={"raw_input": user_input}) or ""
        if not session_id:
            raise RuntimeError("failed to create session_context")
        self.memory_manager.bind_thread(session_id)
        update_session_context(session_id, current_step="main_control", status="active")

        feedback_context = ""
        previous_diagnosis: Dict[str, Any] = {}
        previous_operation_intent: Optional[OperationIntent] = None
        previous_policy_plan: Optional[PolicyPlanDraft] = None
        previous_mediator_decision: Optional[Dict[str, Any]] = None
        latest_result: Optional[ControlRoundResult] = None
        round_traces: List[Dict[str, Any]] = []
        completed = False

        for round_index in range(1, self.max_rounds + 1):
            memory_context = self._build_memory_context(user_input)
            self._apply_trace_metadata(self.main_agent, scenario_id=scenario_id, scenario_tags=scenario_tags)
            try:
                global_intent = self.main_agent.analyze_global_intent(
                    user_input=user_input,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    context=build_main_context(
                        snapshot_id,
                        round_index=round_index,
                        memory_context=memory_context,
                        feedback_context=feedback_context,
                        previous_diagnosis=previous_diagnosis,
                    ),
                )
            finally:
                self._clear_trace_metadata(self.main_agent)
            global_intent = enforce_global_intent(user_input, global_intent)
            if not global_intent.requested_domains:
                raise RuntimeError("Main Agent returned no requested_domains; refusing to infer domains outside the agent.")
            if not str(global_intent.supi or "").strip():
                raise RuntimeError("Main Agent returned no SUPI; refusing to patch identifiers outside the agent.")
            self._remember("MAIN", global_intent)

            retry_category = str(previous_diagnosis.get("root_cause_category") or "").strip().lower()
            reuse_operation_intent = (
                round_index > 1
                and previous_operation_intent is not None
                and retry_category in {
                    "execution_failure",
                    "sla_violation",
                    "cross_domain_inconsistency",
                    "am_policy_dispatch_failure",
                    "mobility_policy_validation_failure",
                }
            )
            if reuse_operation_intent:
                operation_intent = previous_operation_intent.model_copy(deep=True)
                operation_intent.requested_domains = [item.value for item in global_intent.requested_domains]
                operation_intent.objective_profile_hint = global_intent.objective_profile.profile_name
                if global_intent.supi:
                    operation_intent.supi = global_intent.supi
            else:
                self._apply_trace_metadata(self.ie_agent, scenario_id=scenario_id, scenario_tags=scenario_tags)
                try:
                    operation_intent = self.ie_agent.analyze_operation_intent(
                        user_input=user_input,
                        context=self._build_ie_context(
                            global_intent=global_intent.model_dump(mode="json"),
                            round_index=round_index,
                            diagnosis=previous_diagnosis,
                            feedback_context=feedback_context,
                        ),
                        session_id=session_id,
                        snapshot_id=snapshot_id,
                    )
                finally:
                    self._clear_trace_metadata(self.ie_agent)
                self._remember("IEA", operation_intent)

            planning_request = PlanningRequest(
                operation_intent=operation_intent,
                context=build_planning_context(
                    global_intent,
                    session_id,
                    snapshot_id,
                    round_index=round_index,
                    memory_context=memory_context,
                    feedback_context=feedback_context,
                    handoff_history=round_traces,
                    revision_requests=(previous_mediator_decision or {}).get("revision_requests") if isinstance(previous_mediator_decision, dict) else None,
                    unified_constraints=(previous_mediator_decision or {}).get("unified_constraints") if isinstance(previous_mediator_decision, dict) else None,
                ),
            )
            reuse_policy_plan = (
                round_index > 1
                and retry_category == "execution_failure"
                and previous_policy_plan is not None
            )
            if reuse_policy_plan:
                policy_plan = previous_policy_plan.model_copy(deep=True)
            else:
                self._apply_trace_metadata(self.os_agent, scenario_id=scenario_id, scenario_tags=scenario_tags)
                try:
                    policy_plan = self.os_agent.generate_strategy(planning_request)
                finally:
                    self._clear_trace_metadata(self.os_agent)
                self._remember("OSA", policy_plan)
            previous_operation_intent = operation_intent.model_copy(deep=True)
            previous_policy_plan = policy_plan.model_copy(deep=True)

            qos_proposal, mobility_proposal = split_domain_proposals(policy_plan)
            initial_verdicts: List[DomainVerdict] = []
            if qos_proposal:
                initial_verdicts.append(DomainVerdict(domain=ControlDomain.QOS, status=DomainStatus.APPROVED, summary="QoS proposal ready."))
            if mobility_proposal:
                initial_verdicts.append(DomainVerdict(domain=ControlDomain.MOBILITY, status=DomainStatus.APPROVED, summary="AM policy proposal ready."))

            conflict_result = self.cr_agent.run(
                ConflictResolutionRequest(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    **build_conflict_request_payload(
                        policy_plan=policy_plan,
                        ue_context=get_ue_context_by_supi(global_intent.supi or operation_intent.supi or ""),
                    ),
                )
            )
            conflict_verdicts = verdicts_from_conflict_result(
                conflict_result,
                self._execution_order(policy_plan),
            )
            domain_verdicts = initial_verdicts + conflict_verdicts
            mediator_decision = conflict_result.to_mediator_decision()

            report: Optional[FeedbackReport] = None
            dispatch_receipts: List[Dict[str, Any]] = []
            assurance_verdicts: List[Dict[str, Any]] = []
            qos_feedback: Dict[str, Any] = {}
            mobility_feedback: Dict[str, Any] = {}

            if str(conflict_result.mediator_status or "").strip().lower() == "approved":
                self._apply_trace_metadata(self.pd_agent, scenario_id=scenario_id, scenario_tags=scenario_tags)
                try:
                    report = self.pd_agent.execute_and_evaluate(policy_plan)
                finally:
                    self._clear_trace_metadata(self.pd_agent)
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
                telemetry_snapshot=get_latest_snapshot_data() or {},
                session_id=session_id,
                snapshot_id=snapshot_id,
                upstream_context={
                    "global_intent": global_intent.model_dump(mode="json"),
                    "operation_intent": operation_intent.model_dump(mode="json"),
                    "conflict_result": conflict_result.model_dump(mode="json"),
                },
            )
            diagnosis = self.ad_agent.run(diagnosis_request).model_dump(mode="json")
            self._remember("AD", diagnosis)

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
                "recommended_consumer": "optimization_strategy",
            }
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
            status="completed" if completed else "failed",
        )
        if latest_result is None:
            raise RuntimeError("main control orchestrator produced no result")
        return latest_result


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main control orchestrator end-to-end.",
    )
    parser.add_argument(
        "user_input",
        nargs="?",
        help="Natural-language control request. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--scenario-id",
        dest="scenario_id",
        default="",
        help="Optional trace scenario identifier.",
    )
    parser.add_argument(
        "--scenario-tag",
        dest="scenario_tags",
        action="append",
        default=[],
        help="Optional trace scenario tag. Repeat to provide multiple tags.",
    )
    parser.add_argument(
        "--max-rounds",
        dest="max_rounds",
        type=int,
        default=3,
        help="Maximum control-loop rounds to run.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON result.",
    )
    return parser.parse_args(argv)


def _resolve_user_input(cli_value: Optional[str]) -> str:
    text = str(cli_value or "").strip()
    if text:
        return text
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    raise ValueError("user_input is required either as a positional argument or via stdin")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        user_input = _resolve_user_input(args.user_input)
        orchestrator = MainControlOrchestrator(max_rounds=args.max_rounds)
        result = orchestrator.run(
            user_input,
            scenario_id=args.scenario_id,
            scenario_tags=args.scenario_tags,
        )
        payload = result.__dict__
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0
    except Exception as exc:
        error_payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2 if args.pretty else None), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
