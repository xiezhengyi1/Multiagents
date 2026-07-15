from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Inject database dependencies into the generic runtime storage layer
# (mirrors the same call in main_control_orchestrator.py — needed when this
# orchestrator is used standalone by experiment scripts).
from database.connection import SessionLocal  # noqa: E402
import database.models as _db_models  # noqa: E402
from agent_runtime.storage.runtime_store import configure as _configure_runtime_store  # noqa: E402

_configure_runtime_store(session_factory=SessionLocal, orm_module=_db_models)

from ..agents.dispatch import PolicyDispatchAgent
from ..agents.single import SingleControlAgent
from ..diagnostics.diagnosis import AssuranceDiagnosisTool
from ..diagnostics.mediation import ConflictResolutionTool
from ..context import (
    build_feedback_context_from_snapshots,
    build_main_context,
    build_round_feedback_block,
)
from ..domain.policy_plan import PolicyPlanDraft
from .contracts import ControlRoundResult, ControlRoundTrace
from .loop_state import OrchestratorLoopState, append_round_trace, finish_control_session, start_control_session
from .round_execution import execute_planned_round
from shared.logging import log_event


class SingleAgentOrchestrator:
    _startup_banner_printed = False

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
        single_model_name: str = "",
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        single_agent_kwargs: Dict[str, Any] = {
            "use_local_model": use_local_model,
            "rag_enabled": rag_enabled,
        }
        if str(single_model_name or "").strip():
            single_agent_kwargs["model_name"] = str(single_model_name).strip()
        self.single_agent = single_agent or SingleControlAgent(**single_agent_kwargs)
        self.pd_agent = pd_agent or PolicyDispatchAgent(use_local_model=use_local_model)
        self.cr_tool = cr_tool or ConflictResolutionTool()
        self.ad_tool = ad_tool or AssuranceDiagnosisTool()
        self.max_rounds = max_rounds
        if not SingleAgentOrchestrator._startup_banner_printed:
            print(
                "[single-agent] "
                f"model={getattr(self.single_agent, 'model_name', '<unknown>')} "
                f"rag_enabled={rag_enabled} "
                f"use_local_model={use_local_model}",
                flush=True,
            )
            SingleAgentOrchestrator._startup_banner_printed = True

    @staticmethod
    def _build_round_exception_diagnosis(exc: Exception) -> Dict[str, Any]:
        error_text = str(exc).strip() or exc.__class__.__name__
        return {
            "root_cause_category": "single_control_round_exception",
            "root_cause": error_text,
            "reason_summary": error_text,
            "recommended_actions": [
                "Retry the single-agent round with narrower tool usage and reuse any already grounded evidence.",
            ],
        }

    @staticmethod
    def _build_round_exception_feedback(*, exc: Exception) -> Dict[str, Any]:
        error_text = str(exc).strip() or exc.__class__.__name__
        return {
            "execution_status": "Failed",
            "violation_details": error_text,
            "correction_suggestion": "Retry the single-agent round while avoiding repeated or unnecessary tool calls.",
            "recommended_consumer": "single_control",
            "error_type": exc.__class__.__name__,
            "error": error_text,
        }

    def _plan_round(
        self,
        *,
        user_input: str,
        session_id: str,
        snapshot_id: str,
        round_index: int,
        feedback_context: str,
        previous_diagnosis: Dict[str, Any],
        previous_report_payload: Dict[str, Any],
        previous_mediator_decision: Optional[Dict[str, Any]],
        round_traces: List[Dict[str, Any]],
    ) -> PolicyPlanDraft:
        intent_context = build_main_context(
            snapshot_id,
            round_index=round_index,
            feedback_context=feedback_context,
            previous_diagnosis=previous_diagnosis,
            previous_execution_feedback=previous_report_payload,
        )
        return self.single_agent.plan_round(
            user_input=user_input,
            context=intent_context,
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_index=round_index,
            feedback_context=feedback_context,
            round_traces=round_traces,
            previous_mediator_decision=previous_mediator_decision,
        )

    def run(
        self,
        user_input: str,
        *,
        scenario_id: str = "",
        scenario_tags: Optional[List[str]] = None,
        snapshot_id: str = "",
    ) -> ControlRoundResult:
        session_id, snapshot_id = start_control_session(step_name="single_control", user_input=user_input, snapshot_id=snapshot_id)
        state = OrchestratorLoopState()

        for round_index in range(1, self.max_rounds + 1):
            log_event(
                self.single_agent.logger,
                "single_control_round_start",
                session_id=session_id,
                round_index=round_index,
                retry_count=max(0, round_index - 1),
            )
            policy_plan = None
            mediator_decision_payload = None
            report = None
            qos_feedback: Dict[str, Any] = {}
            mobility_feedback: Dict[str, Any] = {}
            domain_verdicts: List[Dict[str, Any]] = []
            diagnosis: Dict[str, Any] = {}
            exception_feedback: Dict[str, Any] = {}
            try:
                policy_plan = self._plan_round(
                    user_input=user_input,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    round_index=round_index,
                    feedback_context=build_feedback_context_from_snapshots(state.rounds),
                    previous_diagnosis=state.previous_diagnosis,
                    previous_report_payload=state.previous_report_payload,
                    previous_mediator_decision=state.previous_mediator_decision,
                    round_traces=state.round_traces,
                )

                round_execution = execute_planned_round(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    round_index=round_index,
                    policy_plan=policy_plan,
                    cr_tool=self.cr_tool,
                    pd_agent=self.pd_agent,
                    ad_tool=self.ad_tool,
                )
                state.completed = round_execution.completed
                report = round_execution.report
                qos_feedback = round_execution.qos_feedback
                mobility_feedback = round_execution.mobility_feedback
                diagnosis = round_execution.diagnosis
                mediator_decision_payload = dict(round_execution.mediator_decision_payload)
                domain_verdicts = list(round_execution.domain_verdict_payloads)
                state.latest_result = ControlRoundResult(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    completed=state.completed,
                    global_intent={},
                    unified_plan=round_execution.unified_plan.model_dump(mode="json"),
                    qos_feedback=qos_feedback,
                    mobility_feedback=mobility_feedback,
                    diagnosis=diagnosis,
                    round_count=round_index,
                    retry_count=max(0, round_index - 1),
                    round_traces=state.round_traces,
                )
            except Exception as exc:
                state.completed = False
                diagnosis = self._build_round_exception_diagnosis(exc)
                exception_feedback = self._build_round_exception_feedback(exc=exc)
                state.latest_result = ControlRoundResult(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    completed=False,
                    global_intent={},
                    unified_plan={},
                    qos_feedback={},
                    mobility_feedback={},
                    diagnosis=diagnosis,
                    round_count=round_index,
                    retry_count=max(0, round_index - 1),
                    round_traces=state.round_traces,
                )
                log_event(
                    self.single_agent.logger,
                    "single_control_round_exception",
                    session_id=session_id,
                    round_index=round_index,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
            trace = ControlRoundTrace(
                round_index=round_index,
                global_intent={},
                operation_intent={},
                policy_plan=policy_plan.model_dump(mode="json") if policy_plan is not None else {},
                domain_verdicts=domain_verdicts,
                pda_feedback=report.model_dump(mode="json") if report is not None else exception_feedback if diagnosis.get("root_cause_category") == "single_control_round_exception" else {},
                qos_feedback=qos_feedback,
                mobility_feedback=mobility_feedback,
                diagnosis=diagnosis,
            )
            log_event(
                self.single_agent.logger,
                "single_control_round_complete",
                session_id=session_id,
                round_index=round_index,
                completed=state.completed,
                diagnosis_category=str(diagnosis.get("root_cause_category") or "").strip() or "<none>",
                execution_status=report.execution_status if report is not None else "error",
            )
            if state.completed:
                append_round_trace(
                    state,
                    trace_payload=json.loads(json.dumps(trace, default=lambda obj: obj.__dict__, ensure_ascii=False)),
                )
                break

            if mediator_decision_payload is not None:
                trace.mediator_decision = dict(mediator_decision_payload)
            report_payload = report.model_dump(mode="json") if report is not None else exception_feedback or {
                "execution_status": "Failed",
                "violation_details": diagnosis.get("reason_summary") or "round execution failed",
                "correction_suggestion": "; ".join(diagnosis.get("recommended_actions") or []),
                "recommended_consumer": "single_control",
            }
            trace.pda_feedback = dict(report_payload)
            feedback_added = build_round_feedback_block(
                pda_feedback=report_payload,
                diagnosis=diagnosis,
                domain_verdicts=domain_verdicts,
                mediator_decision=mediator_decision_payload,
                round_index=round_index,
            )
            append_round_trace(
                state,
                trace_payload=json.loads(json.dumps(trace, default=lambda obj: obj.__dict__, ensure_ascii=False)),
                feedback_added=feedback_added,
            )
            log_event(
                self.single_agent.logger,
                "single_control_round_retry_scheduled",
                session_id=session_id,
                next_round=round_index + 1,
                diagnosis_category=str(diagnosis.get("root_cause_category") or "").strip() or "<none>",
                recommended_consumer=str(report_payload.get("recommended_consumer") or "").strip() or "<none>",
            )
        finish_control_session(session_id=session_id, snapshot_id=snapshot_id, state=state)
        if state.latest_result is None:
            raise RuntimeError("single agent orchestrator produced no result")
        return state.latest_result


__all__ = ["SingleAgentOrchestrator"]

