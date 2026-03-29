from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from domain.collaboration import AgentHandoff, PlanningContext, PlanningRequest
from domain.policy_plan import OperationIntent
from tools.db_tool import create_session_context, get_latest_snapshot_metadata, update_session_context
from utils.logger import log_event, log_timing, setup_logger

if TYPE_CHECKING:
    from agents.intent_encoding import IntentEncodingAgent
    from agents.MemoryManager import MemoryManager
    from agents.optimization_strategy import OptimizationStrategyAgent
    from agents.policy_dispatch import PolicyDispatchAgent


SUCCESS_STATUS = "Success"


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Unsupported payload type: {type(value).__name__}")


def _build_failure_feedback(message: str, suggestion: str = "Check coordinator logs.") -> Dict[str, Any]:
    return {
        "execution_status": "Failed",
        "performance_metrics": "N/A",
        "violation_details": message,
        "correction_suggestion": suggestion,
    }


@dataclass
class RoundTrace:
    round_index: int
    intent: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, Any] = field(default_factory=dict)
    handoffs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "intent": self.intent,
            "strategy": self.strategy,
            "feedback": self.feedback,
            "handoffs": self.handoffs,
        }


@dataclass
class CoordinationResult:
    completed: bool
    final_status: str
    rounds_executed: int
    supi: str = ""
    original_input: str = ""
    final_feedback: Dict[str, Any] = field(default_factory=dict)
    round_traces: List[RoundTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed": self.completed,
            "final_status": self.final_status,
            "rounds_executed": self.rounds_executed,
            "supi": self.supi,
            "original_input": self.original_input,
            "final_feedback": self.final_feedback,
            "round_traces": [trace.to_dict() for trace in self.round_traces],
        }


class MultiAgentSystem:
    """
    Minimal coordinator for the IEA -> OSA -> PDA closed loop.

    Notes:
    1. Each round always starts from IEA and ends at PDA.
    2. PDA feedback is injected into the next IEA round as context.
    3. The loop stops immediately when PDA returns execution_status == "Success".
    4. This coordinator assumes the UE input includes a SUPI. If IEA does not
       return a SUPI, the round is treated as failed because the upstream
       assumption was not satisfied by the extracted structure.
    """

    def __init__(
        self,
        ie_agent: Optional["IntentEncodingAgent"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        os_agent: Optional["OptimizationStrategyAgent"] = None,
        pd_agent: Optional["PolicyDispatchAgent"] = None,
        *,
        max_rounds: int = 3,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[96m")
        if ie_agent is None or os_agent is None or pd_agent is None:
            from agents.intent_encoding import IntentEncodingAgent
            from agents.optimization_strategy import OptimizationStrategyAgent
            from agents.policy_dispatch import PolicyDispatchAgent

            ie_agent = ie_agent or IntentEncodingAgent()
            os_agent = os_agent or OptimizationStrategyAgent()
            pd_agent = pd_agent or PolicyDispatchAgent()

        self.ie_agent = ie_agent
        self.os_agent = os_agent
        self.pd_agent = pd_agent
        self.max_rounds = max_rounds
        self.memory_manager = memory_manager if memory_manager is not None else self._init_memory_manager()

    def _init_memory_manager(self) -> "MemoryManager":
        from agents.MemoryManager import MemoryManager
        return MemoryManager(
            short_term_limit=max(20, self.max_rounds * 8),
        )

    def _build_initial_session_payloads(self, user_input: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        return (
            {
                "original_input": user_input,
                "latest_intent": None,
                "intent_history": [],
            },
            {
                "latest_strategy": None,
                "latest_feedback": None,
                "strategy_history": [],
                "feedback_history": [],
            },
        )

    def _update_session_context(
        self,
        session_id: Optional[str],
        *,
        current_step: Optional[str] = None,
        intent_data: Optional[Dict[str, Any]] = None,
        policy_data: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> None:
        if not session_id:
            raise RuntimeError("session_id is required before updating session context.")
        ok = update_session_context(
            session_id,
            current_step=current_step,
            intent_data=intent_data,
            policy_data=policy_data,
            status=status,
        )
        if not ok:
            raise RuntimeError(f"Failed to persist session context update for {session_id}.")

    def _remember(self, role: str, payload: Any) -> None:
        if isinstance(payload, str):
            content = payload
        elif isinstance(payload, dict):
            content = json.dumps(payload, ensure_ascii=False)
        else:
            content = json.dumps(_to_dict(payload), ensure_ascii=False)
        if role == "IEA":
            payload_dict = payload if isinstance(payload, dict) else _to_dict(payload)
            self.memory_manager.bind_supi(str(payload_dict.get("supi") or "").strip() or None)
        self.memory_manager.add_memory(role, content)

    def _build_memory_context(self, user_input: str) -> str:
        memory_bundle = self.memory_manager.retrieve(user_input)
        short_term = memory_bundle.get("short_term", []) if isinstance(memory_bundle, dict) else []
        long_term = memory_bundle.get("long_term", []) if isinstance(memory_bundle, dict) else []
        blocks: List[str] = []

        if short_term:
            short_lines = []
            for item in short_term[-5:]:
                if isinstance(item, dict):
                    short_lines.append(f"{item.get('role', 'unknown')}: {item.get('content', '')}")
            if short_lines:
                blocks.append("[Memory][Short-Term]\n" + "\n".join(short_lines))

        if long_term:
            blocks.append("[Memory][Long-Term]\n" + "\n".join(str(item) for item in long_term))

        return "\n\n".join(blocks)

    @staticmethod
    def _merge_context_blocks(*blocks: str) -> str:
        cleaned = [str(block).strip() for block in blocks if str(block or "").strip()]
        return "\n\n".join(cleaned)

    def _run_iea_with_retry(
        self,
        user_input: str,
        *,
        context: str = "",
        session_id: str = "",
        snapshot_id: str = "",
    ) -> OperationIntent:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                intent_obj = self.ie_agent.analyze_operation_intent(
                    user_input,
                    context=context,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                )
                if intent_obj is None:
                    raise RuntimeError("IEA returned None")
                if attempt == 1:
                    self.logger.info("IEA succeeded on retry.")
                return intent_obj
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"IEA failed on attempt {attempt + 1}/2: {exc}")
        raise RuntimeError(f"IEA failed after retry: {last_error}")

    def _run_osa_with_retry(self, planning_request: PlanningRequest) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                strategy_obj = self.os_agent.generate_strategy(planning_request)
                if strategy_obj is None:
                    raise RuntimeError("OSA returned None")
                if attempt == 1:
                    self.logger.info("OSA succeeded on retry with the same planning request.")
                return strategy_obj
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"OSA failed on attempt {attempt + 1}/2: {exc}")
        raise RuntimeError(f"OSA failed after retry: {last_error}")

    @staticmethod
    def _require_snapshot_metadata() -> Dict[str, Any]:
        snapshot_meta = get_latest_snapshot_metadata()
        if not isinstance(snapshot_meta, dict):
            raise RuntimeError("No network snapshot available for planning.")
        snapshot_id = str(snapshot_meta.get("snapshot_id") or "").strip()
        if not snapshot_id:
            raise RuntimeError("Latest network snapshot is missing snapshot_id.")
        return snapshot_meta

    @staticmethod
    def _validate_resolved_intent(intent: Any) -> None:
        intent_payload = _to_dict(intent)
        supi = str(intent_payload.get("supi") or "").strip()
        if not supi:
            raise ValueError(
                "UE input is assumed to contain SUPI, but IEA output does not contain a valid supi."
            )
        resolution_status = str(intent_payload.get("resolution_status") or "").strip().lower()
        flows = intent_payload.get("flows") or []
        unresolved_flows = [
            flow for flow in flows
            if str(flow.get("resolution_status") or "").strip().lower() != SUCCESS_STATUS.lower()
            and str(flow.get("resolution_status") or "").strip().lower() != "resolved"
        ]

        if resolution_status == "resolved" and not unresolved_flows:
            return

        details: List[str] = []
        for flow in unresolved_flows:
            flow_name = str(flow.get("name") or "unknown")
            flow_status = str(flow.get("resolution_status") or "unmatched")
            candidates = flow.get("resolution_candidates") or []
            if candidates:
                details.append(f"{flow_name}: {flow_status} (candidates: {', '.join(map(str, candidates))})")
            else:
                details.append(f"{flow_name}: {flow_status}")

        if not details:
            raise ValueError(f"Intent resolution failed: status={resolution_status or 'unmatched'}")
        raise ValueError("Intent resolution failed: " + "; ".join(details))

    @staticmethod
    def _build_feedback_context(previous_context: str, feedback: Dict[str, Any], round_index: int) -> str:
        feedback_block = (
            f"[PDA Feedback][Round {round_index}]\n"
            f"execution_status: {feedback.get('execution_status', '')}\n"
            f"performance_metrics: {feedback.get('performance_metrics', '')}\n"
            f"violation_details: {feedback.get('violation_details', '')}\n"
            f"correction_suggestion: {feedback.get('correction_suggestion', '')}\n"
            "Use this feedback to refine the next round of intent understanding."
        )
        if not previous_context:
            return feedback_block
        return f"{previous_context}\n\n{feedback_block}"

    @staticmethod
    def _build_planning_request(
        *,
        round_index: int,
        operation_intent: OperationIntent,
        session_id: str,
        snapshot_id: str,
        snapshot_metadata: Dict[str, Any],
        memory_context: str,
        feedback_context: str,
        handoff_history: List[Dict[str, Any]],
    ) -> PlanningRequest:
        return PlanningRequest(
            operation_intent=operation_intent,
            context=PlanningContext(
                round_index=round_index,
                session_id=session_id,
                snapshot_id=snapshot_id,
                snapshot_metadata=snapshot_metadata,
                memory_context=memory_context,
                feedback_context=feedback_context,
                handoff_history=handoff_history,
            ),
        )

    @staticmethod
    def _build_handoff(
        *,
        round_index: int,
        source_agent: str,
        target_agent: str,
        artifact_type: str,
        session_id: str,
        snapshot_id: str,
        summary: str,
        payload: Any,
    ) -> AgentHandoff:
        return AgentHandoff(
            round_index=round_index,
            source_agent=source_agent,
            target_agent=target_agent,
            artifact_type=artifact_type,
            session_id=session_id,
            snapshot_id=snapshot_id,
            summary=summary,
            payload=_to_dict(payload),
        )

    def run(self, user_input: str) -> CoordinationResult:
        if not str(user_input).strip():
            raise ValueError("user_input must not be empty")

        total_start = time.perf_counter()
        log_event(self.logger, "coordinator_run_start", max_rounds=self.max_rounds)

        round_traces: List[RoundTrace] = []
        collaboration_history: List[Dict[str, Any]] = []
        feedback_context = ""
        final_feedback: Dict[str, Any] = _build_failure_feedback("Coordinator exited before PDA feedback.")
        final_supi = ""
        intent_session_data, policy_session_data = self._build_initial_session_payloads(user_input)
        session_id = create_session_context(
            current_step="intent",
            intent_data=intent_session_data,
            policy_data=policy_session_data,
            status="active",
        )
        if not session_id:
            raise RuntimeError("Failed to create session context.")
        self.memory_manager.bind_thread(session_id)
        user_memory_written = False

        for round_index in range(1, self.max_rounds + 1):
            round_start = time.perf_counter()
            log_event(self.logger, "coordinator_round_start", round=round_index)
            round_handoffs: List[AgentHandoff] = []

            try:
                snapshot_meta = self._require_snapshot_metadata()
                snapshot_id = str(snapshot_meta["snapshot_id"])
                memory_context = self._build_memory_context(user_input)
                if not user_memory_written:
                    self._remember("user", {"input": user_input})
                    user_memory_written = True
                iea_context = self._merge_context_blocks(memory_context, feedback_context)
                intent_obj = self._run_iea_with_retry(
                    user_input,
                    context=iea_context,
                    session_id=session_id or "",
                    snapshot_id=snapshot_id,
                )
                intent = _to_dict(intent_obj)
                self._remember("IEA", intent)
                intent_session_data["latest_intent"] = intent
                intent_session_data["intent_history"].append(
                    {
                        "round_index": round_index,
                        "agent": "IEA",
                        "output": intent,
                    }
                )
                self._update_session_context(
                    session_id,
                    current_step="intent",
                    intent_data=intent_session_data,
                    policy_data=policy_session_data,
                    status="active",
                )
                self._validate_resolved_intent(intent)
                final_supi = intent["supi"]
                planning_request = self._build_planning_request(
                    round_index=round_index,
                    operation_intent=intent_obj,
                    session_id=session_id or "",
                    snapshot_id=snapshot_id,
                    snapshot_metadata=snapshot_meta,
                    memory_context=memory_context,
                    feedback_context=feedback_context,
                    handoff_history=list(collaboration_history),
                )
                round_handoffs.append(
                    self._build_handoff(
                        round_index=round_index,
                        source_agent="IEA",
                        target_agent="OSA",
                        artifact_type="PlanningRequest",
                        session_id=planning_request.context.session_id,
                        snapshot_id=planning_request.context.snapshot_id,
                        summary=(
                            f"Resolved {len(intent_obj.flows)} flow(s) for SUPI {intent_obj.supi} "
                            f"at round {round_index}."
                        ),
                        payload=planning_request,
                    )
                )

                strategy_obj = self._run_osa_with_retry(planning_request)
                strategy = _to_dict(strategy_obj)
                self._remember("OSA", strategy)
                policy_session_data["snapshot"] = snapshot_meta
                policy_session_data["latest_strategy"] = strategy
                policy_session_data["strategy_history"].append(
                    {
                        "round_index": round_index,
                        "agent": "OSA",
                        "output": strategy,
                    }
                )
                self._update_session_context(
                    session_id,
                    current_step="generation",
                    intent_data=intent_session_data,
                    policy_data=policy_session_data,
                    status="active",
                )
                round_handoffs.append(
                    self._build_handoff(
                        round_index=round_index,
                        source_agent="OSA",
                        target_agent="PDA",
                        artifact_type="PolicyPlanDraft",
                        session_id=str(strategy_obj.session_id or session_id or ""),
                        snapshot_id=str(strategy_obj.snapshot_id or snapshot_id),
                        summary=f"Prepared {len(strategy_obj.all_policies)} policy draft(s) for execution.",
                        payload=strategy_obj,
                    )
                )

                feedback_obj = self.pd_agent.execute_and_evaluate(strategy_obj)
                if feedback_obj is None:
                    raise RuntimeError("PDA returned None")
                feedback = _to_dict(feedback_obj)
                self._remember("PDA", feedback)
                policy_session_data["latest_feedback"] = feedback
                policy_session_data["feedback_history"].append(
                    {
                        "round_index": round_index,
                        "agent": "PDA",
                        "output": feedback,
                    }
                )

                round_traces.append(
                    RoundTrace(
                        round_index=round_index,
                        intent=intent,
                        strategy=strategy,
                        feedback=feedback,
                        handoffs=[handoff.model_dump(mode="json") for handoff in round_handoffs],
                    )
                )
                final_feedback = feedback

                status = str(feedback.get("execution_status") or "").strip()
                persisted_status = "completed" if status == SUCCESS_STATUS else "active"
                self._update_session_context(
                    session_id,
                    current_step="execution",
                    intent_data=intent_session_data,
                    policy_data=policy_session_data,
                    status=persisted_status,
                )
                log_timing(
                    self.logger,
                    "coordinator_round_total",
                    time.perf_counter() - round_start,
                    round=round_index,
                    status=status or "unknown",
                )

                if status == SUCCESS_STATUS:
                    collaboration_history.extend([handoff.model_dump(mode="json") for handoff in round_handoffs])
                    log_timing(
                        self.logger,
                        "coordinator_run_total",
                        time.perf_counter() - total_start,
                        status="success",
                    )
                    return CoordinationResult(
                        completed=True,
                        final_status=status,
                        rounds_executed=round_index,
                        supi=final_supi,
                        original_input=user_input,
                        final_feedback=final_feedback,
                        round_traces=round_traces,
                    )

                if round_index < self.max_rounds:
                    round_handoffs.append(
                        self._build_handoff(
                            round_index=round_index,
                            source_agent="PDA",
                            target_agent="IEA",
                            artifact_type="FeedbackReport",
                            session_id=session_id or "",
                            snapshot_id=snapshot_id,
                            summary=(
                                "Execution failed and feedback was handed back to IEA "
                                "for the next closed-loop round."
                            ),
                            payload=feedback_obj,
                        )
                    )
                    round_traces[-1].handoffs = [handoff.model_dump(mode="json") for handoff in round_handoffs]
                collaboration_history.extend([handoff.model_dump(mode="json") for handoff in round_handoffs])
                feedback_context = self._build_feedback_context(feedback_context, feedback, round_index)
            except Exception as exc:
                failure_feedback = _build_failure_feedback(str(exc))
                final_feedback = failure_feedback
                self._remember("Coordinator", failure_feedback)
                policy_session_data["latest_feedback"] = failure_feedback
                policy_session_data["feedback_history"].append(
                    {
                        "round_index": round_index,
                        "agent": "Coordinator",
                        "output": failure_feedback,
                    }
                )
                self._update_session_context(
                    session_id,
                    current_step="execution",
                    intent_data=intent_session_data,
                    policy_data=policy_session_data,
                    status="failed",
                )
                round_traces.append(
                    RoundTrace(
                        round_index=round_index,
                        feedback=failure_feedback,
                        handoffs=[handoff.model_dump(mode="json") for handoff in round_handoffs],
                    )
                )
                log_timing(
                    self.logger,
                    "coordinator_round_total",
                    time.perf_counter() - round_start,
                    round=round_index,
                    status="error",
                )
                log_timing(
                    self.logger,
                    "coordinator_run_total",
                    time.perf_counter() - total_start,
                    status="error",
                )
                return CoordinationResult(
                    completed=False,
                    final_status="Failed",
                    rounds_executed=round_index,
                    supi=final_supi,
                    original_input=user_input,
                    final_feedback=final_feedback,
                    round_traces=round_traces,
                )

        final_status = str(final_feedback.get("execution_status") or "Failed").strip() or "Failed"
        log_timing(
            self.logger,
            "coordinator_run_total",
            time.perf_counter() - total_start,
            status="max_rounds",
        )
        self._update_session_context(
            session_id,
            current_step="execution",
            intent_data=intent_session_data,
            policy_data=policy_session_data,
            status="completed" if final_status == SUCCESS_STATUS else "failed",
        )
        return CoordinationResult(
            completed=final_status == SUCCESS_STATUS,
            final_status=final_status,
            rounds_executed=len(round_traces),
            supi=final_supi,
            original_input=user_input,
            final_feedback=final_feedback,
            round_traces=round_traces,
        )

    def run_loop(self, user_input: str) -> str:
        result = self.run(user_input)
        if result.completed:
            return (
                f"Coordinator completed in {result.rounds_executed} round(s) for SUPI {result.supi}. "
                f"PDA status: {result.final_status}."
            )

        suggestion = str(result.final_feedback.get("correction_suggestion") or "").strip()
        if suggestion:
            return (
                f"Coordinator stopped after {result.rounds_executed} round(s) for SUPI {result.supi or 'unknown'}. "
                f"PDA status: {result.final_status}. Suggestion: {suggestion}"
            )
        return (
            f"Coordinator stopped after {result.rounds_executed} round(s) for SUPI {result.supi or 'unknown'}. "
            f"PDA status: {result.final_status}."
        )
def main() -> None:
    user_input = "please reduce video flow bandwidth, supi: imsi-20893002"
    coordinator = MultiAgentSystem(max_rounds=3)
    result = coordinator.run(user_input)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    # from tools.init_scenario import init_main,sync_latest_flow_five_tuples_to_ue_context
    # sync_latest_flow_five_tuples_to_ue_context()
