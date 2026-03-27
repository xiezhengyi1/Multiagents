from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tools.db_tool import create_session_context, get_latest_session_context, update_session_context
from utils.logger import log_event, log_timing, setup_logger

if TYPE_CHECKING:
    from multi_agents.IntentEncodingAgent import IntentEncodingAgent
    from multi_agents.MemoryManager import MemoryManager
    from multi_agents.OptimizationStrategyAgent import OptimizationStrategyAgent
    from multi_agents.PolicyDispatchAgent import PolicyDispatchAgent


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "intent": self.intent,
            "strategy": self.strategy,
            "feedback": self.feedback,
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
            from multi_agents.IntentEncodingAgent import IntentEncodingAgent
            from multi_agents.OptimizationStrategyAgent import OptimizationStrategyAgent
            from multi_agents.PolicyDispatchAgent import PolicyDispatchAgent

            ie_agent = ie_agent or IntentEncodingAgent()
            os_agent = os_agent or OptimizationStrategyAgent()
            pd_agent = pd_agent or PolicyDispatchAgent()

        self.ie_agent = ie_agent
        self.os_agent = os_agent
        self.pd_agent = pd_agent
        self.max_rounds = max_rounds
        self.memory_manager = memory_manager if memory_manager is not None else self._init_memory_manager()

    def _init_memory_manager(self) -> Optional["MemoryManager"]:
        try:
            from multi_agents.MemoryManager import MemoryManager

            long_term_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "database",
                "long_term_memory.json",
            )
            return MemoryManager(
                short_term_limit=max(20, self.max_rounds * 8),
                long_term_file=long_term_file,
            )
        except Exception as exc:
            self.logger.warning(f"MemoryManager unavailable, coordinator will run without it: {exc}")
            return None

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

    def _safe_update_session_context(
        self,
        session_id: Optional[str],
        *,
        current_step: Optional[str] = None,
        intent_data: Optional[Dict[str, Any]] = None,
        policy_data: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> None:
        if not session_id:
            return
        ok = update_session_context(
            session_id,
            current_step=current_step,
            intent_data=intent_data,
            policy_data=policy_data,
            status=status,
        )
        if not ok:
            self.logger.warning(f"Failed to persist session context update for {session_id}.")

    def _remember(self, role: str, payload: Any) -> None:
        if self.memory_manager is None:
            return

        try:
            if isinstance(payload, str):
                content = payload
            elif isinstance(payload, dict):
                content = json.dumps(payload, ensure_ascii=False)
            else:
                content = json.dumps(_to_dict(payload), ensure_ascii=False)
            self.memory_manager.add_memory(role, content)
        except Exception as exc:
            self.logger.warning(f"Failed to write memory for role={role}: {exc}")

    def _build_memory_context(self, user_input: str) -> str:
        if self.memory_manager is None:
            return ""

        try:
            memory_bundle = self.memory_manager.retrieve(user_input)
        except Exception as exc:
            self.logger.warning(f"Failed to retrieve memory context: {exc}")
            return ""

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

    def _extract_latest_intent_from_session_context(self) -> Optional[Dict[str, Any]]:
        latest_session = get_latest_session_context()
        if not isinstance(latest_session, dict):
            return None

        intent_data = latest_session.get("intent_data")
        if not isinstance(intent_data, dict):
            return None

        latest_intent = intent_data.get("latest_intent")
        if isinstance(latest_intent, dict) and latest_intent:
            return latest_intent

        history = intent_data.get("intent_history")
        if isinstance(history, list):
            for item in reversed(history):
                if isinstance(item, dict) and isinstance(item.get("output"), dict):
                    return item["output"]
        return None

    def _run_iea_with_retry(self, user_input: str, context: str = "") -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                intent_obj = self.ie_agent.analyze_intent(user_input, context=context)
                if intent_obj is None:
                    raise RuntimeError("IEA returned None")
                if attempt == 1:
                    self.logger.info("IEA succeeded on retry.")
                return intent_obj
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"IEA failed on attempt {attempt + 1}/2: {exc}")
        raise RuntimeError(f"IEA failed after retry: {last_error}")

    def _run_osa_with_retry(self, intent: Dict[str, Any]) -> Any:
        primary_error: Optional[Exception] = None
        try:
            strategy_obj = self.os_agent.generate_strategy(intent)
            if strategy_obj is None:
                raise RuntimeError("OSA returned None")
            return strategy_obj
        except Exception as exc:
            primary_error = exc
            self.logger.warning(f"OSA failed on first attempt: {exc}")

        fallback_intent = self._extract_latest_intent_from_session_context()
        if not isinstance(fallback_intent, dict) or not fallback_intent:
            raise RuntimeError(f"OSA failed and no latest intent_data was available for retry: {primary_error}")

        try:
            strategy_obj = self.os_agent.generate_strategy(fallback_intent)
            if strategy_obj is None:
                raise RuntimeError("OSA returned None on retry")
            self.logger.info("OSA succeeded on retry using latest session_context.intent_data.")
            return strategy_obj
        except Exception as retry_exc:
            raise RuntimeError(
                f"OSA failed after retry. first_error={primary_error}; retry_error={retry_exc}"
            ) from retry_exc

    @staticmethod
    def _normalize_intent_payload(intent_obj: Any, user_input: str) -> Dict[str, Any]:
        intent = _to_dict(intent_obj)
        supi = str(intent.get("supi") or "").strip()
        if not supi:
            raise ValueError(
                "UE input is assumed to contain SUPI, but IEA output does not contain a valid supi."
            )

        flows = intent.get("flows") or []
        for flow in flows:
            if isinstance(flow, dict) and not flow.get("supi"):
                flow["supi"] = supi

        intent["supi"] = supi
        intent.setdefault("raw_input", user_input)
        return intent

    @staticmethod
    def _validate_resolved_intent(intent: Dict[str, Any]) -> None:
        resolution_status = str(intent.get("resolution_status") or "").strip().lower()
        flows = intent.get("flows") or []
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

    def run(self, user_input: str) -> CoordinationResult:
        if not str(user_input).strip():
            raise ValueError("user_input must not be empty")

        total_start = time.perf_counter()
        log_event(self.logger, "coordinator_run_start", max_rounds=self.max_rounds)

        round_traces: List[RoundTrace] = []
        feedback_context = ""
        memory_context = self._build_memory_context(user_input)
        final_feedback: Dict[str, Any] = _build_failure_feedback("Coordinator exited before PDA feedback.")
        final_supi = ""
        intent_session_data, policy_session_data = self._build_initial_session_payloads(user_input)
        session_id = create_session_context(
            current_step="intent",
            intent_data=intent_session_data,
            policy_data=policy_session_data,
            status="active",
        )
        self._remember("user", {"input": user_input})

        for round_index in range(1, self.max_rounds + 1):
            round_start = time.perf_counter()
            log_event(self.logger, "coordinator_round_start", round=round_index)

            try:
                iea_context = self._merge_context_blocks(memory_context, feedback_context)
                intent_obj = self._run_iea_with_retry(user_input, context=iea_context)
                intent = self._normalize_intent_payload(intent_obj, user_input)
                self._remember("IEA", intent)
                intent_session_data["latest_intent"] = intent
                intent_session_data["intent_history"].append(
                    {
                        "round_index": round_index,
                        "agent": "IEA",
                        "output": intent,
                    }
                )
                self._safe_update_session_context(
                    session_id,
                    current_step="intent",
                    intent_data=intent_session_data,
                    policy_data=policy_session_data,
                    status="active",
                )
                self._validate_resolved_intent(intent)
                final_supi = intent["supi"]

                strategy_obj = self._run_osa_with_retry(intent)
                strategy = _to_dict(strategy_obj)
                self._remember("OSA", strategy)
                policy_session_data["latest_strategy"] = strategy
                policy_session_data["strategy_history"].append(
                    {
                        "round_index": round_index,
                        "agent": "OSA",
                        "output": strategy,
                    }
                )
                self._safe_update_session_context(
                    session_id,
                    current_step="generation",
                    intent_data=intent_session_data,
                    policy_data=policy_session_data,
                    status="active",
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
                    )
                )
                final_feedback = feedback

                status = str(feedback.get("execution_status") or "").strip()
                persisted_status = "completed" if status == SUCCESS_STATUS else "active"
                self._safe_update_session_context(
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
                self._safe_update_session_context(
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
        self._safe_update_session_context(
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
    user_input = "请降低视频流的带宽，supi: imsi-20893002"
    coordinator = MultiAgentSystem(max_rounds=3)
    result = coordinator.run(user_input)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    # from tools.init_scenario import init_main,sync_latest_flow_five_tuples_to_ue_context
    # sync_latest_flow_five_tuples_to_ue_context()
