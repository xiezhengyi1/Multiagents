from __future__ import annotations

import argparse
import json
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agent_runtime import ArtifactEnvelope, ArtifactStore
from agents.assurance_diagnosis import AssuranceDiagnosisAgent, AssuranceDiagnosisRequest, AssuranceDiagnosisResult
from agents.conflict_resolution import ConflictResolutionAgent, ConflictResolutionRequest, ConflictResolutionResult
from domain.collaboration import AgentHandoff, PlanningContext, PlanningRequest
from domain.policy_plan import OperationIntent, PolicyPlanDraft
from agents.tools.db_tool import create_session_context, get_latest_snapshot_metadata, update_session_context
from agent_runtime.runtime_store import (
    create_or_update_session,
    record_episodic_experience,
    record_handoff,
    record_stage_result,
)
from utils.logger import log_event, log_timing, setup_logger
from workflows.runtime_registry import RuntimeAgentRegistry

if TYPE_CHECKING:
    from agents.intent_encoding import IntentEncodingAgent
    from agents.MemoryManager import MemoryManager
    from agents.optimization_strategy import OptimizationStrategyAgent
    from agents.policy_dispatch import FeedbackReport, PolicyDispatchAgent


SUCCESS_STATUS = "Success"


# 中文标注：统一序列化入口，仅支持 dict 和 Pydantic v2 model_dump，不再兜底 .dict()
def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Unsupported payload type: {type(value).__name__}")


def _build_failure_feedback(message: str, suggestion: str = "Check coordinator logs.") -> Dict[str, Any]:
    return {
        "execution_status": "Failed",
        "performance_metrics": "N/A",
        "violation_details": message,
        "correction_suggestion": suggestion,
    }


# 中文标注：简化 PDA 指标解析，无效数据直接抛出而非静默降级
def _parse_pda_metrics(feedback: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_metrics = str(feedback.get("performance_metrics") or "").strip()
    if not raw_metrics or raw_metrics == "N/A":
        return [], []
    payload = json.loads(raw_metrics)
    if not isinstance(payload, dict):
        raise TypeError(f"performance_metrics must be a JSON object, got {type(payload).__name__}")
    return (
        payload.get("dispatch_results", []),
        payload.get("assurance_results", []),
    )


def _require_store_write(ok: bool, action: str) -> None:
    if not ok:
        raise RuntimeError(f"Failed to persist {action}.")


@dataclass
class RoundTrace:
    round_index: int
    intent: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, Any] = field(default_factory=dict)
    conflict_resolution: Dict[str, Any] = field(default_factory=dict)
    assurance_diagnosis: Dict[str, Any] = field(default_factory=dict)
    handoffs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "intent": self.intent,
            "strategy": self.strategy,
            "feedback": self.feedback,
            "conflict_resolution": self.conflict_resolution,
            "assurance_diagnosis": self.assurance_diagnosis,
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


class LegacyWorkerAdapter:
    def __init__(self, agent_name: str, legacy_agent: object, artifact_type: str, response_type: str, runner: str) -> None:
        self.agent_name = agent_name
        self.legacy_agent = legacy_agent
        self._artifact_type = artifact_type
        self._response_type = response_type
        self._runner = runner
        self.artifact_store = ArtifactStore()

    def consume_request_artifact(self, request_path: Path) -> Path:
        envelope = self.artifact_store.read_artifact(request_path)
        if self._runner == "iea":
            payload = envelope.payload or {}
            result = self.legacy_agent.analyze_operation_intent(
                str(payload.get("user_input") or ""),
                context=str(payload.get("context") or ""),
                session_id=envelope.session_id,
                snapshot_id=envelope.snapshot_id,
            )
        elif self._runner == "osa":
            result = self.legacy_agent.generate_strategy(PlanningRequest.model_validate(envelope.payload))
        elif self._runner == "pda":
            result = self.legacy_agent.execute_and_evaluate(PolicyPlanDraft.model_validate(envelope.payload))
        else:
            raise ValueError(f"Unsupported legacy runner: {self._runner}")

        response = ArtifactEnvelope(
            artifact_type=self._response_type,
            source_agent=self.agent_name,
            target_agent=envelope.source_agent,
            session_id=envelope.session_id,
            snapshot_id=envelope.snapshot_id,
            correlation_id=envelope.correlation_id,
            upstream_artifact_ids=[envelope.artifact_id],
            payload=_to_dict(result),
        )
        return self.artifact_store.write_response(response)


class MultiAgentSystem:
    def __init__(
        self,
        ie_agent: Optional["IntentEncodingAgent"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        os_agent: Optional["OptimizationStrategyAgent"] = None,
        pd_agent: Optional["PolicyDispatchAgent"] = None,
        cr_agent: Optional[ConflictResolutionAgent] = None,
        ad_agent: Optional[AssuranceDiagnosisAgent] = None,
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
        self.cr_agent = cr_agent or ConflictResolutionAgent()
        self.ad_agent = ad_agent or AssuranceDiagnosisAgent()
        self.max_rounds = max_rounds
        self.memory_manager = memory_manager if memory_manager is not None else self._init_memory_manager()
        self.artifact_store = ArtifactStore()
        self.registry = RuntimeAgentRegistry(max_workers=8)
        self._register_workers()

    def _register_workers(self) -> None:
        if not hasattr(self.ie_agent, "consume_request_artifact"):
            self.ie_agent = LegacyWorkerAdapter("intent_encoding", self.ie_agent, "OperationIntentRequest", "OperationIntent", "iea")
        if not hasattr(self.os_agent, "consume_request_artifact"):
            self.os_agent = LegacyWorkerAdapter("optimization_strategy", self.os_agent, "PlanningRequest", "PolicyPlanDraft", "osa")
        if not hasattr(self.pd_agent, "consume_request_artifact"):
            self.pd_agent = LegacyWorkerAdapter("policy_dispatch", self.pd_agent, "PolicyPlanDraft", "FeedbackReport", "pda")
        self.registry.register(self.ie_agent, input_types=["OperationIntentRequest"], output_types=["OperationIntent"], concurrency=4, blocking=True)
        self.registry.register(self.os_agent, input_types=["PlanningRequest"], output_types=["PolicyPlanDraft"], concurrency=4, blocking=True)
        self.registry.register(self.pd_agent, input_types=["PolicyPlanDraft"], output_types=["FeedbackReport"], concurrency=4, blocking=True)
        self.registry.register(self.cr_agent, input_types=["ConflictResolutionRequest"], output_types=["ConflictResolutionResult"], concurrency=4, blocking=False)
        self.registry.register(self.ad_agent, input_types=["AssuranceDiagnosisRequest"], output_types=["AssuranceDiagnosisResult"], concurrency=4, blocking=False)

    def _init_memory_manager(self) -> "MemoryManager":
        from agents.MemoryManager import MemoryManager
        return MemoryManager(short_term_limit=max(20, self.max_rounds * 8))

    def _remember(self, role: str, payload: Any) -> None:
        if isinstance(payload, str):
            content = payload
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
            blocks.append("[Memory][Short-Term]\n" + "\n".join(f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in short_term[-5:] if isinstance(item, dict)))
        if long_term:
            blocks.append("[Memory][Long-Term]\n" + "\n".join(str(item) for item in long_term))
        return "\n\n".join(block for block in blocks if block.strip())

    @staticmethod
    def _merge_context_blocks(*blocks: str) -> str:
        return "\n\n".join(str(block).strip() for block in blocks if str(block or "").strip())

    @staticmethod
    def _build_feedback_context(previous_context: str, feedback: Dict[str, Any], round_index: int) -> str:
        recommended_consumer = str(feedback.get("recommended_consumer") or "intent_encoding").strip()
        if recommended_consumer == "optimization_strategy":
            guidance = "Use this feedback to refine the next round of policy planning."
        elif recommended_consumer == "intent_encoding":
            guidance = "Use this feedback to refine the next round of intent understanding."
        else:
            guidance = "Use this feedback to refine the next round."
        feedback_block = (
            f"[PDA Feedback][Round {round_index}]\n"
            f"execution_status: {feedback.get('execution_status', '')}\n"
            f"performance_metrics: {feedback.get('performance_metrics', '')}\n"
            f"violation_details: {feedback.get('violation_details', '')}\n"
            f"correction_suggestion: {feedback.get('correction_suggestion', '')}\n"
            f"recommended_consumer: {recommended_consumer}\n"
            f"recommended_action: {feedback.get('recommended_action', '')}\n"
            f"{guidance}"
        )
        return feedback_block if not previous_context else f"{previous_context}\n\n{feedback_block}"

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

    @staticmethod
    def _validate_resolved_intent(intent: Dict[str, Any]) -> None:
        supi = str(intent.get("supi") or "").strip()
        if not supi:
            raise ValueError("UE input is assumed to contain SUPI, but IEA output does not contain a valid supi.")
        resolution_status = str(intent.get("resolution_status") or "").strip().lower()
        flows = intent.get("flows") or []
        unresolved_flows = [
            flow for flow in flows
            if str(flow.get("resolution_status") or "").strip().lower() not in {"success", "resolved"}
        ]
        if resolution_status == "resolved" and not unresolved_flows:
            return
        details = []
        for flow in unresolved_flows:
            flow_name = str(flow.get("name") or "unknown")
            flow_status = str(flow.get("resolution_status") or "unmatched")
            candidates = flow.get("resolution_candidates") or []
            details.append(f"{flow_name}: {flow_status}" + (f" (candidates: {', '.join(map(str, candidates))})" if candidates else ""))
        raise ValueError("Intent resolution failed: " + "; ".join(details or [resolution_status or "unmatched"]))

    def _write_request(self, envelope: ArtifactEnvelope) -> Path:
        return self.artifact_store.write_request(envelope)

    def _dispatch(self, agent_name: str, request_envelope: ArtifactEnvelope) -> Future:
        request_path = self._write_request(request_envelope)
        return self.registry.submit_artifact(agent_name, request_path)

    def _await_response(self, future: Future) -> ArtifactEnvelope:
        response_path = future.result()
        return self.artifact_store.read_artifact(response_path)

    def _update_session_control(self, session_id: str, *, status: str, current_stage: str, current_snapshot_id: str, current_artifact_id: str = "", round_index: int = 0, last_error: str = "") -> None:
        create_or_update_session(
            session_id,
            status=status,
            current_stage=current_stage,
            current_snapshot_id=current_snapshot_id,
            current_artifact_id=current_artifact_id,
            round_index=round_index,
            last_error=last_error or None,
        )
        ok = update_session_context(
            session_id,
            current_step=current_stage,
            status=status,
        )
        if not ok:
            raise RuntimeError(f"Failed to persist session context update for {session_id}.")

    def _build_planning_request(
        self,
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

    def run(self, user_input: str) -> CoordinationResult:
        if not str(user_input).strip():
            raise ValueError("user_input must not be empty")
        log_event(self.logger, "coordinator_run_start", max_rounds=self.max_rounds)

        final_feedback: Dict[str, Any] = _build_failure_feedback("Coordinator exited before PDA feedback.")
        final_supi = ""
        round_traces: List[RoundTrace] = []
        collaboration_history: List[Dict[str, Any]] = []
        feedback_context = ""
        pending_intent_obj: Optional[OperationIntent] = None
        pending_feedback_consumer = "intent_encoding"
        session_id = create_session_context(current_step="intent", intent_data={"original_input": user_input}, policy_data={}, status="active")
        if not session_id:
            raise RuntimeError("Failed to create session context.")
        self.memory_manager.bind_thread(session_id)
        create_or_update_session(session_id, status="active", current_stage="intent", round_index=0)
        self._remember("user", {"input": user_input})

        for round_index in range(1, self.max_rounds + 1):
            round_handoffs: List[AgentHandoff] = []
            try:
                snapshot_meta = get_latest_snapshot_metadata()
                if not isinstance(snapshot_meta, dict):
                    raise RuntimeError("No network snapshot available for planning.")
                snapshot_id = str(snapshot_meta.get("snapshot_id") or "").strip()
                if not snapshot_id:
                    raise RuntimeError("Latest network snapshot is missing snapshot_id.")
                memory_context = self._build_memory_context(user_input)
                ie_response = None
                if pending_intent_obj is None or pending_feedback_consumer != "optimization_strategy":
                    iea_context = self._merge_context_blocks(memory_context, feedback_context)
                    for attempt in range(2):
                        ie_request = ArtifactEnvelope(
                            artifact_type="OperationIntentRequest",
                            source_agent="coordinator",
                            target_agent="intent_encoding",
                            session_id=session_id,
                            snapshot_id=snapshot_id,
                            payload={"user_input": user_input, "context": iea_context},
                        )
                        self._update_session_control(session_id, status="active", current_stage="intent", current_snapshot_id=snapshot_id, current_artifact_id=ie_request.artifact_id, round_index=round_index)
                        try:
                            ie_response = self._await_response(self._dispatch("intent_encoding", ie_request))
                            break
                        except Exception:
                            if attempt == 1:
                                raise
                    if ie_response is None:
                        raise RuntimeError("IEA returned no response artifact.")
                    intent_obj = OperationIntent.model_validate(ie_response.payload)
                    intent = intent_obj.model_dump(mode="json")
                    self._validate_resolved_intent(intent)
                    final_supi = intent_obj.supi
                    self._remember("IEA", intent)
                    _require_store_write(
                        record_stage_result(session_id=session_id, snapshot_id=snapshot_id, round_index=round_index, stage_name="IEA", artifact_id=ie_response.artifact_id, status="succeeded", payload=intent),
                        "IEA stage result",
                    )
                    planning_source_agent = "intent_encoding"
                else:
                    intent_obj = pending_intent_obj.model_copy(update={"session_id": session_id, "snapshot_id": snapshot_id})
                    intent = intent_obj.model_dump(mode="json")
                    final_supi = intent_obj.supi
                    planning_source_agent = "policy_dispatch"

                planning_request = self._build_planning_request(
                    round_index=round_index,
                    operation_intent=intent_obj,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    snapshot_metadata=snapshot_meta,
                    memory_context=memory_context,
                    feedback_context=feedback_context,
                    handoff_history=list(collaboration_history),
                )
                if planning_source_agent == "intent_encoding":
                    round_handoffs.append(self._build_handoff(round_index=round_index, source_agent="IEA", target_agent="OSA", artifact_type="PlanningRequest", session_id=session_id, snapshot_id=snapshot_id, summary=f"Resolved {len(intent_obj.flows)} flow(s) for SUPI {intent_obj.supi}.", payload=planning_request))
                else:
                    round_handoffs.append(self._build_handoff(round_index=round_index, source_agent="PDA", target_agent="OSA", artifact_type="PlanningRequest", session_id=session_id, snapshot_id=snapshot_id, summary="Execution feedback was routed directly back to OSA for plan revision.", payload=planning_request))

                osa_response = None
                for attempt in range(2):
                    osa_request = ArtifactEnvelope(
                        artifact_type="PlanningRequest",
                        source_agent=planning_source_agent,
                        target_agent="optimization_strategy",
                        session_id=session_id,
                        snapshot_id=snapshot_id,
                        correlation_id=ie_response.correlation_id if ie_response is not None else "",
                        upstream_artifact_ids=[ie_response.artifact_id] if ie_response is not None else [],
                        payload=planning_request.model_dump(mode="json"),
                    )
                    self._update_session_control(session_id, status="active", current_stage="generation", current_snapshot_id=snapshot_id, current_artifact_id=osa_request.artifact_id, round_index=round_index)
                    try:
                        osa_response = self._await_response(self._dispatch("optimization_strategy", osa_request))
                        break
                    except Exception:
                        if attempt == 1:
                            raise
                if osa_response is None:
                    raise RuntimeError("OSA returned no response artifact.")
                strategy_obj = PolicyPlanDraft.model_validate(osa_response.payload)
                strategy = strategy_obj.model_dump(mode="json")
                self._remember("OSA", strategy)
                _require_store_write(
                    record_stage_result(session_id=session_id, snapshot_id=snapshot_id, round_index=round_index, stage_name="OSA", artifact_id=osa_response.artifact_id, status="succeeded", payload=strategy),
                    "OSA stage result",
                )
                round_handoffs.append(self._build_handoff(round_index=round_index, source_agent="OSA", target_agent="PDA", artifact_type="PolicyPlanDraft", session_id=session_id, snapshot_id=snapshot_id, summary=f"Prepared {len(strategy_obj.all_policies)} policy draft(s) for execution.", payload=strategy_obj))

                pda_request = ArtifactEnvelope(
                    artifact_type="PolicyPlanDraft",
                    source_agent="optimization_strategy",
                    target_agent="policy_dispatch",
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    correlation_id=osa_response.correlation_id,
                    upstream_artifact_ids=[osa_response.artifact_id],
                    payload=strategy,
                )
                cr_request = ArtifactEnvelope(
                    artifact_type="ConflictResolutionRequest",
                    source_agent="optimization_strategy",
                    target_agent="conflict_resolution",
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    correlation_id=osa_response.correlation_id,
                    upstream_artifact_ids=[osa_response.artifact_id],
                    payload=ConflictResolutionRequest(
                        candidate_policies=strategy.get("all_policies", []),
                        session_id=session_id,
                        snapshot_id=snapshot_id,
                        upstream_context={"planning_request": planning_request.model_dump(mode="json")},
                    ).model_dump(mode="json"),
                )

                pda_future = self._dispatch("policy_dispatch", pda_request)
                cr_future = self._dispatch("conflict_resolution", cr_request)
                pda_response = self._await_response(pda_future)
                cr_response = self._await_response(cr_future)
                feedback_obj = _to_dict(pda_response.payload)
                conflict_obj = ConflictResolutionResult.model_validate(cr_response.payload).model_dump(mode="json")
                final_feedback = feedback_obj
                self._remember("PDA", feedback_obj)
                _require_store_write(
                    record_stage_result(session_id=session_id, snapshot_id=snapshot_id, round_index=round_index, stage_name="PDA", artifact_id=pda_response.artifact_id, status="succeeded", payload=feedback_obj),
                    "PDA stage result",
                )
                _require_store_write(
                    record_stage_result(session_id=session_id, snapshot_id=snapshot_id, round_index=round_index, stage_name="CR", artifact_id=cr_response.artifact_id, status=str(conflict_obj.get("status") or "succeeded"), payload=conflict_obj),
                    "CR stage result",
                )

                dispatch_receipts, assurance_results = _parse_pda_metrics(feedback_obj)
                ad_request = ArtifactEnvelope(
                    artifact_type="AssuranceDiagnosisRequest",
                    source_agent="policy_dispatch",
                    target_agent="assurance_diagnosis",
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    correlation_id=pda_response.correlation_id,
                    upstream_artifact_ids=[pda_response.artifact_id, cr_response.artifact_id],
                    payload=AssuranceDiagnosisRequest(
                        execution_feedback=feedback_obj,
                        dispatch_receipts=dispatch_receipts,
                        assurance_verdicts=assurance_results,
                        telemetry_snapshot={"snapshot_id": snapshot_id},
                        session_id=session_id,
                        snapshot_id=snapshot_id,
                        upstream_context={"conflict_resolution": conflict_obj},
                    ).model_dump(mode="json"),
                )
                ad_response = self._await_response(self._dispatch("assurance_diagnosis", ad_request))
                diagnosis_obj = AssuranceDiagnosisResult.model_validate(ad_response.payload).model_dump(mode="json")
                _require_store_write(
                    record_stage_result(session_id=session_id, snapshot_id=snapshot_id, round_index=round_index, stage_name="AD", artifact_id=ad_response.artifact_id, status=str(diagnosis_obj.get("status") or "succeeded"), payload=diagnosis_obj),
                    "AD stage result",
                )

                status = str(feedback_obj.get("execution_status") or "").strip()
                if status == SUCCESS_STATUS:
                    pending_intent_obj = None
                    pending_feedback_consumer = "intent_encoding"
                    for handoff in round_handoffs:
                        collaboration_history.append(handoff.model_dump(mode="json"))
                        _require_store_write(
                            record_handoff(
                                session_id=handoff.session_id,
                                snapshot_id=handoff.snapshot_id,
                                round_index=handoff.round_index,
                                source_agent=handoff.source_agent,
                                target_agent=handoff.target_agent,
                                artifact_id="",
                                artifact_type=handoff.artifact_type,
                                summary=handoff.summary,
                                payload=handoff.payload,
                            ),
                            f"handoff {handoff.source_agent}->{handoff.target_agent}",
                        )
                    round_traces.append(
                        RoundTrace(
                            round_index=round_index,
                            intent=intent,
                            strategy=strategy,
                            feedback=feedback_obj,
                            conflict_resolution=conflict_obj,
                            assurance_diagnosis=diagnosis_obj,
                            handoffs=[handoff.model_dump(mode="json") for handoff in round_handoffs],
                        )
                    )
                    self._update_session_control(session_id, status="completed", current_stage="execution", current_snapshot_id=snapshot_id, current_artifact_id=pda_response.artifact_id, round_index=round_index)
                    _require_store_write(
                        record_episodic_experience(
                            raw_intent=user_input,
                            applied_policy=strategy,
                            environment_state={"snapshot_id": snapshot_id},
                            feedback_metrics=feedback_obj,
                            reward_score=1.0,
                        ),
                        "episodic experience",
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

                pending_feedback_consumer = str(feedback_obj.get("recommended_consumer") or "intent_encoding").strip() or "intent_encoding"
                feedback_target_agent = "OSA" if pending_feedback_consumer == "optimization_strategy" else "IEA"
                feedback_summary = (
                    "Execution failed and feedback was handed back to OSA for the next round."
                    if feedback_target_agent == "OSA"
                    else "Execution failed and feedback was handed back to IEA for the next round."
                )
                round_handoffs.append(
                    self._build_handoff(
                        round_index=round_index,
                        source_agent="PDA",
                        target_agent=feedback_target_agent,
                        artifact_type="FeedbackReport",
                        session_id=session_id,
                        snapshot_id=snapshot_id,
                        summary=feedback_summary,
                        payload=feedback_obj,
                    )
                )
                pending_intent_obj = intent_obj if feedback_target_agent == "OSA" else None
                for handoff in round_handoffs:
                    collaboration_history.append(handoff.model_dump(mode="json"))
                    _require_store_write(
                        record_handoff(
                            session_id=handoff.session_id,
                            snapshot_id=handoff.snapshot_id,
                            round_index=handoff.round_index,
                            source_agent=handoff.source_agent,
                            target_agent=handoff.target_agent,
                            artifact_id="",
                            artifact_type=handoff.artifact_type,
                            summary=handoff.summary,
                            payload=handoff.payload,
                        ),
                        f"handoff {handoff.source_agent}->{handoff.target_agent}",
                    )
                round_traces.append(
                    RoundTrace(
                        round_index=round_index,
                        intent=intent,
                        strategy=strategy,
                        feedback=feedback_obj,
                        conflict_resolution=conflict_obj,
                        assurance_diagnosis=diagnosis_obj,
                        handoffs=[handoff.model_dump(mode="json") for handoff in round_handoffs],
                    )
                )
                feedback_context = self._build_feedback_context(feedback_context, feedback_obj, round_index)
                self._update_session_control(session_id, status="active", current_stage="execution", current_snapshot_id=snapshot_id, current_artifact_id=pda_response.artifact_id, round_index=round_index)
            except Exception as exc:
                final_feedback = _build_failure_feedback(str(exc))
                self._remember("Coordinator", final_feedback)
                create_or_update_session(session_id, status="failed", last_error=str(exc))
                update_session_context(session_id, current_step="execution", status="failed")
                round_traces.append(RoundTrace(round_index=round_index, feedback=final_feedback))
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
        create_or_update_session(session_id, status="completed" if final_status == SUCCESS_STATUS else "failed")
        update_session_context(session_id, current_step="execution", status="completed" if final_status == SUCCESS_STATUS else "failed")
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
            return f"Coordinator completed in {result.rounds_executed} round(s) for SUPI {result.supi}. PDA status: {result.final_status}."
        suggestion = str(result.final_feedback.get("correction_suggestion") or "").strip()
        return (
            f"Coordinator stopped after {result.rounds_executed} round(s) for SUPI {result.supi or 'unknown'}. "
            f"PDA status: {result.final_status}." + (f" Suggestion: {suggestion}" if suggestion else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent closed loop")
    parser.add_argument("--user-input", default="please reduce video flow bandwidth, supi: imsi-20893002")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    coordinator = MultiAgentSystem(max_rounds=args.max_rounds)
    result = coordinator.run(args.user_input)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    # from tools.init_scenario import init_main
    # init_main()
