from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import ToolRuntime, tool

from agents.tools.wrapper_think import tool_with_reason

from agent_runtime import AgentRuntimeContext, ArtifactEnvelope
from agent_runtime.trace.builder import build_run_tree_record
from agent_runtime.trace.writer import JsonlTraceWriter
from agents.BaseAgent import BaseAgent
from domain.policy_compiler import PolicyCompiler
from agents.tools.db_tool import (
    get_snapshot_data_by_id,
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    record_mobility_event,
    upsert_am_policy_association,
    upsert_serving_nf_binding,
    upsert_ue_context,
)
from agents.tools.pcf_tools import dispatch_policy_to_pcf_request
from workflows.assurance_evaluator import AssuranceEvaluator
from workflows.execution_controller import ExecutionController

from agents.worker import ArtifactWorkerMixin
from utils.logger import setup_logger

from .contracts import FeedbackReport, FeedbackSummaryDraft
from .prompts import (
    PDA_FEEDBACK_SUMMARY_SYSTEM_PROMPT,
    PDA_FEEDBACK_SUMMARY_USER_PROMPT,
)


@tool_with_reason
def tool_feedback_to_iea(
    supi: str,
    feedback_reason: str,
    correction_suggestion: str,
    policy_id: str = "",
    flow_id: str = "",
    dispatch_attempts: int = 0,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """Build a machine-readable feedback payload for Intent Encoding Agent."""
    payload = {
        "target_agent": "intent_encoding",
        "feedback_type": "intent_resolution",
        "supi": str(supi or "").strip(),
        "policy_id": str(policy_id or "").strip(),
        "flow_id": str(flow_id or "").strip(),
        "reason": str(feedback_reason or "").strip(),
        "correction_suggestion": str(correction_suggestion or "").strip(),
        "dispatch_attempts": int(dispatch_attempts or 0),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool_with_reason
def tool_feedback_to_osa(
    supi: str,
    feedback_reason: str,
    correction_suggestion: str,
    policy_id: str = "",
    flow_id: str = "",
    dispatch_attempts: int = 0,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """Build a machine-readable feedback payload for Optimization Strategy Agent."""
    payload = {
        "target_agent": "optimization_strategy",
        "feedback_type": "policy_plan_adjustment",
        "supi": str(supi or "").strip(),
        "policy_id": str(policy_id or "").strip(),
        "flow_id": str(flow_id or "").strip(),
        "reason": str(feedback_reason or "").strip(),
        "correction_suggestion": str(correction_suggestion or "").strip(),
        "dispatch_attempts": int(dispatch_attempts or 0),
    }
    return json.dumps(payload, ensure_ascii=False)

class PolicyDispatchAgent(ArtifactWorkerMixin):
    agent_name = "policy_dispatch"
    TRACE_SYSTEM_PROMPT = (
        "You are the Policy Dispatch Agent. "
        "Compile policies, dispatch them to PCF, run assurance checks, and return a FeedbackReport."
    )

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        *,
        enable_llm_feedback_summary: bool = True,
        feedback_llm: Any = None,
    ):
        self.agent_name = "policy_dispatch"
        self.model_name = model_name
        self.enable_llm_feedback_summary = bool(enable_llm_feedback_summary)
        self._feedback_llm = feedback_llm
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[92m")
        self._trace_writer = JsonlTraceWriter(self.agent_name)
        self.init_worker_runtime()

    def expected_request_type(self) -> str:
        return "PolicyPlanDraft"

    def response_artifact_type(self) -> str:
        return "FeedbackReport"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> FeedbackReport:
        return self.execute_and_evaluate_from_request(
            strategy_output=envelope.payload,
            request_envelope=envelope,
        )

    def _build_execution_controller(self) -> ExecutionController:
        return ExecutionController(
            dispatch_policy=dispatch_policy_to_pcf_request,
            assurance_evaluator=self._build_assurance_evaluator(),
            load_ue_context=get_ue_context_by_supi,
            load_ue_flow_catalog=get_ue_flow_catalog_by_supi,
            persist_ue_context=upsert_ue_context,
            persist_am_policy_association=upsert_am_policy_association,
            record_mobility_event=record_mobility_event,
            persist_serving_nf_binding=upsert_serving_nf_binding,
            logger=self.logger,
        )

    @staticmethod
    def _build_assurance_evaluator() -> AssuranceEvaluator:
        return AssuranceEvaluator(load_snapshot_by_id=get_snapshot_data_by_id)

    @staticmethod
    def _serialize_artifact_payload(payload: Any) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"Unsupported artifact payload type: {type(payload).__name__}")

    def _cache_received_request(self, strategy_output: Any) -> ArtifactEnvelope:
        payload = (
            strategy_output.model_dump(mode="json")
            if hasattr(strategy_output, "model_dump")
            else PolicyCompiler.json_friendly(strategy_output)
        )
        envelope = ArtifactEnvelope(
            artifact_type="PolicyPlanDraft",
            source_agent="coordinator",
            target_agent=self.agent_name,
            session_id=str(getattr(strategy_output, "session_id", "") or "").strip(),
            snapshot_id=str(getattr(strategy_output, "snapshot_id", "") or "").strip(),
            payload=payload if isinstance(payload, dict) else {"value": payload},
        )
        self.cache.cache_received(envelope)
        return envelope

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        feedback_report: FeedbackReport,
    ) -> None:
        envelope = ArtifactEnvelope(
            artifact_type="FeedbackReport",
            source_agent=self.agent_name,
            target_agent=request_envelope.source_agent,
            session_id=request_envelope.session_id,
            snapshot_id=request_envelope.snapshot_id,
            correlation_id=request_envelope.correlation_id,
            upstream_artifact_ids=[request_envelope.artifact_id],
            payload=self._serialize_artifact_payload(feedback_report),
        )
        self.cache.cache_produced(envelope)

    def execute_and_evaluate(self, strategy_output: Any) -> FeedbackReport:
        self.ensure_worker_runtime_initialized()
        request_envelope = self._cache_received_request(strategy_output)
        return self.execute_and_evaluate_from_request(
            strategy_output=strategy_output,
            request_envelope=request_envelope,
        )

    def execute_and_evaluate_from_request(
        self,
        *,
        strategy_output: Any,
        request_envelope: ArtifactEnvelope,
    ) -> FeedbackReport:
        outcome = self._build_execution_controller().execute(strategy_output)
        feedback_payload = self._build_feedback_payload(strategy_output=strategy_output, outcome=outcome)
        outcome_payload = outcome.to_dict()
        if feedback_payload:
            outcome_payload["feedback_payload"] = feedback_payload
        if self.enable_llm_feedback_summary and str(outcome_payload.get("recommended_action") or "").strip() == "feedback":
            summary = self._summarize_feedback_with_llm(
                strategy_output=strategy_output,
                outcome_payload=outcome_payload,
            )
            outcome_payload["violation_details"] = summary.violation_details
            outcome_payload["correction_suggestion"] = summary.correction_suggestion
            current_payload = outcome_payload.get("feedback_payload")
            if isinstance(current_payload, dict):
                current_payload["llm_summary"] = summary.model_dump(mode="json")
        feedback_report = FeedbackReport(**outcome_payload)
        self._write_execution_trace(
            strategy_output=strategy_output,
            outcome=outcome,
            feedback_report=feedback_report,
            request_envelope=request_envelope,
        )
        self._cache_produced_result(
            request_envelope=request_envelope,
            feedback_report=feedback_report,
        )
        return feedback_report

    def _write_execution_trace(
        self,
        *,
        strategy_output: Any,
        outcome: Any,
        feedback_report: FeedbackReport,
        request_envelope: ArtifactEnvelope,
    ) -> None:
        context = AgentRuntimeContext(
            agent_name=self.agent_name,
            session_id=request_envelope.session_id,
            snapshot_id=request_envelope.snapshot_id,
            supi=str(getattr(strategy_output, "supi", "") or "").strip() or None,
            thread_id=request_envelope.session_id or request_envelope.snapshot_id or self.agent_name,
            allow_user_interaction=False,
        )
        payload = {
            "messages": [
                HumanMessage(
                    content=(
                        "Policy plan:\n"
                        f"{json.dumps(self._serialize_artifact_payload(strategy_output), ensure_ascii=False)}\n\n"
                        "Execute dispatch, assurance, and feedback generation."
                    )
                )
            ],
            "trace_metadata": self._pending_trace_metadata_payload(),
        }
        result = {
            "messages": self._build_trace_messages(
                strategy_output=strategy_output,
                outcome=outcome,
                feedback_report=feedback_report,
            ),
            "structured_response": feedback_report.model_dump(mode="json"),
        }
        now = datetime.now(timezone.utc)
        record = build_run_tree_record(
            agent_name=self.agent_name,
            model_name=self.model_name,
            system_prompt=self.TRACE_SYSTEM_PROMPT,
            run_id=f"run-{uuid4()}",
            payload=payload,
            context=context,
            result=result,
            status="success",
            captured_error=None,
            start_dt=now,
            end_dt=now,
        )
        self._trace_writer.write(record)

    def _pending_trace_metadata_payload(self) -> Dict[str, Any]:
        metadata = getattr(self, "_pending_trace_metadata", None)
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError("_pending_trace_metadata must be a dict when present")
        scenario_tags = metadata.get("scenario_tags") or []
        if not isinstance(scenario_tags, list):
            raise TypeError("_pending_trace_metadata.scenario_tags must be a list")
        return {
            "scenario_id": str(metadata.get("scenario_id") or "").strip() or None,
            "scenario_tags": [str(item).strip() for item in scenario_tags if str(item).strip()],
        }

    @staticmethod
    def _append_trace_tool_call(
        messages: list[Any],
        *,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        status: str = "success",
    ) -> None:
        tool_call_id = f"call-{uuid4().hex}"
        messages.append(
            AIMessage(
                content="",
                id=f"msg-{uuid4().hex}",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": PolicyCompiler.json_friendly(args),
                    }
                ],
            )
        )
        messages.append(
            {
                "type": "tool",
                "content": (
                    result
                    if isinstance(result, str)
                    else json.dumps(PolicyCompiler.json_friendly(result), ensure_ascii=False)
                ),
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "status": status,
                "id": f"{tool_call_id}-result",
            }
        )

    def _build_trace_messages(
        self,
        *,
        strategy_output: Any,
        outcome: Any,
        feedback_report: FeedbackReport,
    ) -> list[Any]:
        messages: list[Any] = []
        strategy_payload = self._serialize_artifact_payload(strategy_output)
        metrics_text = str(getattr(outcome, "performance_metrics", "") or "").strip()
        metrics_payload = json.loads(metrics_text) if metrics_text else {}
        if metrics_payload and not isinstance(metrics_payload, dict):
            raise TypeError("performance_metrics must decode to a JSON object")

        feedback_payload = (
            feedback_report.model_dump(mode="json")
            if hasattr(feedback_report, "model_dump")
            else self._serialize_artifact_payload(feedback_report)
        )
        failure_phase = str((getattr(outcome, "feedback_payload", {}) or {}).get("phase") or "").strip()

        compile_result = metrics_payload.get("policy_plan") or {
            "session_id": strategy_payload.get("session_id"),
            "snapshot_id": strategy_payload.get("snapshot_id"),
            "supi": strategy_payload.get("supi"),
            "policies": strategy_payload.get("all_policies") or [],
        }
        compile_status = "error" if failure_phase == "compile" else "success"
        compile_payload: Any = compile_result
        if compile_status == "error":
            compile_payload = {
                "error": str(getattr(outcome, "violation_details", "") or "compile failed"),
                "feedback_payload": getattr(outcome, "feedback_payload", {}) or {},
            }
        self._append_trace_tool_call(
            messages,
            tool_name="compile_policy_plan",
            args={"policy_plan_draft": strategy_payload},
            result=compile_payload,
            status=compile_status,
        )
        if compile_status == "error":
            return messages

        policies = compile_result.get("policies") or strategy_payload.get("all_policies") or []
        if not isinstance(policies, list):
            raise TypeError("Compiled policy list must be a list")
        for policy in policies:
            if not isinstance(policy, dict):
                raise TypeError("Each compiled policy must be a dict")
            self._append_trace_tool_call(
                messages,
                tool_name="validate_policy",
                args={"policy": policy},
                result={"status": "validated", "policy_id": policy.get("policy_id"), "policy": policy},
            )

        dispatch_results = metrics_payload.get("dispatch_results") or []
        if not isinstance(dispatch_results, list):
            raise TypeError("dispatch_results must be a list")
        for receipt in dispatch_results:
            if not isinstance(receipt, dict):
                raise TypeError("Each dispatch result must be a dict")
            self._append_trace_tool_call(
                messages,
                tool_name="dispatch_policy_to_pcf_request",
                args={
                    "policy_id": receipt.get("policy_id"),
                    "policy_type": receipt.get("policy_type"),
                    "flow_id": receipt.get("flow_id"),
                    "session_id": strategy_payload.get("session_id"),
                    "snapshot_id": strategy_payload.get("snapshot_id"),
                },
                result=receipt,
                status="success" if str(receipt.get("status") or "").strip().lower() == "success" else "error",
            )

        assurance_results = metrics_payload.get("assurance_results") or []
        if not isinstance(assurance_results, list):
            raise TypeError("assurance_results must be a list")
        for verdict in assurance_results:
            if not isinstance(verdict, dict):
                raise TypeError("Each assurance result must be a dict")
            verdict_status = str(verdict.get("status") or "").strip().lower()
            self._append_trace_tool_call(
                messages,
                tool_name="assurance_evaluate",
                args={
                    "policy_id": verdict.get("policy_id"),
                    "flow_id": verdict.get("flow_id"),
                    "snapshot_id": strategy_payload.get("snapshot_id"),
                },
                result=verdict,
                status="success" if verdict_status in {"satisfied", "skipped"} else "error",
            )

        feedback_payload_data = getattr(outcome, "feedback_payload", {}) or {}
        if not isinstance(feedback_payload_data, dict):
            raise TypeError("feedback_payload must be a dict")
        if feedback_payload_data:
            self._append_trace_tool_call(
                messages,
                tool_name="build_feedback_payload",
                args={"recommended_consumer": feedback_payload.get("recommended_consumer")},
                result=feedback_payload_data,
                status="success",
            )

        if str(feedback_report.execution_status or "").strip().lower() == "success":
            self._append_trace_tool_call(
                messages,
                tool_name="commit_policy_updates",
                args={
                    "supi": strategy_payload.get("supi"),
                    "policy_ids": [policy.get("policy_id") for policy in policies],
                    "session_id": strategy_payload.get("session_id"),
                    "snapshot_id": strategy_payload.get("snapshot_id"),
                },
                result={"status": "committed", "execution_status": feedback_report.execution_status},
                status="success",
            )

        return messages

    def _build_feedback_payload(self, *, strategy_output: Any, outcome: Any) -> Dict[str, Any]:
        if str(getattr(outcome, "recommended_action", "") or "").strip() != "feedback":
            return getattr(outcome, "feedback_payload", {}) or {}

        seed = getattr(outcome, "feedback_payload", {}) or {}
        failures = seed.get("failures") if isinstance(seed.get("failures"), list) else []
        first_failure = failures[0] if failures else {}
        tool_input = {
            "supi": str(getattr(strategy_output, "supi", "") or seed.get("supi") or "").strip(),
            "reason": str(
                getattr(outcome, "violation_details", "")
                or first_failure.get("error")
                or seed.get("error")
                or ""
            ).strip(),
            "correction_suggestion": str(getattr(outcome, "correction_suggestion", "") or "").strip(),
            "policy_id": str(first_failure.get("policy_id") or seed.get("policy_id") or "").strip(),
            "flow_id": str(first_failure.get("flow_id") or seed.get("flow_id") or "").strip(),
            "dispatch_attempts": int(getattr(outcome, "dispatch_attempts", 0) or 0),
        }

        recommended_consumer = str(getattr(outcome, "recommended_consumer", "") or "").strip()
        if recommended_consumer == "intent_encoding":
            payload_text = tool_feedback_to_iea.func(
                supi=tool_input["supi"],
                feedback_reason=tool_input["reason"],
                correction_suggestion=tool_input["correction_suggestion"],
                policy_id=tool_input["policy_id"],
                flow_id=tool_input["flow_id"],
                dispatch_attempts=tool_input["dispatch_attempts"],
            )
        elif recommended_consumer == "optimization_strategy":
            payload_text = tool_feedback_to_osa.func(
                supi=tool_input["supi"],
                feedback_reason=tool_input["reason"],
                correction_suggestion=tool_input["correction_suggestion"],
                policy_id=tool_input["policy_id"],
                flow_id=tool_input["flow_id"],
                dispatch_attempts=tool_input["dispatch_attempts"],
            )
        else:
            return seed

        payload = self._extract_json_payload_from_dispatch_output(payload_text)
        if not isinstance(payload, dict):
            return seed
        if seed:
            payload["controller_feedback"] = seed
        failure_scope = str(getattr(outcome, "failure_scope", "") or "").strip()
        if failure_scope:
            payload["failure_scope"] = failure_scope
        return payload

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                    parts.append(str(block.get("text") or ""))
            return "".join(parts).strip()
        return str(content or "").strip()

    def _get_feedback_llm(self) -> Any:
        if self._feedback_llm is None:
            self._feedback_llm = BaseAgent(model_name=self.model_name).get_llm()
        return self._feedback_llm

    def _summarize_feedback_with_llm(
        self,
        *,
        strategy_output: Any,
        outcome_payload: Dict[str, Any],
    ) -> FeedbackSummaryDraft:
        metrics_payload = json.loads(str(outcome_payload.get("performance_metrics") or "{}"))
        context_payload = {
            "policy_plan": (
                strategy_output.model_dump(mode="json")
                if hasattr(strategy_output, "model_dump")
                else PolicyCompiler.json_friendly(strategy_output)
            ),
            "controller_feedback": {
                "execution_status": outcome_payload.get("execution_status"),
                "violation_details": outcome_payload.get("violation_details"),
                "correction_suggestion": outcome_payload.get("correction_suggestion"),
                "recommended_consumer": outcome_payload.get("recommended_consumer"),
                "recommended_action": outcome_payload.get("recommended_action"),
                "failure_scope": outcome_payload.get("failure_scope"),
                "dispatch_attempts": outcome_payload.get("dispatch_attempts"),
                "feedback_payload": outcome_payload.get("feedback_payload") or {},
            },
            "execution_metrics": metrics_payload,
        }
        response = self._get_feedback_llm().invoke(
            [
                SystemMessage(content=PDA_FEEDBACK_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(
                    content=PDA_FEEDBACK_SUMMARY_USER_PROMPT.format(
                        context_json=json.dumps(context_payload, ensure_ascii=False)
                    )
                ),
            ]
        )
        return FeedbackSummaryDraft.model_validate_json(self._extract_message_text(response))

    @staticmethod
    def _extract_json_payload_from_dispatch_output(dispatch_output: str) -> Optional[Dict[str, Any]]:
        text = str(dispatch_output or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            json_start = text.find("{")
            if json_start < 0:
                return None
            try:
                payload = json.loads(text[json_start:])
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _extract_flow_id(policy_details: Any) -> Optional[str]:
        return PolicyCompiler.extract_flow_id(policy_details)
