from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from shared.runtime import AgentRuntimeContext, ArtifactEnvelope
from shared.runtime import build_run_tree_record
from shared.runtime import JsonlTraceWriter

from shared.runtime import ArtifactWorkerMixin
from shared.logging import setup_logger

from ...domain.policy_compiler import PolicyCompiler
from ...execution import AssuranceEvaluator, ExecutionController
from ...integrations.pcf import dispatch_policy_to_pcf_request
from ...integrations.storage import (
    get_snapshot_data_by_id,
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    record_mobility_event,
    upsert_am_policy_association,
    upsert_serving_nf_binding,
    upsert_ue_context,
)
from .contracts import FeedbackReport

class PolicyDispatchAgent(ArtifactWorkerMixin):
    agent_name = "policy_dispatch"
    TRACE_SYSTEM_PROMPT = (
        "You are the Policy Dispatch execution component. "
        "Compile policies, dispatch them to PCF, run assurance checks, and return a deterministic FeedbackReport."
    )

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        *,
        use_local_model: bool = False,
    ):
        self.agent_name = "policy_dispatch"
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[92m")
        self._trace_writer = JsonlTraceWriter(self.agent_name)
        self._assurance_evaluator = AssuranceEvaluator(load_snapshot_by_id=get_snapshot_data_by_id)
        self._execution_controller = ExecutionController(
            dispatch_policy=dispatch_policy_to_pcf_request,
            assurance_evaluator=self._assurance_evaluator,
            load_ue_context=get_ue_context_by_supi,
            load_ue_flow_catalog=get_ue_flow_catalog_by_supi,
            persist_ue_context=upsert_ue_context,
            persist_am_policy_association=upsert_am_policy_association,
            record_mobility_event=record_mobility_event,
            persist_serving_nf_binding=upsert_serving_nf_binding,
            logger=self.logger,
        )
        self.init_worker_runtime()

    def handle_artifact(self, envelope: ArtifactEnvelope) -> FeedbackReport:
        return self.execute_and_evaluate_from_request(
            strategy_output=envelope.payload,
            request_envelope=envelope,
        )

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
            payload=self.build_response_payload(feedback_report),
        )
        self.cache.cache_produced(envelope)

    def execute_and_evaluate(self, strategy_output: Any, *, trace_metadata: Dict[str, Any] | None = None) -> FeedbackReport:
        self.ensure_worker_runtime_initialized()
        request_envelope = self._cache_received_request(strategy_output)
        return self.execute_and_evaluate_from_request(
            strategy_output=strategy_output,
            request_envelope=request_envelope,
            trace_metadata=trace_metadata,
        )

    def execute_and_evaluate_from_request(
        self,
        *,
        strategy_output: Any,
        request_envelope: ArtifactEnvelope,
        trace_metadata: Dict[str, Any] | None = None,
    ) -> FeedbackReport:
        outcome = self._execution_controller.execute(strategy_output)
        feedback_payload = self._build_feedback_payload(strategy_output=strategy_output, outcome=outcome)
        outcome_payload = outcome.to_dict()
        if feedback_payload:
            outcome_payload["feedback_payload"] = feedback_payload
        feedback_report = FeedbackReport(**outcome_payload)
        self._write_execution_trace(
            strategy_output=strategy_output,
            outcome=outcome,
            feedback_report=feedback_report,
            request_envelope=request_envelope,
            trace_metadata=trace_metadata,
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
        trace_metadata: Dict[str, Any] | None = None,
    ) -> None:
        context = AgentRuntimeContext(
            agent_name=self.agent_name,
            session_id=request_envelope.session_id,
            snapshot_id=request_envelope.snapshot_id,
            supi=str(getattr(strategy_output, "supi", "") or "").strip() or None,
            thread_id=request_envelope.session_id or request_envelope.snapshot_id or self.agent_name,
            allow_user_interaction=False,
            trace_metadata=dict(trace_metadata or {}),
        )
        payload = {
            "messages": [
                HumanMessage(
                    content=(
                        "Policy plan:\n"
                        f"{json.dumps(self.build_response_payload(strategy_output), ensure_ascii=False)}\n\n"
                        "Execute dispatch, assurance, and feedback generation."
                    )
                )
            ],
            "trace_metadata": self._trace_metadata_payload(trace_metadata),
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

    def _trace_metadata_payload(self, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        if not metadata:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError("trace_metadata must be a dict when present")
        scenario_tags = metadata.get("scenario_tags") or []
        if not isinstance(scenario_tags, list):
            raise TypeError("trace_metadata.scenario_tags must be a list")
        return {
            "scenario_id": str(metadata.get("scenario_id") or "").strip() or None,
            "scenario_tags": [str(item).strip() for item in scenario_tags if str(item).strip()],
        }

    @staticmethod
    def _decode_metrics_payload(outcome: Any) -> Dict[str, Any]:
        metrics_text = str(getattr(outcome, "performance_metrics", "") or "").strip()
        metrics_payload = json.loads(metrics_text) if metrics_text else {}
        if metrics_payload and not isinstance(metrics_payload, dict):
            raise TypeError("performance_metrics must decode to a JSON object")
        return metrics_payload

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
        strategy_payload = self.build_response_payload(strategy_output)
        metrics_payload = self._decode_metrics_payload(outcome)

        feedback_payload = (
            feedback_report.model_dump(mode="json")
            if hasattr(feedback_report, "model_dump")
            else self.build_response_payload(feedback_report)
        )
        failure_phase = str((getattr(outcome, "feedback_payload", {}) or {}).get("phase") or "").strip()

        compile_result = metrics_payload.get("policy_plan")
        compile_status = "error" if failure_phase == "compile" else ("success" if isinstance(compile_result, dict) else "missing")
        compile_payload: Any = compile_result
        if compile_status == "error":
            compile_payload = {
                "error": str(getattr(outcome, "violation_details", "") or "compile failed"),
                "feedback_payload": getattr(outcome, "feedback_payload", {}) or {},
            }
        elif compile_status == "missing":
            compile_payload = {"compile_status": "missing", "osa_output": strategy_payload}
        self._append_trace_tool_call(
            messages,
            tool_name="compile_policy_plan",
            args={"policy_plan_draft": strategy_payload},
            result=compile_payload,
            status=compile_status,
        )
        if compile_status != "success":
            return messages

        policies = compile_result.get("policies") or []
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
                args={"failure_scope": feedback_payload_data.get("failure_scope") or ""},
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
        execution_status = str(getattr(outcome, "execution_status", "") or "").strip().lower()
        if execution_status not in {"failed", "partial success"}:
            return getattr(outcome, "feedback_payload", {}) or {}

        seed = getattr(outcome, "feedback_payload", {}) or {}
        failures = seed.get("failures") if isinstance(seed.get("failures"), list) else []
        first_failure = failures[0] if failures else {}

        payload = {
            "supi": str(getattr(strategy_output, "supi", "") or seed.get("supi") or "").strip(),
            "policy_id": str(first_failure.get("policy_id") or seed.get("policy_id") or "").strip(),
            "flow_id": str(first_failure.get("flow_id") or seed.get("flow_id") or "").strip(),
            "reason": str(
                getattr(outcome, "violation_details", "")
                or first_failure.get("error")
                or seed.get("error")
                or ""
            ).strip(),
            "dispatch_attempts": int(getattr(outcome, "dispatch_attempts", 0) or 0),
            "target_bindings_at_risk": [
                str(value).strip()
                for value in [
                    first_failure.get("supi"),
                    first_failure.get("app_id"),
                    first_failure.get("flow_id"),
                ]
                if str(value or "").strip()
            ],
            "policy_objects_at_risk": [
                str(value).strip()
                for value in [
                    first_failure.get("policy_id"),
                    first_failure.get("policy_type"),
                ]
                if str(value or "").strip()
            ],
            "reason_by_domain": {
                str(first_failure.get("domain") or "unknown"): str(
                    first_failure.get("error")
                    or getattr(outcome, "violation_details", "")
                    or ""
                ).strip()
            },
        }
        if seed:
            payload["controller_feedback"] = seed
        failure_scope = str(getattr(outcome, "failure_scope", "") or "").strip()
        if failure_scope:
            payload["failure_scope"] = failure_scope
        return payload



