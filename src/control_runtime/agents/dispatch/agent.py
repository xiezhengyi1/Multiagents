from __future__ import annotations

from typing import Any, Dict

from shared.runtime import ArtifactEnvelope, ArtifactWorkerMixin, JsonlTraceWriter, RuntimePersistence
from shared.runtime.trace.execution import write_deterministic_execution_trace
from shared.logging import setup_logger
from application_runtime.persistence import create_runtime_persistence

from ...execution import AssuranceEvaluator, ExecutionController
from ...integrations.pcf import dispatch_policy_to_pcf_request
from ...integrations.storage import (
    commit_applied_policy_plan,
    get_snapshot_data_by_id,
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
        persistence: RuntimePersistence | None = None,
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
            commit_policies=commit_applied_policy_plan,
            logger=self.logger,
        )
        runtime_persistence = persistence if persistence is not None else create_runtime_persistence()
        self.init_worker_runtime(persistence=runtime_persistence)

    def handle_artifact(self, envelope: ArtifactEnvelope) -> FeedbackReport:
        return self.execute_and_evaluate_from_request(
            strategy_output=envelope.payload,
            request_envelope=envelope,
        )

    def execute_and_evaluate(self, strategy_output: Any, *, trace_metadata: Dict[str, Any] | None = None) -> FeedbackReport:
        self.ensure_worker_runtime_initialized()
        request_envelope = self.cache_received_artifact(
            artifact_type="PolicyPlanDraft",
            payload=strategy_output,
            session_id=str(getattr(strategy_output, "session_id", "") or "").strip(),
            snapshot_id=str(getattr(strategy_output, "snapshot_id", "") or "").strip(),
        )
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
        write_deterministic_execution_trace(
            writer=self._trace_writer,
            agent_name=self.agent_name,
            model_name=self.model_name,
            system_prompt=self.TRACE_SYSTEM_PROMPT,
            policy_plan=self.build_response_payload(strategy_output),
            outcome=outcome,
            feedback_report=feedback_report.model_dump(mode="json"),
            request_envelope=request_envelope,
            trace_metadata=trace_metadata,
        )
        self.cache_produced_artifact(
            artifact_type="FeedbackReport",
            request_envelope=request_envelope,
            payload=feedback_report,
        )
        return feedback_report

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
