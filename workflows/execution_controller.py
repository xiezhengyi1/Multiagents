from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from domain.policy_plan import PolicyPlan
from domain.policy_compiler import PolicyCompiler
from domain.policy_guard import PolicyGuard
from workflows.assurance_evaluator import AssuranceEvaluator
from utils.logger import log_event, log_timing, setup_logger


@dataclass(frozen=True)
class ExecutionOutcome:
    execution_status: str
    performance_metrics: str
    violation_details: str
    correction_suggestion: str
    recommended_consumer: str = "none"
    recommended_action: str = "none"
    feedback_payload: Dict[str, Any] = field(default_factory=dict)
    dispatch_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "performance_metrics": self.performance_metrics,
            "violation_details": self.violation_details,
            "correction_suggestion": self.correction_suggestion,
            "recommended_consumer": self.recommended_consumer,
            "recommended_action": self.recommended_action,
            "feedback_payload": self.feedback_payload,
            "dispatch_attempts": self.dispatch_attempts,
        }


class ExecutionDecisionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        recommended_consumer: str,
        correction_suggestion: str,
        feedback_payload: Optional[Dict[str, Any]] = None,
        dispatch_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.recommended_consumer = recommended_consumer
        self.correction_suggestion = correction_suggestion
        self.feedback_payload = feedback_payload or {}
        self.dispatch_attempts = dispatch_attempts


class ExecutionController:
    def __init__(
        self,
        *,
        compiler: Optional[PolicyCompiler] = None,
        guard: Optional[PolicyGuard] = None,
        dispatch_policy: Callable[..., Dict[str, Any]],
        assurance_evaluator: AssuranceEvaluator,
        load_ue_context: Callable[[str], Optional[Dict[str, Any]]],
        load_ue_flow_catalog: Callable[[str], Dict[str, Any]],
        persist_ue_context: Callable[..., bool],
        max_dispatch_attempts: int = 2,
        logger: Any = None,
    ) -> None:
        self.compiler = compiler or PolicyCompiler()
        self.guard = guard or PolicyGuard()
        self.dispatch_policy = dispatch_policy
        self.assurance_evaluator = assurance_evaluator
        self.load_ue_context = load_ue_context
        self.load_ue_flow_catalog = load_ue_flow_catalog
        self.persist_ue_context = persist_ue_context
        self.max_dispatch_attempts = max(1, int(max_dispatch_attempts or 1))
        self.logger = logger or setup_logger(self.__class__.__name__, default_msg_color="\033[92m")

    def _dispatch_single_policy(
        self,
        *,
        policy: Dict[str, Any],
        session_id: str,
        snapshot_id: str,
    ) -> tuple[Dict[str, Any], int]:
        attempts = 0
        last_result: Dict[str, Any] = {}
        request_id = f"req-{uuid.uuid4()}"

        while attempts < self.max_dispatch_attempts:
            attempts += 1
            result = self.dispatch_policy(
                policy["policy_type"],
                policy["policy_details"],
                request_id=request_id,
                session_id=session_id or None,
                snapshot_id=snapshot_id or None,
            )
            last_result = result if isinstance(result, dict) else {"status": "failed", "error": str(result)}

            failure_message = self._extract_dispatch_failure_message(policy=policy, result=last_result)
            if failure_message is None:
                return last_result, attempts

            if attempts < self.max_dispatch_attempts and self._should_retry_dispatch(result=last_result):
                continue

            recommended_consumer = self._classify_feedback_consumer(
                detail=failure_message,
                policy=policy,
                phase="dispatch",
            )
            raise ExecutionDecisionError(
                failure_message,
                recommended_consumer=recommended_consumer,
                correction_suggestion=self._build_correction_suggestion(
                    recommended_consumer=recommended_consumer,
                    phase="dispatch",
                ),
                feedback_payload={
                    "phase": "dispatch",
                    "policy_id": policy.get("policy_id"),
                    "policy_type": policy.get("policy_type"),
                    "flow_id": policy.get("flow_id") or policy.get("policy_details", {}).get("flow_id"),
                    "error": failure_message,
                    "last_dispatch_result": last_result,
                },
                dispatch_attempts=attempts,
            )

        raise ExecutionDecisionError(
            f"policy {policy['policy_id']} dispatch failed after retries",
            recommended_consumer="optimization_strategy",
            correction_suggestion=self._build_correction_suggestion(
                recommended_consumer="optimization_strategy",
                phase="dispatch",
            ),
            feedback_payload={
                "phase": "dispatch",
                "policy_id": policy.get("policy_id"),
                "policy_type": policy.get("policy_type"),
                "flow_id": policy.get("flow_id") or policy.get("policy_details", {}).get("flow_id"),
                "last_dispatch_result": last_result,
            },
            dispatch_attempts=attempts,
        )

    @staticmethod
    def _extract_dispatch_failure_message(*, policy: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        if result.get("status") != "success":
            error = str(result.get("error") or "unknown dispatch error").strip()
            return f"policy {policy['policy_id']} dispatch failed: {error}"

        ack = result.get("ack")
        if not isinstance(ack, dict):
            return f"policy {policy['policy_id']} missing ack in PCF response"
        if ack.get("completed") is not True:
            return (
                f"policy {policy['policy_id']} ack incomplete: expected={ack.get('expected')}, "
                f"received={ack.get('received')}, completed={ack.get('completed')}"
            )
        return None

    @staticmethod
    def _should_retry_dispatch(*, result: Dict[str, Any]) -> bool:
        error_text = str(result.get("error") or "").strip().lower()
        response_code = result.get("response_code")
        ack = result.get("ack") if isinstance(result.get("ack"), dict) else None

        if isinstance(response_code, int) and response_code >= 500:
            return True
        if any(token in error_text for token in ["timeout", "temporar", "connection", "request failed", "503", "502", "504"]):
            return True
        if ack and ack.get("completed") is not True:
            expected = ack.get("expected")
            received = ack.get("received")
            return isinstance(expected, int) and isinstance(received, int) and received < expected
        return False

    @staticmethod
    def _classify_feedback_consumer(
        *,
        detail: str,
        policy: Optional[Dict[str, Any]] = None,
        phase: str,
    ) -> str:
        normalized_detail = str(detail or "").strip().lower()
        if phase == "assurance":
            return "optimization_strategy"

        iea_signals = [
            "supi is required",
            "flow_id is required",
            "app_id is required",
            "policy_id is required",
            "unknown ue",
            "unknown flow",
            "not found",
            "intent",
            "resolution",
        ]
        if any(token in normalized_detail for token in iea_signals):
            return "intent_encoding"

        if policy and not str(policy.get("flow_id") or policy.get("policy_details", {}).get("flow_id") or "").strip():
            return "intent_encoding"
        return "optimization_strategy"

    @staticmethod
    def _build_correction_suggestion(*, recommended_consumer: str, phase: str) -> str:
        if phase == "assurance":
            return "Adjust policy parameters or optimization target before re-dispatch."
        if recommended_consumer == "intent_encoding":
            return "Re-check SUPI, app_id, flow_id, and intent resolution before building the next plan."
        return "Revise the policy plan or dispatch strategy before the next execution round."

    @staticmethod
    def _build_failure_outcome(
        *,
        detail: str,
        correction_suggestion: str,
        recommended_consumer: str,
        feedback_payload: Optional[Dict[str, Any]] = None,
        dispatch_attempts: int = 0,
    ) -> ExecutionOutcome:
        metrics = json.dumps(
            {
                "dispatch_attempts": dispatch_attempts,
                "recommended_consumer": recommended_consumer,
                "feedback_payload": feedback_payload or {},
            },
            ensure_ascii=False,
        )
        return ExecutionOutcome(
            execution_status="Failed",
            performance_metrics=metrics,
            violation_details=detail,
            correction_suggestion=correction_suggestion,
            recommended_consumer=recommended_consumer,
            recommended_action="feedback",
            feedback_payload=feedback_payload or {},
            dispatch_attempts=dispatch_attempts,
        )

    def _commit_successful_policies(self, supi: str, policies: list[Dict[str, Any]]) -> str:
        existing = self.load_ue_context(supi) or {}
        merged = self.compiler.merge_policies_into_context(existing, policies)
        catalog = self.load_ue_flow_catalog(supi)

        ok = self.persist_ue_context(
            supi=supi,
            sm_policy_data=merged["sm_policy_data"],
            pcc_rules=merged["pcc_rules"],
            qos_decs=merged["qos_decs"],
            sess_rules=merged["sess_rules"],
            traff_cont_decs=merged["traff_cont_decs"],
            chg_decs=merged["chg_decs"],
            ursp_rules=merged["ursp_rules"],
            app_catalog=catalog.get("app_catalog") or existing.get("app_catalog") or [],
            flow_catalog=catalog.get("flow_catalog") or existing.get("flow_catalog") or [],
        )
        if not ok:
            raise RuntimeError(f"failed to persist policies for supi {supi}")
        return "success"

    @staticmethod
    def _build_policy_plan_payload(plan: Any) -> PolicyPlan:
        return PolicyPlan(
            session_id=plan.session_id,
            snapshot_id=plan.snapshot_id,
            supi=plan.supi,
            policies=plan.policies,
        )

    def execute(self, strategy_output: Any) -> ExecutionOutcome:
        total_start = datetime.now().timestamp()
        log_event(self.logger, "pda_execute_start")

        try:
            plan = self.compiler.compile_plan(strategy_output)
            policy_plan = self._build_policy_plan_payload(plan)
            policies = [self.guard.validate_policy(policy) for policy in plan.policies]
        except Exception as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            recommended_consumer = self._classify_feedback_consumer(
                detail=str(exc),
                policy=None,
                phase="compile",
            )
            return self._build_failure_outcome(
                detail=str(exc),
                correction_suggestion=self._build_correction_suggestion(
                    recommended_consumer=recommended_consumer,
                    phase="compile",
                ),
                recommended_consumer=recommended_consumer,
                feedback_payload={"phase": "compile", "error": str(exc)},
                dispatch_attempts=0,
            )

        try:

            execution_receipts = []
            assurance_results = []
            total_dispatch_attempts = 0
            for index, policy in enumerate(policies, start=1):
                policy_start = datetime.now().timestamp()
                dispatch_result, dispatch_attempts = self._dispatch_single_policy(
                    policy=policy,
                    session_id=policy_plan.session_id,
                    snapshot_id=policy_plan.snapshot_id,
                )
                total_dispatch_attempts += dispatch_attempts
                execution_receipts.append(dispatch_result)

                verdict = self.assurance_evaluator.evaluate(policy=policy, snapshot_id=policy_plan.snapshot_id)
                assurance_results.append(verdict.model_dump(mode="json"))
                if verdict.status == "violated":
                    raise ExecutionDecisionError(
                        f"policy {policy['policy_id']} assurance violated for flow {policy.get('flow_id')}",
                        recommended_consumer="optimization_strategy",
                        correction_suggestion=self._build_correction_suggestion(
                            recommended_consumer="optimization_strategy",
                            phase="assurance",
                        ),
                        feedback_payload={
                            "phase": "assurance",
                            "policy_id": policy.get("policy_id"),
                            "policy_type": policy.get("policy_type"),
                            "flow_id": policy.get("flow_id"),
                            "assurance_result": verdict.model_dump(mode="json"),
                        },
                        dispatch_attempts=total_dispatch_attempts,
                    )
                if verdict.status == "failed":
                    raise ExecutionDecisionError(
                        f"policy {policy['policy_id']} assurance failed: {verdict.reason}",
                        recommended_consumer="optimization_strategy",
                        correction_suggestion=self._build_correction_suggestion(
                            recommended_consumer="optimization_strategy",
                            phase="assurance",
                        ),
                        feedback_payload={
                            "phase": "assurance",
                            "policy_id": policy.get("policy_id"),
                            "policy_type": policy.get("policy_type"),
                            "flow_id": policy.get("flow_id"),
                            "assurance_result": verdict.model_dump(mode="json"),
                        },
                        dispatch_attempts=total_dispatch_attempts,
                    )

                log_timing(
                    self.logger,
                    "pda_policy_total",
                    datetime.now().timestamp() - policy_start,
                    policy_index=index,
                    policy_id=policy["policy_id"],
                )

            commit_supi = plan.supi or policies[0]["supi"]
            self._commit_successful_policies(commit_supi, policies)

            performance_metrics = json.dumps(
                {
                    "policy_plan": policy_plan.model_dump(mode="json"),
                    "dispatch_results": execution_receipts,
                    "assurance_results": assurance_results,
                    "dispatch_attempts": total_dispatch_attempts,
                },
                ensure_ascii=False,
            )
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="success")
            return ExecutionOutcome(
                execution_status="Success",
                performance_metrics=performance_metrics,
                violation_details="None",
                correction_suggestion="None",
                recommended_consumer="none",
                recommended_action="commit",
                feedback_payload={},
                dispatch_attempts=total_dispatch_attempts,
            )
        except ExecutionDecisionError as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return self._build_failure_outcome(
                detail=str(exc),
                correction_suggestion=exc.correction_suggestion,
                recommended_consumer=exc.recommended_consumer,
                feedback_payload=exc.feedback_payload,
                dispatch_attempts=exc.dispatch_attempts,
            )
        except Exception as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return self._build_failure_outcome(
                detail=str(exc),
                correction_suggestion="Check policy payload, guard validation, PCF ack, or telemetry path.",
                recommended_consumer="optimization_strategy",
                feedback_payload={"phase": "runtime", "error": str(exc)},
                dispatch_attempts=0,
            )
