from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from shared.logging import log_event, log_timing, setup_logger

from .assurance_evaluator import AssuranceEvaluator
from .policy_batch_executor import PolicyBatchExecutor
from .policy_compiler import PolicyCompiler


@dataclass(frozen=True)
class ExecutionOutcome:
    execution_status: str
    performance_metrics: str
    violation_details: str
    failure_scope: str = "none"
    feedback_payload: Dict[str, Any] = field(default_factory=dict)
    dispatch_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "performance_metrics": self.performance_metrics,
            "violation_details": self.violation_details,
            "failure_scope": self.failure_scope,
            "feedback_payload": self.feedback_payload,
            "dispatch_attempts": self.dispatch_attempts,
        }


class ExecutionController:
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

    def __init__(
        self,
        *,
        compiler: Optional[PolicyCompiler] = None,
        dispatch_policy: Callable[..., Dict[str, Any]],
        assurance_evaluator: AssuranceEvaluator,
        commit_policies: Callable[..., None],
        logger: Any = None,
    ) -> None:
        self.compiler = compiler or PolicyCompiler()
        self.commit_policies = commit_policies
        self.logger = logger or setup_logger(self.__class__.__name__, default_msg_color="\033[92m")
        self.policy_batch_executor = PolicyBatchExecutor(
            dispatch_policy=dispatch_policy,
            assurance_evaluator=assurance_evaluator,
            logger=self.logger,
            mobility_policy_type=self.AM_POLICY_TYPE,
        )

    @staticmethod
    def _build_failure_outcome(
        *,
        detail: str,
        failure_scope: str = "none",
        feedback_payload: Optional[Dict[str, Any]] = None,
        dispatch_attempts: int = 0,
    ) -> ExecutionOutcome:
        metrics = json.dumps(
            {
                "dispatch_attempts": dispatch_attempts,
                "failure_scope": failure_scope,
                "feedback_payload": feedback_payload or {},
            },
            ensure_ascii=False,
        )
        return ExecutionOutcome(
            execution_status="Failed",
            performance_metrics=metrics,
            violation_details=detail,
            failure_scope=failure_scope,
            feedback_payload=feedback_payload or {},
            dispatch_attempts=dispatch_attempts,
        )

    def _failure_scope_from_policies(self, policies: list[Dict[str, Any]]) -> str:
        if not policies:
            return "none"
        scopes = {
            "mobility" if str(item.get("policy_type") or "").strip() == self.AM_POLICY_TYPE else "qos"
            for item in policies
        }
        if scopes == {"qos"}:
            return "qos"
        if scopes == {"mobility"}:
            return "mobility"
        return "mixed"

    def _failure_scope_from_entries(self, entries: list[Dict[str, Any]]) -> str:
        policy_types = [entry.get("policy_type") for entry in entries if isinstance(entry, dict)]
        return self._failure_scope_from_policies([{"policy_type": item} for item in policy_types if item])

    def execute(self, strategy_output: Any) -> ExecutionOutcome:
        total_start = datetime.now().timestamp()
        policies = list(getattr(strategy_output, "all_policies", []) or [])
        log_event(
            self.logger,
            "pda_execute_start",
            session_id=str(getattr(strategy_output, "session_id", "") or "").strip() or "<empty>",
            snapshot_id=str(getattr(strategy_output, "snapshot_id", "") or "").strip() or "<empty>",
            supi=str(getattr(strategy_output, "supi", "") or "").strip() or "<empty>",
            policy_count=len(policies),
        )

        try:
            plan = self.compiler.compile_plan(strategy_output)
            policy_plan_payload = {
                "session_id": plan.session_id,
                "snapshot_id": plan.snapshot_id,
                "supi": plan.supi,
                "policies": plan.policies,
            }
            policies = plan.policies
        except Exception as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return self._build_failure_outcome(
                detail=str(exc),
                failure_scope="compile",
                feedback_payload={
                    "phase": "compile",
                    "error": str(exc),
                },
                dispatch_attempts=0,
            )

        batch = self.policy_batch_executor.run(
            policies=policies,
            session_id=plan.session_id,
            snapshot_id=plan.snapshot_id,
        )
        execution_receipts = batch.execution_receipts
        assurance_results = batch.assurance_results
        total_dispatch_attempts = batch.total_dispatch_attempts
        failures = batch.failures

        if failures:
            successful_policy_ids = [
                str(item.get("policy_id") or "").strip()
                for item in execution_receipts
                if str(item.get("status") or "").strip().lower() == "success"
            ]
            failed_policy_ids = [
                str(item.get("policy_id") or "").strip()
                for item in failures
                if str(item.get("policy_id") or "").strip()
            ]
            failure_scope = self._failure_scope_from_entries(failures)
            performance_metrics = json.dumps(
                {
                    "policy_plan": policy_plan_payload,
                    "dispatch_results": execution_receipts,
                    "assurance_results": assurance_results,
                    "dispatch_attempts": total_dispatch_attempts,
                    "domain_receipts": {
                        "qos": [item for item in execution_receipts if item.get("policy_type") != self.AM_POLICY_TYPE],
                        "mobility": [item for item in execution_receipts if item.get("policy_type") == self.AM_POLICY_TYPE],
                    },
                    "failure_summaries": failures,
                },
                ensure_ascii=False,
            )
            has_assurance_failure = any(str(item.get("phase") or "").strip() == "assurance" for item in failures)
            retry_forbidden = any(
                isinstance(item.get("feedback_payload"), dict)
                and bool(item["feedback_payload"].get("retry_forbidden"))
                for item in failures
                if isinstance(item, dict)
            )
            execution_status = "Failed" if has_assurance_failure else ("Partial Success" if successful_policy_ids else "Failed")
            violation_details = "; ".join(
                str(item.get("error") or "").strip() for item in failures if str(item.get("error") or "").strip()
            )
            log_timing(
                self.logger,
                "pda_total",
                datetime.now().timestamp() - total_start,
                status="partial" if execution_status == "Partial Success" else "error",
                failure_scope=failure_scope,
                failed_policy_count=len(failed_policy_ids),
                successful_policy_count=len(successful_policy_ids),
            )
            return ExecutionOutcome(
                execution_status=execution_status,
                performance_metrics=performance_metrics,
                violation_details=violation_details or "execution failed",
                failure_scope=failure_scope,
                feedback_payload={
                    "phase": "execution",
                    "failure_scope": failure_scope,
                    "successful_policy_ids": successful_policy_ids,
                    "failed_policy_ids": failed_policy_ids,
                    "failures": failures,
                    "retry_forbidden": retry_forbidden,
                },
                dispatch_attempts=total_dispatch_attempts,
            )

        commit_supi = plan.supi or policies[0]["supi"]
        self.commit_policies(
            supi=commit_supi,
            policies=policies,
            session_id=plan.session_id,
            snapshot_id=plan.snapshot_id,
        )
        performance_metrics = json.dumps(
            {
                "policy_plan": policy_plan_payload,
                "dispatch_results": execution_receipts,
                "assurance_results": assurance_results,
                "dispatch_attempts": total_dispatch_attempts,
                "domain_receipts": {
                    "qos": [item for item in execution_receipts if item.get("policy_type") != self.AM_POLICY_TYPE],
                    "mobility": [item for item in execution_receipts if item.get("policy_type") == self.AM_POLICY_TYPE],
                },
            },
            ensure_ascii=False,
        )
        log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="success")
        return ExecutionOutcome(
            execution_status="Success",
            performance_metrics=performance_metrics,
            violation_details="None",
            failure_scope="none",
            feedback_payload={},
            dispatch_attempts=total_dispatch_attempts,
        )


__all__ = ["ExecutionOutcome", "ExecutionController"]
