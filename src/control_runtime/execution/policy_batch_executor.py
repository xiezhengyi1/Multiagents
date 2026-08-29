"""Dispatch and assure an already-validated policy batch."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from shared.logging import log_timing

from .assurance_evaluator import AssuranceEvaluator, AssuranceVerdict


class ExecutionDecisionError(RuntimeError):
    """A terminal decision made while dispatching one policy."""

    def __init__(
        self,
        message: str,
        *,
        feedback_payload: Optional[Dict[str, Any]] = None,
        dispatch_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.feedback_payload = feedback_payload or {}
        self.dispatch_attempts = dispatch_attempts


@dataclass(frozen=True)
class PolicyExecutionBatch:
    """The complete dispatch and assurance record for one policy plan."""

    execution_receipts: list[Dict[str, Any]]
    assurance_results: list[Dict[str, Any]]
    failures: list[Dict[str, Any]]
    total_dispatch_attempts: int


class PolicyBatchExecutor:
    """Execute policy dispatches and collect their post-apply assurance.

    The executor receives all collaborators needed by the batch protocol.  It
    does not own policy compilation, UE-context persistence, or snapshot
    writeback, so it can be tested independently of controller orchestration.
    """

    def __init__(
        self,
        *,
        dispatch_policy: Callable[..., Dict[str, Any]],
        assurance_evaluator: AssuranceEvaluator,
        logger: Any,
        mobility_policy_type: str,
    ) -> None:
        self.dispatch_policy = dispatch_policy
        self.assurance_evaluator = assurance_evaluator
        self.logger = logger
        self.mobility_policy_type = mobility_policy_type

    @staticmethod
    def _policy_flow_id(policy: Optional[Dict[str, Any]]) -> str:
        if not isinstance(policy, dict):
            return ""
        policy_details = policy.get("policy_details")
        detail_flow_id = policy_details.get("flow_id") if isinstance(policy_details, dict) else ""
        return str(policy.get("flow_id") or detail_flow_id or "").strip()

    @staticmethod
    def _normalize_snssai(value: Any) -> str:
        if isinstance(value, dict):
            try:
                sst = int(value.get("sst"))
            except (TypeError, ValueError):
                return ""
            sd = str(value.get("sd") or "").strip().lower()
            if not sd or len(sd) != 6 or any(char not in "0123456789abcdef" for char in sd):
                return ""
            return f"{sst:02x}{sd}"
        return str(value or "").strip().lower()

    @classmethod
    def _slice_execution_error(
        cls,
        *,
        policy: Dict[str, Any],
        dispatch_result: Dict[str, Any],
    ) -> tuple[str | None, Dict[str, Any]]:
        policy_details = dispatch_result.get("policy_details")
        if not isinstance(policy_details, dict):
            policy_details = policy.get("policy_details") if isinstance(policy.get("policy_details"), dict) else {}
        upstream_context = policy_details.get("upstreamSmPolicyContextData")
        if not isinstance(upstream_context, dict):
            return None, {}
        requested_slice = cls._normalize_snssai(upstream_context.get("sliceInfo"))
        if not requested_slice:
            return None, {}

        monitoring_data = dispatch_result.get("monitoring_data")
        observed_allocation = monitoring_data.get("observed_allocation") if isinstance(monitoring_data, dict) else {}
        observed_slice = cls._normalize_snssai(
            observed_allocation.get("current_slice_snssai") if isinstance(observed_allocation, dict) else None
        )
        evidence = {
            "requested_slice_snssai": requested_slice,
            "observed_slice_snssai": observed_slice or None,
        }
        if not observed_slice:
            return (
                f"policy {policy['policy_id']} dispatch was acknowledged but ns-3 returned no observed slice state "
                f"for requested slice {requested_slice}",
                evidence,
            )
        if observed_slice != requested_slice:
            return (
                f"policy {policy['policy_id']} dispatch was acknowledged but ns-3 remained on slice "
                f"{observed_slice} instead of requested slice {requested_slice}",
                evidence,
            )
        return None, evidence

    def _is_mobility_policy(self, policy: Optional[Dict[str, Any]]) -> bool:
        return isinstance(policy, dict) and str(policy.get("policy_type") or "").strip() == self.mobility_policy_type

    @staticmethod
    def _dispatch_failure_detail(dispatch_result: Dict[str, Any]) -> str:
        explicit_error = str(dispatch_result.get("error") or "").strip()
        if explicit_error:
            return explicit_error
        diagnostics = {
            key: dispatch_result.get(key)
            for key in ("phase", "status_code", "response_code", "operation_id")
            if dispatch_result.get(key) not in (None, "")
        }
        if diagnostics:
            details = ", ".join(f"{key}={value}" for key, value in diagnostics.items())
            return f"gateway returned a failed response without an error field ({details})"
        return "gateway returned a failed response without an error field"

    def _dispatch_single_policy(
        self,
        *,
        policy: Dict[str, Any],
        session_id: str,
        snapshot_id: str,
    ) -> tuple[Dict[str, Any], int]:
        result = self.dispatch_policy(
            policy["policy_type"],
            policy,
            request_id=f"req-{uuid.uuid4()}",
            session_id=session_id or None,
            snapshot_id=snapshot_id or None,
        )
        last_result = result if isinstance(result, dict) else {"status": "failed", "error": str(result)}
        pda_status = str(last_result.get("status") or "").strip().lower()
        pda_phase = str(last_result.get("phase") or "").strip().lower()
        # The gateway acknowledges a synchronous dispatch as ``success`` and
        # an asynchronously-polled completion as ``applied`` (or an equivalent
        # terminal success label).  Normalize those labels for the rest of the
        # execution pipeline, which deliberately uses one internal success
        # vocabulary.
        if pda_status in {"success", "applied", "completed", "succeeded"}:
            if pda_status != "success":
                last_result = {**last_result, "status": "success", "pda_status": pda_status}
            slice_error, slice_evidence = self._slice_execution_error(
                policy=policy,
                dispatch_result=last_result,
            )
            if slice_error:
                raise ExecutionDecisionError(
                    slice_error,
                    feedback_payload={
                        "phase": "dispatch",
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": self._policy_flow_id(policy),
                        "error": slice_error,
                        "slice_execution": slice_evidence,
                        "last_dispatch_result": last_result,
                    },
                    dispatch_attempts=1,
                )
            return last_result, 1
        pda_wait_expired = pda_status == "pending" or pda_phase in {"pda_deadline", "ns3_timeout", "waiting_for_ns3"}
        if pda_wait_expired:
            operation_id = str(last_result.get("operation_id") or "").strip()
            failure_message = (
                f"policy {policy['policy_id']} did not reach a terminal PDA status before the dispatch deadline"
                f"{f' (operation_id={operation_id})' if operation_id else ''}; "
                "terminate this case without retrying or redispatching the policy"
            )
            phase = pda_phase if pda_phase in {"pda_deadline", "ns3_timeout", "waiting_for_ns3"} else "pda_pending_timeout"
        else:
            dispatch_detail = self._dispatch_failure_detail(last_result)
            failure_message = f"policy {policy['policy_id']} dispatch failed: {dispatch_detail}"
            phase = str(last_result.get("phase") or "dispatch")
        raise ExecutionDecisionError(
            failure_message,
            feedback_payload={
                "phase": phase,
                "policy_id": policy.get("policy_id"),
                "policy_type": policy.get("policy_type"),
                "flow_id": self._policy_flow_id(policy),
                "error": failure_message,
                "last_dispatch_result": last_result,
                "retry_forbidden": pda_wait_expired,
            },
            dispatch_attempts=1,
        )

    def _post_apply_assurance_verdict(
        self,
        *,
        policy: Dict[str, Any],
        dispatch_result: Dict[str, Any],
        fallback_snapshot_id: str,
    ) -> AssuranceVerdict | None:
        """Prefer the gateway's same-tick compliance checks over a second sample."""
        if str(policy.get("target_type") or "").strip() != "flow":
            return None
        monitoring = dispatch_result.get("monitoring_data")
        monitoring = monitoring if isinstance(monitoring, dict) else {}
        checks = monitoring.get("qos_parameter_checks")
        applied_tick = monitoring.get("latest_tick", dispatch_result.get("applied_tick", ""))
        metrics = {
            "snapshot_id": f"{fallback_snapshot_id}@ns3-tick-{applied_tick}",
            "source": "policy_post_apply_receipt",
            "qos_parameter_checks": checks,
            "observed_telemetry": monitoring.get("observed_telemetry", {}),
            "run_id": monitoring.get("run_id"),
            "reset_generation": monitoring.get("reset_generation"),
            "baseline_tick": monitoring.get("baseline_tick"),
        }
        try:
            baseline_tick = int(monitoring["baseline_tick"])
            latest_tick = int(monitoring["latest_tick"])
            sla_epoch_tick = int((monitoring.get("observed_telemetry") or {})["sla_epoch_start_tick"])
            contract_valid = (
                monitoring.get("assurance_contract_version") == 1
                and bool(str(monitoring.get("run_id") or "").strip())
                and int(monitoring["reset_generation"]) >= 0
                and latest_tick > baseline_tick
                and sla_epoch_tick > baseline_tick
            )
        except (KeyError, TypeError, ValueError):
            contract_valid = False
        if not contract_valid or not isinstance(checks, list) or not checks:
            metrics["source"] = "unverified_gateway_receipt"
            return AssuranceVerdict(
                policy_id=str(policy.get("policy_id") or "").strip(),
                flow_id=self._policy_flow_id(policy) or None,
                status="failed",
                reason="Gateway response lacks same-generation post-policy ns-3 assurance evidence.",
                metrics=metrics,
            )
        satisfied = all(isinstance(check, dict) and bool(check.get("satisfied")) for check in checks)
        return AssuranceVerdict(
            policy_id=str(policy.get("policy_id") or "").strip(),
            flow_id=self._policy_flow_id(policy) or None,
            status="satisfied" if satisfied else "violated",
            reason=(
                "Post-apply ns-3 checks satisfy the policy requirements."
                if satisfied
                else "Post-apply ns-3 checks violate the policy requirements."
            ),
            metrics=metrics,
        )

    def run(
        self,
        *,
        policies: list[Dict[str, Any]],
        session_id: str,
        snapshot_id: str,
    ) -> PolicyExecutionBatch:
        """Dispatch every policy once and evaluate its post-apply assurance."""
        execution_receipts: list[Dict[str, Any]] = []
        assurance_results: list[Dict[str, Any]] = []
        failures: list[Dict[str, Any]] = []
        total_dispatch_attempts = 0
        for index, policy in enumerate(policies, start=1):
            policy_start = datetime.now().timestamp()
            try:
                dispatch_result, dispatch_attempts = self._dispatch_single_policy(
                    policy=policy,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                )
                total_dispatch_attempts += dispatch_attempts
                execution_receipts.append(
                    {
                        **dispatch_result,
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                    }
                )
            except ExecutionDecisionError as exc:
                total_dispatch_attempts += exc.dispatch_attempts
                feedback_payload = dict(exc.feedback_payload or {})
                execution_receipts.append(
                    {
                        "status": "failed",
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                        "error": str(exc),
                        "phase": "dispatch",
                        "dispatch_attempts": exc.dispatch_attempts,
                        "feedback_payload": feedback_payload,
                        "last_dispatch_result": feedback_payload.get("last_dispatch_result"),
                    }
                )
                failures.append(
                    {
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                        "phase": "dispatch",
                        "error": str(exc),
                        "dispatch_attempts": exc.dispatch_attempts,
                        "feedback_payload": exc.feedback_payload,
                    }
                )
                log_timing(
                    self.logger,
                    "pda_policy_total",
                    datetime.now().timestamp() - policy_start,
                    policy_index=index,
                    policy_id=policy["policy_id"],
                    status="dispatch_failed",
                    failure_scope="mobility" if self._is_mobility_policy(policy) else "qos",
                    error_summary=str(exc),
                )
                continue

            verdict = self._post_apply_assurance_verdict(
                policy=policy,
                dispatch_result=dispatch_result,
                fallback_snapshot_id=snapshot_id,
            ) or self.assurance_evaluator.evaluate(
                policy=policy,
                snapshot_id=snapshot_id,
            )
            verdict_payload = verdict.model_dump(mode="json")
            verdict_payload.setdefault("policy_type", policy.get("policy_type"))
            assurance_results.append(verdict_payload)
            if verdict.status in {"violated", "failed"}:
                failures.append(
                    {
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                        "phase": "assurance",
                        "error": (
                            f"policy {policy['policy_id']} assurance violated for flow {policy.get('flow_id')}"
                            if verdict.status == "violated"
                            else f"policy {policy['policy_id']} assurance failed: {verdict.reason}"
                        ),
                        "assurance_result": verdict_payload,
                    }
                )
            log_timing(
                self.logger,
                "pda_policy_total",
                datetime.now().timestamp() - policy_start,
                policy_index=index,
                policy_id=policy["policy_id"],
            )
        return PolicyExecutionBatch(
            execution_receipts=execution_receipts,
            assurance_results=assurance_results,
            failures=failures,
            total_dispatch_attempts=total_dispatch_attempts,
        )


__all__ = ["ExecutionDecisionError", "PolicyBatchExecutor", "PolicyExecutionBatch"]
