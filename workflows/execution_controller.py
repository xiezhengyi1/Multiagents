from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "performance_metrics": self.performance_metrics,
            "violation_details": self.violation_details,
            "correction_suggestion": self.correction_suggestion,
        }


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
        logger: Any = None,
    ) -> None:
        self.compiler = compiler or PolicyCompiler()
        self.guard = guard or PolicyGuard()
        self.dispatch_policy = dispatch_policy
        self.assurance_evaluator = assurance_evaluator
        self.load_ue_context = load_ue_context
        self.load_ue_flow_catalog = load_ue_flow_catalog
        self.persist_ue_context = persist_ue_context
        self.logger = logger or setup_logger(self.__class__.__name__, default_msg_color="\033[92m")

    def _dispatch_single_policy(
        self,
        *,
        policy: Dict[str, Any],
        session_id: str,
        snapshot_id: str,
    ) -> Dict[str, Any]:
        request_id = f"req-{uuid.uuid4()}"
        result = self.dispatch_policy(
            policy["policy_type"],
            policy["policy_details"],
            request_id=request_id,
            session_id=session_id or None,
            snapshot_id=snapshot_id or None,
        )
        if result.get("status") != "success":
            error = str(result.get("error") or "unknown dispatch error").strip()
            raise RuntimeError(f"policy {policy['policy_id']} dispatch failed: {error}")

        ack = result.get("ack")
        if not isinstance(ack, dict):
            raise RuntimeError(f"policy {policy['policy_id']} missing ack in PCF response")
        if ack.get("completed") is not True:
            raise RuntimeError(
                f"policy {policy['policy_id']} ack incomplete: expected={ack.get('expected')}, "
                f"received={ack.get('received')}, completed={ack.get('completed')}"
            )

        return result

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

            execution_receipts = []
            assurance_results = []
            for index, policy in enumerate(policies, start=1):
                policy_start = datetime.now().timestamp()
                dispatch_result = self._dispatch_single_policy(
                    policy=policy,
                    session_id=policy_plan.session_id,
                    snapshot_id=policy_plan.snapshot_id,
                )
                execution_receipts.append(dispatch_result)

                verdict = self.assurance_evaluator.evaluate(policy=policy, snapshot_id=policy_plan.snapshot_id)
                assurance_results.append(verdict.model_dump(mode="json"))
                if verdict.status == "violated":
                    raise RuntimeError(f"policy {policy['policy_id']} assurance violated for flow {policy.get('flow_id')}")
                if verdict.status == "failed":
                    raise RuntimeError(f"policy {policy['policy_id']} assurance failed: {verdict.reason}")

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
                },
                ensure_ascii=False,
            )
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="success")
            return ExecutionOutcome(
                execution_status="Success",
                performance_metrics=performance_metrics,
                violation_details="None",
                correction_suggestion="None",
            )
        except Exception as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return ExecutionOutcome(
                execution_status="Failed",
                performance_metrics="N/A",
                violation_details=str(exc),
                correction_suggestion="Check policy payload, guard validation, PCF ack, or telemetry path.",
            )
