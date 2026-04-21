from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from domain.policy_plan import PolicyPlan
from domain.policy_compiler import PolicyCompiler
from domain.policy_guard import PolicyGuard
from agents.tools.db_tool import get_latest_snapshot_metadata, update_scenario_in_db
from agents.tools.init_scenario import cache_scenario, get_cached_control_scenario, get_current_scenario
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
    failure_scope: str = "none"
    feedback_payload: Dict[str, Any] = field(default_factory=dict)
    dispatch_attempts: int = 0
    committed_snapshot_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "performance_metrics": self.performance_metrics,
            "violation_details": self.violation_details,
            "correction_suggestion": self.correction_suggestion,
            "recommended_consumer": self.recommended_consumer,
            "recommended_action": self.recommended_action,
            "failure_scope": self.failure_scope,
            "feedback_payload": self.feedback_payload,
            "dispatch_attempts": self.dispatch_attempts,
            "committed_snapshot_id": self.committed_snapshot_id,
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
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

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
        persist_am_policy_association: Optional[Callable[..., bool]] = None,
        record_mobility_event: Optional[Callable[..., bool]] = None,
        persist_serving_nf_binding: Optional[Callable[..., bool]] = None,
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
        self.persist_am_policy_association = persist_am_policy_association
        self.record_mobility_event = record_mobility_event
        self.persist_serving_nf_binding = persist_serving_nf_binding
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
                policy,
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
        failure_scope: str = "none",
        feedback_payload: Optional[Dict[str, Any]] = None,
        dispatch_attempts: int = 0,
    ) -> ExecutionOutcome:
        metrics = json.dumps(
            {
                "dispatch_attempts": dispatch_attempts,
                "recommended_consumer": recommended_consumer,
                "failure_scope": failure_scope,
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
            failure_scope=failure_scope,
            feedback_payload=feedback_payload or {},
            dispatch_attempts=dispatch_attempts,
            committed_snapshot_id="",
        )

    @classmethod
    def _failure_scope_from_policies(cls, policies: list[Dict[str, Any]]) -> str:
        if not policies:
            return "none"
        scopes = {
            "mobility" if str(item.get("policy_type") or "").strip() == cls.AM_POLICY_TYPE else "qos"
            for item in policies
        }
        if scopes == {"qos"}:
            return "qos"
        if scopes == {"mobility"}:
            return "mobility"
        return "mixed"

    @classmethod
    def _failure_scope_from_entries(cls, entries: list[Dict[str, Any]]) -> str:
        policy_types = [entry.get("policy_type") for entry in entries if isinstance(entry, dict)]
        return cls._failure_scope_from_policies([{"policy_type": item} for item in policy_types if item])

    @classmethod
    def _select_recommended_consumer(cls, failures: list[Dict[str, Any]]) -> str:
        consumers = [str(item.get("recommended_consumer") or "").strip() for item in failures if isinstance(item, dict)]
        normalized = {item for item in consumers if item}
        if not normalized:
            return "optimization_strategy"
        if normalized == {"intent_encoding"}:
            return "intent_encoding"
        return "optimization_strategy"

    @staticmethod
    def _build_partial_correction_suggestion(*, recommended_consumer: str, failure_scope: str) -> str:
        if recommended_consumer == "intent_encoding":
            return "Re-check UE, app, or flow binding before retrying the failed execution scope."
        if failure_scope == "mobility":
            return "Revise the AM policy payload or mobility constraints before retrying mobility execution."
        if failure_scope == "qos":
            return "Revise the QoS policy payload or optimization output before retrying QoS execution."
        return "Revise the failed policy subset before the next execution round."

    def _commit_successful_policies(self, supi: str, policies: list[Dict[str, Any]], *, session_id: str, snapshot_id: str) -> str:
        existing = self.load_ue_context(supi) or {}
        access_policies = [policy for policy in policies if str(policy.get("policy_type") or "").strip() != self.AM_POLICY_TYPE]
        mobility_policies = [policy for policy in policies if str(policy.get("policy_type") or "").strip() == self.AM_POLICY_TYPE]

        if access_policies:
            merged = self.compiler.merge_policies_into_context(existing, access_policies)
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

        for policy in mobility_policies:
            policy_details = policy.get("policy_details") or {}
            request = policy_details.get("request")
            if not isinstance(request, dict):
                raise RuntimeError(f"AM policy {policy.get('policy_id')} missing request payload")
            association_id = str(policy.get("policy_id") or "").strip()
            if not association_id:
                raise RuntimeError("AM policy missing top-level policy_id")
            if self.persist_am_policy_association is None or self.record_mobility_event is None:
                raise RuntimeError("AM policy persistence callbacks are not configured")

            persisted = self.persist_am_policy_association(
                supi=supi,
                pol_asso_id=association_id,
                association_request=request,
                association_policy=policy_details,
                status="applied",
                trigger_event="JOINT_CONTROL_REEVALUATION",
                session_id=session_id,
                snapshot_id=snapshot_id,
                round_index=1,
            )
            if not persisted:
                raise RuntimeError(f"failed to persist AM policy association for {association_id}")

            am_policy_context = dict(existing.get("amPolicyContext") or {})
            associations = dict(am_policy_context.get("associations") or {})
            associations[association_id] = policy_details
            am_policy_context["associations"] = associations
            am_policy_context["allowedSnssais"] = request.get("allowedSnssais") or []
            am_policy_context["targetSnssais"] = request.get("targetSnssais") or []
            am_policy_context["mappingSnssais"] = request.get("mappingSnssais") or []
            am_policy_context["rfsp"] = request.get("rfsp")
            am_policy_context["ueAmbr"] = request.get("ueAmbr") or policy_details.get("ueAmbr")
            am_policy_context["ueSliceMbrs"] = policy_details.get("ueSliceMbrs") or []
            am_policy_context["pras"] = policy_details.get("pras") or {}

            updated = self.persist_ue_context(
                supi=supi,
                access_mobility_context=existing.get("accessMobilityContext") or {},
                am_policy_context=am_policy_context,
                serving_nf_context=existing.get("servingNfContext") or {},
                mobility_summary={
                    "currentAssociationId": association_id,
                    "currentTriggers": policy_details.get("triggers") or [],
                    "currentRfsp": request.get("rfsp"),
                    "lastUpdatedReason": "policy_dispatch",
                },
            )
            if not updated:
                raise RuntimeError(f"failed to update UE mobility context for {association_id}")

            if self.persist_serving_nf_binding is not None:
                serving_nf_context = existing.get("servingNfContext") or {}
                if serving_nf_context.get("amf_id") or serving_nf_context.get("amf_uri"):
                    self.persist_serving_nf_binding(
                        supi=supi,
                        nf_type="AMF",
                        nf_instance_id=str(serving_nf_context.get("amf_id") or ""),
                        nf_uri=str(serving_nf_context.get("amf_uri") or ""),
                        binding_info=serving_nf_context,
                    )

            recorded = self.record_mobility_event(
                supi=supi,
                session_id=session_id,
                snapshot_id=snapshot_id,
                event_type="JOINT_CONTROL_REEVALUATION",
                event_summary="AM policy association updated from PDA",
                event_payload={"association_id": association_id, "request": request, "policy": policy_details},
            )
            if not recorded:
                raise RuntimeError(f"failed to record mobility event for {association_id}")
        return "success"

    @staticmethod
    def _build_policy_plan_payload(plan: Any) -> PolicyPlan:
        return PolicyPlan(
            session_id=plan.session_id,
            snapshot_id=plan.snapshot_id,
            supi=plan.supi,
            policies=plan.policies,
        )

    @staticmethod
    def _find_app_for_patch(
        apps: list[Any],
        *,
        app_id: str,
        app_name: str,
        supi: str,
    ) -> Optional[Any]:
        normalized_app_id = str(app_id or "").strip()
        normalized_app_name = str(app_name or "").strip()
        normalized_supi = str(supi or "").strip()
        for app in apps:
            if normalized_app_id and str(getattr(app, "id", "") or "").strip() == normalized_app_id:
                return app
            if normalized_app_name and str(getattr(app, "name", "") or "").strip() == normalized_app_name:
                return app
        if normalized_supi:
            for app in apps:
                if str(getattr(app, "supi", "") or "").strip() == normalized_supi:
                    return app
        return None

    @staticmethod
    def _apply_flow_patch(flow: Any, patch: Dict[str, Any]) -> None:
        sla_patch = patch.get("sla") if isinstance(patch.get("sla"), dict) else {}
        allocation_patch = patch.get("allocation") if isinstance(patch.get("allocation"), dict) else {}
        telemetry_patch = patch.get("telemetry") if isinstance(patch.get("telemetry"), dict) else {}

        for attr, key in (
            ("bandwidth_ul", "bandwidth_ul"),
            ("bandwidth_dl", "bandwidth_dl"),
            ("guaranteed_bandwidth_ul", "guaranteed_bandwidth_ul"),
            ("guaranteed_bandwidth_dl", "guaranteed_bandwidth_dl"),
            ("latency", "latency"),
            ("jitter", "jitter"),
            ("loss_rate", "loss_rate"),
            ("priority", "priority"),
        ):
            if key in sla_patch and sla_patch[key] is not None:
                setattr(flow.sla, attr, sla_patch[key])

        for attr, key in (
            ("current_slice_snssai", "current_slice_snssai"),
            ("allocated_bandwidth_ul", "allocated_bandwidth_ul"),
            ("allocated_bandwidth_dl", "allocated_bandwidth_dl"),
        ):
            if key in allocation_patch:
                setattr(flow.allocation, attr, allocation_patch[key])

        for attr, key in (
            ("throughput_ul", "throughput_ul"),
            ("throughput_dl", "throughput_dl"),
            ("latency", "latency"),
            ("jitter", "jitter"),
            ("loss_rate", "loss_rate"),
            ("packet_sent", "packet_sent"),
            ("packet_received", "packet_received"),
        ):
            if key in telemetry_patch:
                setattr(flow.telemetry, attr, telemetry_patch[key])

    def _commit_snapshot_writeback(self, strategy_output: Any) -> str:
        planning_metadata = getattr(strategy_output, "planning_metadata", {}) or {}
        if not isinstance(planning_metadata, dict):
            return ""
        writeback_patch = planning_metadata.get("snapshot_writeback_patch") or {}
        if not isinstance(writeback_patch, dict):
            return ""

        qos_plan = writeback_patch.get("qos_plan") or {}
        if not isinstance(qos_plan, dict):
            return ""

        target_app_patch = qos_plan.get("target_app") or {}
        impacted_flows = qos_plan.get("impacted_flows") or []
        if not isinstance(target_app_patch, dict) and not isinstance(impacted_flows, list):
            return ""

        apps, slices, nodes = deepcopy(get_current_scenario())
        cached_control = get_cached_control_scenario()
        flow_patches: list[tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]] = []

        if isinstance(target_app_patch, dict):
            target_flow_patches = target_app_patch.get("flows") or []
            if isinstance(target_flow_patches, list):
                for item in target_flow_patches:
                    if isinstance(item, dict):
                        flow_patches.append(
                            (
                                str(target_app_patch.get("id") or "").strip() or None,
                                str(target_app_patch.get("name") or "").strip() or None,
                                str(target_app_patch.get("supi") or getattr(strategy_output, "supi", "") or "").strip() or None,
                                item,
                            )
                        )

        if isinstance(impacted_flows, list):
            for item in impacted_flows:
                if not isinstance(item, dict):
                    continue
                flow_payload = item.get("flow")
                if not isinstance(flow_payload, dict):
                    continue
                flow_patches.append(
                    (
                        str(item.get("app_id") or "").strip() or None,
                        str(item.get("app_name") or "").strip() or None,
                        str(item.get("supi") or "").strip() or None,
                        flow_payload,
                    )
                )

        if not flow_patches:
            return ""

        for app_id, app_name, supi, flow_patch in flow_patches:
            flow_id = str(flow_patch.get("id") or "").strip()
            if not flow_id:
                continue
            app = self._find_app_for_patch(apps, app_id=app_id or "", app_name=app_name or "", supi=supi or "")
            if app is None:
                continue
            target_flow = next((flow for flow in app.flows if str(getattr(flow, "id", "") or "").strip() == flow_id), None)
            if target_flow is None:
                continue
            self._apply_flow_patch(target_flow, flow_patch)

        persisted = update_scenario_in_db(
            apps,
            slices,
            nodes,
            mobility_data=cached_control.get("mobility") or [],
            policy_data=cached_control.get("policy_state") or {},
            trigger="Workflow-Commit",
        )
        if not persisted:
            raise RuntimeError("failed to persist committed scenario snapshot")
        cache_scenario(
            apps,
            slices,
            nodes,
            cached_control.get("mobility") or [],
            cached_control.get("policy_state") or {},
        )
        latest_snapshot = get_latest_snapshot_metadata() or {}
        return str(latest_snapshot.get("snapshot_id") or "").strip()

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
                failure_scope="compile",
                feedback_payload={"phase": "compile", "error": str(exc)},
                dispatch_attempts=0,
            )

        try:

            execution_receipts = []
            assurance_results = []
            total_dispatch_attempts = 0
            failures: list[Dict[str, Any]] = []
            for index, policy in enumerate(policies, start=1):
                policy_start = datetime.now().timestamp()
                try:
                    dispatch_result, dispatch_attempts = self._dispatch_single_policy(
                        policy=policy,
                        session_id=policy_plan.session_id,
                        snapshot_id=policy_plan.snapshot_id,
                    )
                    total_dispatch_attempts += dispatch_attempts
                    dispatch_receipt = {
                        **dispatch_result,
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                    }
                    execution_receipts.append(dispatch_receipt)
                except ExecutionDecisionError as exc:
                    total_dispatch_attempts += exc.dispatch_attempts
                    failure_receipt = {
                        "status": "failed",
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                        "error": str(exc),
                        "phase": "dispatch",
                    }
                    execution_receipts.append(failure_receipt)
                    failures.append(
                        {
                            "policy_id": policy.get("policy_id"),
                            "policy_type": policy.get("policy_type"),
                            "flow_id": policy.get("flow_id"),
                            "phase": "dispatch",
                            "error": str(exc),
                            "recommended_consumer": exc.recommended_consumer,
                            "correction_suggestion": exc.correction_suggestion,
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
                        failure_scope=(
                            "mobility"
                            if str(policy.get("policy_type") or "").strip() == self.AM_POLICY_TYPE
                            else "qos"
                        ),
                        error_summary=str(exc),
                        recommended_consumer=exc.recommended_consumer,
                    )
                    continue

                verdict = self.assurance_evaluator.evaluate(policy=policy, snapshot_id=policy_plan.snapshot_id)
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
                            "recommended_consumer": "optimization_strategy",
                            "correction_suggestion": self._build_correction_suggestion(
                                recommended_consumer="optimization_strategy",
                                phase="assurance",
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
                recommended_consumer = self._select_recommended_consumer(failures)
                correction_suggestion = self._build_partial_correction_suggestion(
                    recommended_consumer=recommended_consumer,
                    failure_scope=failure_scope,
                )
                performance_metrics = json.dumps(
                    {
                        "policy_plan": policy_plan.model_dump(mode="json"),
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
                has_assurance_failure = any(
                    str(item.get("phase") or "").strip() == "assurance"
                    for item in failures
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
                    recommended_consumer=recommended_consumer,
                    failed_policy_count=len(failed_policy_ids),
                    successful_policy_count=len(successful_policy_ids),
                )
                return ExecutionOutcome(
                    execution_status=execution_status,
                    performance_metrics=performance_metrics,
                    violation_details=violation_details or "execution failed",
                    correction_suggestion=correction_suggestion,
                    recommended_consumer=recommended_consumer,
                    recommended_action="feedback",
                    failure_scope=failure_scope,
                    feedback_payload={
                        "phase": "execution",
                        "failure_scope": failure_scope,
                        "successful_policy_ids": successful_policy_ids,
                        "failed_policy_ids": failed_policy_ids,
                        "failures": failures,
                    },
                    dispatch_attempts=total_dispatch_attempts,
                )

            commit_supi = plan.supi or policies[0]["supi"]
            self._commit_successful_policies(
                commit_supi,
                policies,
                session_id=policy_plan.session_id,
                snapshot_id=policy_plan.snapshot_id,
            )
            committed_snapshot_id = self._commit_snapshot_writeback(strategy_output)

            performance_metrics = json.dumps(
                {
                    "policy_plan": policy_plan.model_dump(mode="json"),
                    "dispatch_results": execution_receipts,
                    "assurance_results": assurance_results,
                    "dispatch_attempts": total_dispatch_attempts,
                    "committed_snapshot_id": committed_snapshot_id,
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
                correction_suggestion="None",
                recommended_consumer="none",
                recommended_action="commit",
                failure_scope="none",
                feedback_payload={},
                dispatch_attempts=total_dispatch_attempts,
                committed_snapshot_id=committed_snapshot_id,
            )
        except ExecutionDecisionError as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return self._build_failure_outcome(
                detail=str(exc),
                correction_suggestion=exc.correction_suggestion,
                recommended_consumer=exc.recommended_consumer,
                failure_scope="mixed",
                feedback_payload=exc.feedback_payload,
                dispatch_attempts=exc.dispatch_attempts,
            )
        except Exception as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return self._build_failure_outcome(
                detail=str(exc),
                correction_suggestion="Check policy payload, guard validation, PCF ack, or telemetry path.",
                recommended_consumer="optimization_strategy",
                failure_scope="mixed",
                feedback_payload={"phase": "runtime", "error": str(exc)},
                dispatch_attempts=0,
            )
