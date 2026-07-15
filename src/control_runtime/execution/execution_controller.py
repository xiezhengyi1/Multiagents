from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from shared.logging import log_event, log_timing, setup_logger

from ..domain.policy_compiler import PolicyCompiler
from ..domain.policy_guard import PolicyGuard
from ..domain.policy_plan import PolicyPlan
from .assurance_evaluator import AssuranceEvaluator


@dataclass(frozen=True)
class ExecutionOutcome:
    execution_status: str
    performance_metrics: str
    violation_details: str
    failure_scope: str = "none"
    feedback_payload: Dict[str, Any] = field(default_factory=dict)
    dispatch_attempts: int = 0
    committed_snapshot_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "performance_metrics": self.performance_metrics,
            "violation_details": self.violation_details,
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
        feedback_payload: Optional[Dict[str, Any]] = None,
        dispatch_attempts: int = 0,
    ) -> None:
        super().__init__(message)
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
        load_ue_context: Callable[..., Optional[Dict[str, Any]]],
        load_ue_flow_catalog: Callable[..., Dict[str, Any]],
        persist_ue_context: Callable[..., bool],
        persist_am_policy_association: Optional[Callable[..., bool]] = None,
        record_mobility_event: Optional[Callable[..., bool]] = None,
        persist_serving_nf_binding: Optional[Callable[..., bool]] = None,
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
        self.logger = logger or setup_logger(self.__class__.__name__, default_msg_color="\033[92m")

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
            committed_snapshot_id="",
        )

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
        return isinstance(policy, dict) and str(policy.get("policy_type") or "").strip() == self.AM_POLICY_TYPE

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
        if last_result.get("status") == "success":
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
        failure_message = (
            f"policy {policy['policy_id']} dispatch failed: "
            f"{str(last_result.get('error') or 'unknown dispatch error').strip()}"
        )
        raise ExecutionDecisionError(
            failure_message,
            feedback_payload={
                "phase": "dispatch",
                "policy_id": policy.get("policy_id"),
                "policy_type": policy.get("policy_type"),
                "flow_id": self._policy_flow_id(policy),
                "error": failure_message,
                "last_dispatch_result": last_result,
            },
            dispatch_attempts=1,
        )

    def _run_policies(self, *, policies: list[Dict[str, Any]], policy_plan: PolicyPlan) -> Dict[str, Any]:
        batch = {
            "execution_receipts": [],
            "assurance_results": [],
            "failures": [],
            "total_dispatch_attempts": 0,
        }
        for index, policy in enumerate(policies, start=1):
            policy_start = datetime.now().timestamp()
            try:
                dispatch_result, dispatch_attempts = self._dispatch_single_policy(
                    policy=policy,
                    session_id=policy_plan.session_id,
                    snapshot_id=policy_plan.snapshot_id,
                )
                batch["total_dispatch_attempts"] += dispatch_attempts
                batch["execution_receipts"].append(
                    {
                        **dispatch_result,
                        "policy_id": policy.get("policy_id"),
                        "policy_type": policy.get("policy_type"),
                        "flow_id": policy.get("flow_id"),
                    }
                )
            except ExecutionDecisionError as exc:
                batch["total_dispatch_attempts"] += exc.dispatch_attempts
                feedback_payload = dict(exc.feedback_payload or {})
                batch["execution_receipts"].append(
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
                batch["failures"].append(
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

            verdict = self.assurance_evaluator.evaluate(policy=policy, snapshot_id=policy_plan.snapshot_id)
            verdict_payload = verdict.model_dump(mode="json")
            verdict_payload.setdefault("policy_type", policy.get("policy_type"))
            batch["assurance_results"].append(verdict_payload)
            if verdict.status in {"violated", "failed"}:
                batch["failures"].append(
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
        return batch

    def _commit_successful_policies(
        self,
        supi: str,
        policies: list[Dict[str, Any]],
        *,
        session_id: str,
        snapshot_id: str,
    ) -> None:
        existing = self.load_ue_context(supi, snapshot_id=snapshot_id) or {}
        current_ue_context = deepcopy(existing)
        access_policies = [policy for policy in policies if not self._is_mobility_policy(policy)]
        mobility_policies = [policy for policy in policies if self._is_mobility_policy(policy)]
        if access_policies:
            merged = self.compiler.merge_policies_into_context(existing, access_policies)
            catalog = self.load_ue_flow_catalog(supi, snapshot_id=snapshot_id)
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
            current_ue_context.update(
                {
                    "smPolicyData": merged["sm_policy_data"],
                    "pccRules": merged["pcc_rules"],
                    "qosDecs": merged["qos_decs"],
                    "sessRules": merged["sess_rules"],
                    "traffContDecs": merged["traff_cont_decs"],
                    "chgDecs": merged["chg_decs"],
                    "urspRules": merged["ursp_rules"],
                    "app_catalog": catalog.get("app_catalog") or existing.get("app_catalog") or [],
                    "flow_catalog": catalog.get("flow_catalog") or existing.get("flow_catalog") or [],
                }
            )

        for policy in mobility_policies:
            if self.persist_am_policy_association is None or self.record_mobility_event is None:
                raise RuntimeError("AM policy persistence callbacks are not configured")
            policy_details = policy.get("policy_details") or {}
            request = policy_details.get("request")
            if not isinstance(request, dict):
                raise RuntimeError(f"AM policy {policy.get('policy_id')} missing request payload")
            association_id = str(policy.get("policy_id") or "").strip()
            if not association_id:
                raise RuntimeError("AM policy missing top-level policy_id")
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

            am_policy_context = dict(current_ue_context.get("amPolicyContext") or {})
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
            access_mobility_context = current_ue_context.get("accessMobilityContext") or {}
            serving_nf_context = current_ue_context.get("servingNfContext") or {}
            mobility_summary = {
                "currentAssociationId": association_id,
                "currentTriggers": policy_details.get("triggers") or [],
                "currentRfsp": request.get("rfsp"),
                "lastUpdatedReason": "policy_dispatch",
            }
            updated = self.persist_ue_context(
                supi=supi,
                access_mobility_context=access_mobility_context,
                am_policy_context=am_policy_context,
                serving_nf_context=serving_nf_context,
                mobility_summary=mobility_summary,
            )
            if not updated:
                raise RuntimeError(f"failed to update UE mobility context for {association_id}")
            current_ue_context["amPolicyContext"] = am_policy_context
            current_ue_context["mobilitySummary"] = mobility_summary
            current_ue_context["accessMobilityContext"] = access_mobility_context
            current_ue_context["servingNfContext"] = serving_nf_context

            if self.persist_serving_nf_binding is not None and (serving_nf_context.get("amf_id") or serving_nf_context.get("amf_uri")):
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
                event_summary="AM policy association updated from policy dispatch execution",
                event_payload={"association_id": association_id, "request": request, "policy": policy_details},
            )
            if not recorded:
                raise RuntimeError(f"failed to record mobility event for {association_id}")

    @staticmethod
    def _find_app_for_patch(apps: list[Any], *, app_id: str, app_name: str, supi: str) -> Optional[Any]:
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

    def _extract_snapshot_flow_patches(self, strategy_output: Any) -> list[tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]]:
        optimizer_result = getattr(strategy_output, "optimizer_result", {}) or {}
        if not isinstance(optimizer_result, dict):
            return []
        writeback_patch = optimizer_result.get("snapshot_writeback_patch") or {}
        if not isinstance(writeback_patch, dict):
            return []
        qos_plan = writeback_patch.get("qos_plan") or {}
        if not isinstance(qos_plan, dict):
            return []
        target_app_patch = qos_plan.get("target_app") or {}
        impacted_flows = qos_plan.get("impacted_flows") or []
        flow_patches: list[tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]] = []
        malformed_entries: list[str] = []

        if isinstance(target_app_patch, dict):
            target_flow_patches = target_app_patch.get("flows") or []
            if isinstance(target_flow_patches, list):
                for index, item in enumerate(target_flow_patches, start=1):
                    if not isinstance(item, dict):
                        malformed_entries.append(f"target_app.flows[{index}] is not an object")
                        continue
                    flow_patches.append(
                        (
                            str(target_app_patch.get("id") or "").strip() or None,
                            str(target_app_patch.get("name") or "").strip() or None,
                            str(target_app_patch.get("supi") or getattr(strategy_output, "supi", "") or "").strip() or None,
                            item,
                        )
                    )

        if isinstance(impacted_flows, list):
            for index, item in enumerate(impacted_flows, start=1):
                if not isinstance(item, dict):
                    malformed_entries.append(f"impacted_flows[{index}] is not an object")
                    continue
                flow_payload = item.get("flow")
                if not isinstance(flow_payload, dict):
                    malformed_entries.append(f"impacted_flows[{index}].flow is missing or not an object")
                    continue
                flow_patches.append(
                    (
                        str(item.get("app_id") or "").strip() or None,
                        str(item.get("app_name") or "").strip() or None,
                        str(item.get("supi") or "").strip() or None,
                        flow_payload,
                    )
                )

        if malformed_entries:
            raise RuntimeError("malformed snapshot writeback patch: " + "; ".join(malformed_entries))
        return flow_patches

    def _apply_snapshot_flow_patches(
        self,
        *,
        apps: list[Any],
        flow_patches: list[tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]],
    ) -> int:
        binding_errors: list[str] = []
        applied_count = 0
        for app_id, app_name, supi, flow_patch in flow_patches:
            flow_id = str(flow_patch.get("id") or "").strip()
            if not flow_id:
                binding_errors.append("writeback flow patch is missing flow id")
                continue
            app = self._find_app_for_patch(apps, app_id=app_id or "", app_name=app_name or "", supi=supi or "")
            if app is None:
                binding_errors.append(
                    f"writeback patch could not bind app for flow_id={flow_id} app_id={app_id or ''} app_name={app_name or ''} supi={supi or ''}"
                )
                continue
            target_flow = next((flow for flow in app.flows if str(getattr(flow, "id", "") or "").strip() == flow_id), None)
            if target_flow is None:
                binding_errors.append(
                    f"writeback patch could not bind flow_id={flow_id} under app_id={str(getattr(app, 'id', '') or '').strip()}"
                )
                continue
            self._apply_flow_patch(target_flow, flow_patch)
            applied_count += 1
        if binding_errors:
            raise RuntimeError("snapshot writeback binding failed: " + "; ".join(binding_errors))
        return applied_count

    def _commit_snapshot_writeback(self, strategy_output: Any) -> str:
        flow_patches = self._extract_snapshot_flow_patches(strategy_output)
        if not flow_patches:
            return ""
        return str(getattr(strategy_output, "snapshot_id", "") or "").strip()

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
            policy_plan = PolicyPlan(
                session_id=plan.session_id,
                snapshot_id=plan.snapshot_id,
                supi=plan.supi,
                policies=plan.policies,
            )
            policies = [self.guard.validate_policy(policy) for policy in plan.policies]
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

        try:
            batch = self._run_policies(policies=policies, policy_plan=policy_plan)
            execution_receipts = batch["execution_receipts"]
            assurance_results = batch["assurance_results"]
            total_dispatch_attempts = batch["total_dispatch_attempts"]
            failures = batch["failures"]

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
                has_assurance_failure = any(str(item.get("phase") or "").strip() == "assurance" for item in failures)
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
                failure_scope="none",
                feedback_payload={},
                dispatch_attempts=total_dispatch_attempts,
                committed_snapshot_id=committed_snapshot_id,
            )
        except ExecutionDecisionError as exc:
            log_timing(self.logger, "pda_total", datetime.now().timestamp() - total_start, status="error")
            return self._build_failure_outcome(
                detail=str(exc),
                failure_scope="mixed",
                feedback_payload=exc.feedback_payload,
                dispatch_attempts=exc.dispatch_attempts,
            )


__all__ = ["ExecutionOutcome", "ExecutionDecisionError", "ExecutionController"]
