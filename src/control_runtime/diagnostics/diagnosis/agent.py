from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from shared.runtime import ArtifactEnvelope
from shared.runtime import ArtifactWorkerMixin

from .contracts import AssuranceDiagnosisRequest, AssuranceDiagnosisResult


class AssuranceDiagnosisTool(ArtifactWorkerMixin):
    AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

    def __init__(self) -> None:
        self.agent_name = "assurance_diagnosis"
        self.init_worker_runtime()

    def handle_artifact(self, envelope: ArtifactEnvelope) -> AssuranceDiagnosisResult:
        request = AssuranceDiagnosisRequest.model_validate(envelope.payload)
        return self.run(request)

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _collect_affected_ids(
        self,
        *,
        dispatch_receipts: List[Dict[str, Any]],
        assurance_verdicts: List[Dict[str, Any]],
        pda_feedback: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        policy_ids: List[str] = []
        flow_ids: List[str] = []

        for item in dispatch_receipts:
            policy_ids.append(str(item.get("policy_id") or "").strip())
            flow_ids.append(str(item.get("flow_id") or "").strip())
        for item in assurance_verdicts:
            policy_ids.append(str(item.get("policy_id") or "").strip())
            flow_ids.append(str(item.get("flow_id") or "").strip())

        feedback_payload = self._as_dict(pda_feedback.get("feedback_payload"))
        controller_feedback = self._as_dict(feedback_payload.get("controller_feedback"))
        failures = controller_feedback.get("failures")
        if isinstance(failures, list):
            for item in failures:
                if not isinstance(item, dict):
                    continue
                policy_ids.append(str(item.get("policy_id") or "").strip())
                flow_ids.append(str(item.get("flow_id") or "").strip())

        return self._dedupe(policy_ids), self._dedupe(flow_ids)

    def _build_result(
        self,
        *,
        status: str,
        root_cause_category: str,
        root_cause: str,
        reason_summary: str,
        recommended_actions: Iterable[str],
        affected_policy_ids: Iterable[str] = (),
        affected_flow_ids: Iterable[str] = (),
    ) -> AssuranceDiagnosisResult:
        return AssuranceDiagnosisResult(
            status=status,
            root_cause_category=root_cause_category,
            root_cause=str(root_cause or "").strip(),
            reason_summary=str(reason_summary or "").strip(),
            affected_policy_ids=self._dedupe(affected_policy_ids),
            affected_flow_ids=self._dedupe(affected_flow_ids),
            recommended_actions=self._dedupe(recommended_actions),
        )

    def _diagnose_conflict_block(
        self,
        *,
        conflict_result: Dict[str, Any],
        affected_policy_ids: List[str],
        affected_flow_ids: List[str],
    ) -> Optional[AssuranceDiagnosisResult]:
        mediator_status = str(conflict_result.get("mediator_status") or conflict_result.get("status") or "").strip().lower()
        if mediator_status not in {"revise", "reject", "unresolved"}:
            return None
        revision_requests = conflict_result.get("revision_requests")
        extra_actions: List[str] = []
        if isinstance(revision_requests, list):
            for item in revision_requests:
                if not isinstance(item, dict):
                    continue
                reason = str(item.get("reason") or "").strip()
                if reason:
                    extra_actions.append(reason)
        return self._build_result(
            status="diagnosed",
            root_cause_category="cross_domain_inconsistency",
            root_cause=str(conflict_result.get("reason_summary") or "Execution was blocked by conflict resolution.").strip(),
            reason_summary="Conflict resolution blocked execution before policy dispatch.",
            recommended_actions=[
                "Revise the policy plan using the mediator revision requests before retrying execution.",
                "Re-check cross-domain consistency for S-NSSAI, UE-AMBR, and service-area constraints.",
                *extra_actions,
            ],
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )

    @staticmethod
    def _first_by_status(
        entries: List[Dict[str, Any]],
        *,
        field: str,
        accepted: set[str],
        reject: set[str] | None = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_accept = {str(item).strip().lower() for item in accepted if str(item).strip()}
        normalized_reject = {str(item).strip().lower() for item in (reject or set()) if str(item).strip()}
        for entry in entries:
            status = str(entry.get(field) or "").strip().lower()
            if normalized_accept and status in normalized_accept:
                return entry
            if normalized_reject and status not in normalized_reject:
                return entry
        return None

    def _diagnose_execution_failure(
        self,
        *,
        failure: Dict[str, Any],
        affected_policy_ids: List[str],
        affected_flow_ids: List[str],
        source: str,
    ) -> AssuranceDiagnosisResult:
        policy_type = str(failure.get("policy_type") or "").strip()
        phase = str(failure.get("phase") or "").strip().lower()
        detail = str(
            failure.get("error")
            or failure.get("detail")
            or failure.get("violation_details")
            or "Policy dispatch failed."
        ).strip()
        is_mobility = policy_type == self.AM_POLICY_TYPE
        category = "am_policy_dispatch_failure" if is_mobility else "execution_failure"
        summary = "Mobility policy execution failed before assurance could complete." if is_mobility else (
            "Execution-layer evidence shows the policy failed before or during application."
            if source == "dispatch"
            else "Execution feedback indicates the control loop failed before a valid success verdict was produced."
        )
        actions = [
            "Inspect the failed dispatch receipt and the upstream gateway response before retrying execution.",
            "Re-check the compiled policy payload against the live UE/app/flow binding.",
        ]
        if phase == "assurance":
            actions[0] = "Inspect the assurance input snapshot and execution receipts before retrying."
        return self._build_result(
            status="diagnosed",
            root_cause_category=category,
            root_cause=detail,
            reason_summary=summary,
            recommended_actions=actions,
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )

    def _diagnose_assurance_result(
        self,
        *,
        verdict: Dict[str, Any],
        affected_policy_ids: List[str],
        affected_flow_ids: List[str],
    ) -> AssuranceDiagnosisResult:
        verdict_status = str(verdict.get("status") or "").strip().lower()
        detail = str(verdict.get("reason") or "SLA violation detected after policy execution.").strip()
        if verdict_status == "violated":
            return self._build_result(
                status="diagnosed",
                root_cause_category="sla_violation",
                root_cause=detail,
                reason_summary="Post-dispatch assurance verdicts show the requested SLA was not achieved.",
                recommended_actions=[
                    "Compare requested SLA targets against observed latency, jitter, throughput, and loss before retrying.",
                    "Revise the QoS plan or target slice selection instead of re-sending the same policy.",
                ],
                affected_policy_ids=affected_policy_ids,
                affected_flow_ids=affected_flow_ids,
            )
        policy_type = str(verdict.get("policy_type") or "").strip()
        return self._build_result(
            status="diagnosed",
            root_cause_category="mobility_policy_validation_failure" if policy_type == self.AM_POLICY_TYPE else "assurance_evaluation_failure",
            root_cause=detail,
            reason_summary="Execution completed, but the assurance stage could not produce a valid success verdict.",
            recommended_actions=[
                "Verify the snapshot binding and telemetry inputs used by the assurance evaluator.",
                "Check execution receipts and assurance prerequisites before retrying diagnosis.",
            ],
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )

    def _diagnose_pda_feedback(
        self,
        *,
        pda_feedback: Dict[str, Any],
        affected_policy_ids: List[str],
        affected_flow_ids: List[str],
    ) -> Optional[AssuranceDiagnosisResult]:
        execution_status = str(pda_feedback.get("execution_status") or "").strip().lower()
        if execution_status not in {"failed", "partial success"}:
            return None

        feedback_payload = self._as_dict(pda_feedback.get("feedback_payload"))
        controller_feedback = self._as_dict(feedback_payload.get("controller_feedback"))
        failures = controller_feedback.get("failures")
        failure = next((item for item in failures if isinstance(item, dict)), {}) if isinstance(failures, list) else {}
        failure = failure if isinstance(failure, dict) else {}
        phase = str(failure.get("phase") or controller_feedback.get("phase") or feedback_payload.get("phase") or "").strip().lower()
        detail = str(
            failure.get("error")
            or controller_feedback.get("error")
            or feedback_payload.get("reason")
            or pda_feedback.get("violation_details")
            or "Execution controller reported a failed round."
        ).strip()
        policy_type = str(failure.get("policy_type") or "").strip()

        if phase == "compile":
            summary = "Policy compilation failed before dispatch."
            actions = [
                "Re-check the policy plan fields and compiler assumptions before retrying.",
                "Revise the optimization output instead of reusing the same compiled payload.",
            ]
        elif phase == "dispatch":
            summary = "Execution controller reported a dispatch-stage failure."
            actions = [
                "Inspect gateway dispatch errors and target bindings before retrying execution.",
                "Re-check the compiled policy payload against the current snapshot.",
            ]
        elif phase == "assurance":
            summary = "Execution controller reported an assurance-stage failure after dispatch."
            actions = [
                "Verify the assurance snapshot and telemetry inputs for the affected flow.",
                "Revise the policy plan if the applied state no longer matches the requested SLA.",
            ]
        elif phase == "runtime":
            summary = "Execution controller hit a runtime failure after dispatch."
            actions = [
                "Inspect execution-controller runtime errors and snapshot writeback inputs before retrying.",
                "Re-check live bindings for SUPI, app_id, and flow_id.",
            ]
        else:
            summary = "Execution controller reported a failed round."
            actions = [
                "Inspect execution-controller feedback payloads before retrying.",
                "Re-check policy bindings and execution-stage evidence.",
            ]

        category = "am_policy_dispatch_failure" if policy_type == self.AM_POLICY_TYPE and phase == "dispatch" else "execution_failure"
        return self._build_result(
            status="diagnosed",
            root_cause_category=category,
            root_cause=detail,
            reason_summary=summary,
            recommended_actions=actions,
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )

    def run(self, request: AssuranceDiagnosisRequest) -> AssuranceDiagnosisResult:
        execution_feedback = self._as_dict(request.execution_feedback)
        dispatch_receipts = [item for item in (request.dispatch_receipts or []) if isinstance(item, dict)]
        assurance_verdicts = [item for item in (request.assurance_verdicts or []) if isinstance(item, dict)]
        telemetry_snapshot = self._as_dict(request.telemetry_snapshot)
        upstream_context = self._as_dict(request.upstream_context)

        pda_feedback = self._as_dict(execution_feedback.get("pda"))
        qos_feedback = self._as_dict(execution_feedback.get("qos"))
        mobility_feedback = self._as_dict(execution_feedback.get("mobility"))
        affected_policy_ids, affected_flow_ids = self._collect_affected_ids(
            dispatch_receipts=dispatch_receipts,
            assurance_verdicts=assurance_verdicts,
            pda_feedback=pda_feedback,
        )

        conflict_result = self._as_dict(upstream_context.get("conflict_result"))
        diagnosis = self._diagnose_conflict_block(
            conflict_result=conflict_result,
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )
        if diagnosis is not None:
            return diagnosis

        if not execution_feedback and not dispatch_receipts and not assurance_verdicts and not telemetry_snapshot:
            return self._build_result(
                status="insufficient_evidence",
                root_cause_category="missing_evidence",
                root_cause="No execution feedback, dispatch receipts, assurance verdicts, or telemetry snapshot were provided.",
                reason_summary="Diagnosis skipped because no execution evidence was available.",
                recommended_actions=[
                    "Provide execution feedback or assurance artifacts before retrying diagnosis.",
                ],
            )

        failed_receipt = self._first_by_status(
            dispatch_receipts,
            field="status",
            accepted=set(),
            reject={"success"},
        )
        if failed_receipt is not None:
            return self._diagnose_execution_failure(
                failure=failed_receipt,
                affected_policy_ids=affected_policy_ids,
                affected_flow_ids=affected_flow_ids,
                source="dispatch",
            )

        assurance_issue = self._first_by_status(
            assurance_verdicts,
            field="status",
            accepted={"violated", "failed"},
        )
        if assurance_issue is not None:
            return self._diagnose_assurance_result(
                verdict=assurance_issue,
                affected_policy_ids=affected_policy_ids,
                affected_flow_ids=affected_flow_ids,
            )

        diagnosis = self._diagnose_pda_feedback(
            pda_feedback=pda_feedback,
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )
        if diagnosis is not None:
            return diagnosis

        mobility_status = str(mobility_feedback.get("status") or "").strip().lower()
        if mobility_status == "failed":
            return self._diagnose_execution_failure(
                failure={
                    "policy_type": self.AM_POLICY_TYPE,
                    "phase": "dispatch",
                    "error": str(mobility_feedback.get("error") or "Mobility execution feedback reported failure.").strip(),
                },
                affected_policy_ids=affected_policy_ids,
                affected_flow_ids=affected_flow_ids,
                source="mobility_feedback",
            )

        qos_status = str(qos_feedback.get("execution_status") or "").strip().lower()
        if qos_status == "failed":
            return self._diagnose_execution_failure(
                failure={
                    "policy_type": "",
                    "phase": "dispatch",
                    "error": str(qos_feedback.get("violation_details") or "QoS execution feedback reported failure.").strip(),
                },
                affected_policy_ids=affected_policy_ids,
                affected_flow_ids=affected_flow_ids,
                source="qos_feedback",
            )

        return self._build_result(
            status="insufficient_evidence",
            root_cause_category="inconclusive",
            root_cause="The provided artifacts do not show a blocking conflict, failed dispatch, or violated assurance verdict.",
            reason_summary="The available artifacts are not sufficient for a conclusive root-cause diagnosis.",
            recommended_actions=[
                "Provide richer execution receipts, assurance verdicts, or telemetry artifacts for diagnosis.",
            ],
            affected_policy_ids=affected_policy_ids,
            affected_flow_ids=affected_flow_ids,
        )
