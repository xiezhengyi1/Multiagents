from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agent_runtime import AgentWorkspace, ArtifactCache, ArtifactEnvelope, ArtifactStore
from agents.worker import ArtifactWorkerMixin

from .contracts import AssuranceDiagnosisRequest, AssuranceDiagnosisResult


class AssuranceDiagnosisAgent(ArtifactWorkerMixin):
    def __init__(self) -> None:
        self.agent_name = "assurance_diagnosis"
        self.init_worker_runtime()

    def expected_request_type(self) -> str:
        return "AssuranceDiagnosisRequest"

    def response_artifact_type(self) -> str:
        return "AssuranceDiagnosisResult"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> AssuranceDiagnosisResult:
        request = AssuranceDiagnosisRequest.model_validate(envelope.payload)
        return self.run(request)

    def run(self, request: AssuranceDiagnosisRequest) -> AssuranceDiagnosisResult:
        execution_feedback = request.execution_feedback or {}
        dispatch_receipts = request.dispatch_receipts or []
        assurance_verdicts = request.assurance_verdicts or []
        telemetry_snapshot = request.telemetry_snapshot or {}

        affected_policy_ids: List[str] = []
        affected_flow_ids: List[str] = []

        for receipt in dispatch_receipts:
            policy_id = str(receipt.get("policy_id") or "").strip()
            if policy_id:
                affected_policy_ids.append(policy_id)

        for verdict in assurance_verdicts:
            policy_id = str(verdict.get("policy_id") or "").strip()
            flow_id = str(verdict.get("flow_id") or "").strip()
            if policy_id:
                affected_policy_ids.append(policy_id)
            if flow_id:
                affected_flow_ids.append(flow_id)

        if not execution_feedback and not dispatch_receipts and not assurance_verdicts and not telemetry_snapshot:
            return AssuranceDiagnosisResult(
                status="insufficient_evidence",
                root_cause_category="missing_evidence",
                root_cause="No execution feedback, dispatch receipts, assurance verdicts, or telemetry snapshot were provided.",
                reason_summary="Diagnosis skipped because no evidence was provided.",
                recommended_actions=["Provide execution feedback or assurance verdict artifacts before retrying diagnosis."],
            )

        execution_status = str(execution_feedback.get("execution_status") or "").strip().lower()
        failed_receipts = [receipt for receipt in dispatch_receipts if str(receipt.get("status") or "").strip().lower() != "success"]
        violated_verdicts = [
            verdict for verdict in assurance_verdicts
            if str(verdict.get("status") or "").strip().lower() == "violated"
        ]
        failed_verdicts = [
            verdict for verdict in assurance_verdicts
            if str(verdict.get("status") or "").strip().lower() == "failed"
        ]

        if failed_receipts or execution_status == "failed":
            first_failure = failed_receipts[0] if failed_receipts else execution_feedback
            detail = str(first_failure.get("error") or first_failure.get("violation_details") or "dispatch or execution failed").strip()
            return AssuranceDiagnosisResult(
                status="diagnosed",
                root_cause_category="execution_failure",
                root_cause=detail,
                affected_policy_ids=sorted(set(affected_policy_ids)),
                affected_flow_ids=sorted(set(affected_flow_ids)),
                recommended_actions=[
                    "Inspect dispatch receipts and downstream ack details for the failed policy path.",
                    "Verify the target policy payload before re-running assurance.",
                ],
                reason_summary="Execution-layer evidence indicates the failure happened before or during policy application.",
            )

        if violated_verdicts:
            top_verdict = violated_verdicts[0]
            detail = str(top_verdict.get("reason") or "SLA violation detected after policy execution.").strip()
            return AssuranceDiagnosisResult(
                status="diagnosed",
                root_cause_category="sla_violation",
                root_cause=detail,
                affected_policy_ids=sorted(set(affected_policy_ids)),
                affected_flow_ids=sorted(set(affected_flow_ids)),
                recommended_actions=[
                    "Inspect post-change telemetry for the affected flow before issuing another policy change.",
                    "Compare requested SLA targets against observed throughput, latency, jitter, and loss.",
                ],
                reason_summary="Assurance verdicts show a post-execution SLA violation.",
            )

        if failed_verdicts:
            top_verdict = failed_verdicts[0]
            detail = str(top_verdict.get("reason") or "Assurance evaluation failed.").strip()
            return AssuranceDiagnosisResult(
                status="diagnosed",
                root_cause_category="assurance_evaluation_failure",
                root_cause=detail,
                affected_policy_ids=sorted(set(affected_policy_ids)),
                affected_flow_ids=sorted(set(affected_flow_ids)),
                recommended_actions=[
                    "Verify telemetry availability and snapshot binding for the affected flow.",
                    "Check assurance evaluator inputs before re-running diagnosis.",
                ],
                reason_summary="Assurance evidence exists, but evaluation could not complete successfully.",
            )

        return AssuranceDiagnosisResult(
            status="insufficient_evidence",
            root_cause_category="inconclusive",
            root_cause="The provided evidence does not show a failed execution or violated assurance verdict.",
            affected_policy_ids=sorted(set(affected_policy_ids)),
            affected_flow_ids=sorted(set(affected_flow_ids)),
            recommended_actions=["Provide richer telemetry or failure artifacts for a conclusive diagnosis."],
            reason_summary="The available artifacts are insufficient for a conclusive diagnosis.",
        )

    def run_from_artifact(self, request_path: Path) -> Path:
        return self.consume_request_artifact(request_path)
