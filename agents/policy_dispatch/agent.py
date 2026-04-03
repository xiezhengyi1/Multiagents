from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from agent_runtime import AgentRuntimeContext, ArtifactEnvelope
from domain.policy_compiler import PolicyCompiler
from tools.db_tool import (
    get_latest_snapshot_data,
    get_snapshot_data_by_id,
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    upsert_ue_context,
)
from tools.pcf_tools import dispatch_policy_to_pcf, dispatch_policy_to_pcf_request
from workflows.assurance_evaluator import AssuranceEvaluator
from workflows.execution_controller import ExecutionController

from agents.BaseAgent import BaseAgent
from agents.worker import ArtifactWorkerMixin


@tool
def tool_dispatch_policy(
    policy_type: str,
    policy_json: str,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """Dispatch a policy payload to PCF."""
    return dispatch_policy_to_pcf(policy_type, policy_json)


@tool
def tool_evaluate_sla(
    supi: str,
    flow_id: str,
    k: float = 0.2,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """Evaluate whether a specific flow satisfies its SLA in the latest snapshot."""
    target_supi = str(supi or "").strip()
    target_flow_id = str(flow_id or "").strip()
    if not target_supi or not target_flow_id:
        return "evaluation_failed"

    snapshot = get_latest_snapshot_data() or {}
    app_data = snapshot.get("apps", []) if isinstance(snapshot, dict) else []

    for app in app_data:
        if not isinstance(app, dict):
            continue
        app_supi = str(app.get("supi") or "").strip()
        flows = app.get("flows", [])
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_supi = app_supi or str(flow.get("supi") or "").strip()
            if flow_supi != target_supi or str(flow.get("flow_id") or "").strip() != target_flow_id:
                continue

            lat_req = float(flow.get("lat") or 0.0)
            jitter_req = float(flow.get("jitter_req") or 0.0)
            gbr_ul = float(flow.get("gbr_ul") or 0.0)
            gbr_dl = float(flow.get("gbr_dl") or 0.0)

            sim_latency = float(flow.get("sim_latency") or 0.0)
            sim_jitter = float(flow.get("sim_jitter") or 0.0)
            sim_throughput_ul = float(flow.get("sim_throughput_ul") or 0.0)
            sim_throughput_dl = float(flow.get("sim_throughput_dl") or 0.0)

            k_lat = (sim_latency - lat_req) / (lat_req if lat_req > 0 else 1.0)
            k_jitter = (sim_jitter - jitter_req) / (jitter_req if jitter_req > 0 else 1.0)
            k_ul = (gbr_ul - sim_throughput_ul) / (gbr_ul if gbr_ul > 0 else 1.0)
            k_dl = (gbr_dl - sim_throughput_dl) / (gbr_dl if gbr_dl > 0 else 1.0)

            if all(k_i < 0 for k_i in (k_lat, k_jitter, k_ul, k_dl)):
                return "satisfied"
            if all(k_i <= k for k_i in (k_lat, k_jitter, k_ul, k_dl)):
                return "satisfied"
            return f"{target_flow_id} violated"

    return "evaluation_failed"


@tool
def tool_update_db_after_success(
    supi: str,
    policy: str = "",
    policies_json: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """Commit successfully applied policies into UE context storage."""
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return "database update failed: supi is required"

    raw_policies: List[Dict[str, Any]] = []
    if policies_json:
        try:
            parsed = json.loads(policies_json)
        except json.JSONDecodeError as exc:
            return f"database update failed: policies_json is not valid JSON: {exc}"
        if not isinstance(parsed, list):
            return "database update failed: policies_json must be a list"
        raw_policies = [PolicyCompiler.json_friendly(item) for item in parsed if isinstance(item, dict)]
    elif policy:
        try:
            parsed_policy = json.loads(policy) if isinstance(policy, str) else policy
        except json.JSONDecodeError as exc:
            return f"database update failed: policy is not valid JSON: {exc}"
        if not isinstance(parsed_policy, dict):
            return "database update failed: policy must be an object"
        raw_policies = [{"policy_details": PolicyCompiler.json_friendly(parsed_policy)}]
    else:
        return "database update failed: missing policy or policies_json"

    existing = get_ue_context_by_supi(normalized_supi) or {}
    merged = PolicyCompiler.merge_policies_into_context(existing, raw_policies)
    catalog = get_ue_flow_catalog_by_supi(normalized_supi)

    ok = upsert_ue_context(
        supi=normalized_supi,
        sm_policy_data=merged.get("sm_policy_data"),
        pcc_rules=merged.get("pcc_rules"),
        qos_decs=merged.get("qos_decs"),
        sess_rules=merged.get("sess_rules"),
        traff_cont_decs=merged.get("traff_cont_decs"),
        chg_decs=merged.get("chg_decs"),
        ursp_rules=merged.get("ursp_rules"),
        app_catalog=catalog.get("app_catalog") or existing.get("app_catalog") or [],
        flow_catalog=catalog.get("flow_catalog") or existing.get("flow_catalog") or [],
    )
    return "database update success" if ok else "database update failed"


class FeedbackReport(BaseModel):
    execution_status: str = Field(description="Overall execution status: Success, Partial Success, or Failed")
    performance_metrics: str = Field(description="Summary of execution receipts and SLA results")
    violation_details: str = Field(description="Explicit failure or violation details, or None")
    correction_suggestion: str = Field(description="Actionable remediation guidance")


class PolicyDispatchAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "policy_dispatch"
    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.agent_name = "policy_dispatch"
        self.initialize_agent_runtime(logger_color="\033[92m")
        self.tools = [tool_dispatch_policy, tool_update_db_after_success, tool_evaluate_sla]
        self.tool_map = {tool_obj.name: tool_obj for tool_obj in self.tools}

    def expected_request_type(self) -> str:
        return "PolicyPlanDraft"

    def response_artifact_type(self) -> str:
        return "FeedbackReport"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> FeedbackReport:
        return self.execute_and_evaluate_from_request(
            strategy_output=envelope.payload,
            request_envelope=envelope,
        )

    def _build_execution_controller(self) -> ExecutionController:
        return ExecutionController(
            dispatch_policy=dispatch_policy_to_pcf_request,
            assurance_evaluator=self._build_assurance_evaluator(),
            load_ue_context=get_ue_context_by_supi,
            load_ue_flow_catalog=get_ue_flow_catalog_by_supi,
            persist_ue_context=upsert_ue_context,
            logger=self.logger,
        )

    @staticmethod
    def _build_assurance_evaluator() -> AssuranceEvaluator:
        return AssuranceEvaluator(load_snapshot_by_id=get_snapshot_data_by_id)

    def _cache_received_request(self, strategy_output: Any) -> ArtifactEnvelope:
        payload = (
            strategy_output.model_dump(mode="json")
            if hasattr(strategy_output, "model_dump")
            else PolicyCompiler.json_friendly(strategy_output)
        )
        return self.cache_received_artifact(
            artifact_type="PolicyPlanDraft",
            payload=payload if isinstance(payload, dict) else {"value": payload},
            session_id=str(getattr(strategy_output, "session_id", "") or "").strip(),
            snapshot_id=str(getattr(strategy_output, "snapshot_id", "") or "").strip(),
        )

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        feedback_report: FeedbackReport,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="FeedbackReport",
            request_envelope=request_envelope,
            payload=feedback_report,
        )

    def execute_and_evaluate(self, strategy_output: Any) -> FeedbackReport:
        self.ensure_worker_runtime_initialized()
        request_envelope = self._cache_received_request(strategy_output)
        return self.execute_and_evaluate_from_request(
            strategy_output=strategy_output,
            request_envelope=request_envelope,
        )

    def execute_and_evaluate_from_request(
        self,
        *,
        strategy_output: Any,
        request_envelope: ArtifactEnvelope,
    ) -> FeedbackReport:
        outcome = self._build_execution_controller().execute(strategy_output)
        feedback_report = FeedbackReport(**outcome.to_dict())
        self._cache_produced_result(
            request_envelope=request_envelope,
            feedback_report=feedback_report,
        )
        return feedback_report

    @staticmethod
    def _extract_json_payload_from_dispatch_output(dispatch_output: str) -> Optional[Dict[str, Any]]:
        text = str(dispatch_output or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            json_start = text.find("{")
            if json_start < 0:
                return None
            try:
                payload = json.loads(text[json_start:])
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _extract_feedback_outputs(cls, dispatch_output: str) -> List[Dict[str, Any]]:
        payload = cls._extract_json_payload_from_dispatch_output(dispatch_output)
        if not isinstance(payload, dict):
            return []
        ack = payload.get("ack")
        if isinstance(ack, dict):
            return [ack]
        return []

    @staticmethod
    def _summarize_feedback_outputs(feedback_outputs: List[Dict[str, Any]]) -> str:
        if not feedback_outputs:
            return "No ack feedback received."
        ack = feedback_outputs[-1]
        summary = {
            "request_id": ack.get("request_id"),
            "policy_id": ack.get("policy_id"),
            "expected": ack.get("expected"),
            "received": ack.get("received"),
            "completed": ack.get("completed"),
            "result_count": len(ack.get("results", [])) if isinstance(ack.get("results"), list) else 0,
        }
        return json.dumps(summary, ensure_ascii=False)

    @staticmethod
    def _build_feedback_report_from_ack(feedback_outputs: List[Dict[str, Any]], *, aborted: bool) -> FeedbackReport:
        if not feedback_outputs:
            return FeedbackReport(
                execution_status="Failed",
                performance_metrics="No ack feedback received.",
                violation_details="Missing ack in PDA dispatch response.",
                correction_suggestion="Check PDA downstream ack path and PCF response format.",
            )

        ack = feedback_outputs[-1]
        expected = ack.get("expected")
        received = ack.get("received")
        completed = ack.get("completed")
        results = ack.get("results", []) if isinstance(ack.get("results"), list) else []

        if completed is True:
            execution_status = "Success"
            violation_details = "None"
            correction_suggestion = "None"
        elif isinstance(expected, int) and isinstance(received, int) and received > 0:
            execution_status = "Partial Success"
            violation_details = f"Ack incomplete: expected={expected}, received={received}, completed={completed}"
            correction_suggestion = "Inspect missing ack results before re-planning."
        else:
            execution_status = "Failed"
            violation_details = f"Ack failed: expected={expected}, received={received}, completed={completed}"
            correction_suggestion = "Check policy dispatch pipeline and downstream executor status."

        if aborted and execution_status == "Success":
            execution_status = "Partial Success"
            correction_suggestion = "Execution stopped after an earlier downstream failure."

        return FeedbackReport(
            execution_status=execution_status,
            performance_metrics=json.dumps(
                {
                    "request_id": ack.get("request_id"),
                    "policy_id": ack.get("policy_id"),
                    "expected": expected,
                    "received": received,
                    "completed": completed,
                    "results": results,
                },
                ensure_ascii=False,
            ),
            violation_details=violation_details,
            correction_suggestion=correction_suggestion,
        )

    @staticmethod
    def _extract_flow_id(policy_details: Any) -> Optional[str]:
        return PolicyCompiler.extract_flow_id(policy_details)

    @classmethod
    def merge_policies_into_context(cls, existing_ctx: Dict[str, Any], policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        return PolicyCompiler.merge_policies_into_context(existing_ctx, policies)

    def _evaluate_policy_sla(self, policy: Dict[str, Any]) -> str:
        if policy["target_type"] != "flow":
            return "skipped"

        flow_id = str(policy.get("flow_id") or "").strip()
        if not flow_id:
            raise RuntimeError(f"flow-scoped policy {policy['policy_id']} missing flow_id for SLA evaluation")

        sla_result = tool_evaluate_sla.invoke({"supi": policy["supi"], "flow_id": flow_id})
        normalized = str(sla_result or "").strip().lower()
        if normalized == "satisfied":
            return "satisfied"
        if "violated" in normalized:
            raise RuntimeError(f"policy {policy['policy_id']} SLA violated for flow {flow_id}")
        raise RuntimeError(f"policy {policy['policy_id']} SLA evaluation failed: {sla_result}")
