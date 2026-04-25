from __future__ import annotations

import time
from typing import Any

from agent_runtime import ArtifactEnvelope
from agents.BaseAgent import BaseAgent, extract_grounding_tool_names
from agents.tools import get_knowledge_by_key, search_semantic_knowledge, think
from agents.tools.optimizer import run_joint_control_optimizer as run_joint_control_service
from agents.worker import ArtifactWorkerMixin
from domain.collaboration import PlanningRequest
from domain.policy_plan import PolicyPlanDraft
from utils.logger import log_event, log_timing

from .advisor import OptimizationStrategyAdvisor
from .compiler import OptimizationStrategyCompiler
from .policy_normalizer import json_friendly as _json_friendly
from .policy_normalizer import normalize_app_id as _normalize_app_id
from .policy_normalizer import normalize_policy_plan_draft
from .request_builder import build_joint_optimizer_request


class OptimizationStrategyAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "optimization_strategy"
    QOS_GROUNDING_TOOLS = {
        "preview_qos_optimizer",
        "fetch_qos_network_status",
    }
    MOBILITY_GROUNDING_TOOLS = {
        "inspect_mobility_ue_policies",
    }
    GROUNDING_TOOLS = {
        *QOS_GROUNDING_TOOLS,
        *MOBILITY_GROUNDING_TOOLS,
        "search_semantic_knowledge",
        "get_knowledge_by_key",
    }
    _BASE_TOOLS = [think, search_semantic_knowledge, get_knowledge_by_key]

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507", use_local_model: bool = False) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "optimization_strategy"
        self.compiler = OptimizationStrategyCompiler()
        self.advisor = OptimizationStrategyAdvisor(self, self.compiler)
        self.initialize_agent_runtime(logger_color="\033[94m")

    def expected_request_type(self) -> str:
        return "PlanningRequest"

    def response_artifact_type(self) -> str:
        return "PolicyPlanDraft"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> PolicyPlanDraft:
        planning_request = PlanningRequest.model_validate(envelope.payload)
        return self.generate_strategy_from_request(
            planning_request=planning_request,
            request_envelope=envelope,
        )

    def _cache_received_request(self, planning_request: PlanningRequest) -> ArtifactEnvelope:
        return self.cache_received_artifact(
            artifact_type="PlanningRequest",
            payload=planning_request,
            session_id=planning_request.context.session_id,
            snapshot_id=planning_request.context.snapshot_id,
        )

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        policy_plan: PolicyPlanDraft,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="PolicyPlanDraft",
            request_envelope=request_envelope,
            payload=policy_plan,
        )

    def generate_strategy(self, planning_request: PlanningRequest) -> PolicyPlanDraft:
        self.ensure_worker_runtime_initialized()
        if not isinstance(planning_request, PlanningRequest):
            raise TypeError("generate_strategy expects a PlanningRequest instance")
        request_envelope = self._cache_received_request(planning_request)
        return self.generate_strategy_from_request(
            planning_request=planning_request,
            request_envelope=request_envelope,
        )

    def generate_strategy_from_request(
        self,
        *,
        planning_request: PlanningRequest,
        request_envelope: ArtifactEnvelope,
    ) -> PolicyPlanDraft:
        operation_intent = planning_request.operation_intent
        normalized_user_intent = _json_friendly(operation_intent.model_dump(mode="json"))
        normalized_user_intent["app_id"] = _normalize_app_id(normalized_user_intent.get("app_id"))
        coordination_context = _json_friendly(planning_request.context.model_dump(mode="json"))

        total_start = time.perf_counter()
        log_event(self.logger, "osa_generate_start")
        advisor_invocation = None
        try:
            optimizer_preview = self._run_joint_optimizer_direct(planning_request)
            planning_evidence = self.compiler.build_planning_evidence(planning_request, optimizer_preview)
            advisor_invocation = self.advisor.advise(
                planning_request=planning_request,
                normalized_user_intent=normalized_user_intent,
                coordination_context=coordination_context,
                optimizer_preview=optimizer_preview,
                planning_evidence=planning_evidence,
            )
            grounding_tools = extract_grounding_tool_names(
                advisor_invocation.raw_result,
                self.GROUNDING_TOOLS,
            )
            validation_errors = self.compiler.validate_advisor_output(
                advisor_output=advisor_invocation.advisor_output,
                planning_request=planning_request,
                grounding_tools=grounding_tools,
                planning_evidence=planning_evidence,
            )
            if validation_errors:
                raise RuntimeError("OSA advisor grounding validation failed: " + "; ".join(validation_errors))

            final_output = self.compiler.assemble_policy_plan(
                advisor_output=advisor_invocation.advisor_output,
                planning_request=planning_request,
                optimizer_preview=optimizer_preview,
            )
            advisor_invocation.write_final_trace(
                status="success",
                compiler_output=final_output.model_dump(mode="json"),
            )
            self._cache_produced_result(
                request_envelope=request_envelope,
                policy_plan=final_output,
            )
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="success")
            return final_output
        except Exception as exc:
            if advisor_invocation is not None:
                advisor_invocation.write_final_trace(
                    status="error",
                    error=str(exc),
                )
            self.logger.error(f"Failed to generate optimization strategy: {exc}")
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    @staticmethod
    def _build_joint_optimizer_request(planning_request: PlanningRequest):
        return build_joint_optimizer_request(planning_request)

    @staticmethod
    def _normalize_policy_plan_draft(raw_output: PolicyPlanDraft, operation_intent):
        return normalize_policy_plan_draft(raw_output, operation_intent)

    def _run_joint_optimizer_direct(self, planning_request: PlanningRequest) -> Any:
        request = self._build_joint_optimizer_request(planning_request)
        return run_joint_control_service(request)


__all__ = ["OptimizationStrategyAgent"]
