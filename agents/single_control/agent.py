from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from agent_runtime import ArtifactEnvelope
from agent_runtime.messages import extract_tool_calls, extract_tool_results
from agents.BaseAgent import BaseAgent, coerce_structured_response, extract_grounding_tool_names
from agents.intent_encoding.compiler import IntentCompiler
from agents.optimization_strategy.advisor import build_advisor_user_prompt
from agents.optimization_strategy.compiler import OptimizationStrategyCompiler
from agents.optimization_strategy.response_models import OsaAdvisorOutput
from agents.optimization_strategy.tools import _summarize_optimizer_result, build_request_tools
from agents.tools import get_knowledge_by_key, search_semantic_knowledge, think
from agents.tools.pcf_tools import (
    get_am_policy_context,
    get_sm_ue_context,
    get_sm_ue_flow_catalog,
    search_am_policy_targets,
    search_sm_flow_targets,
)
from agents.worker import ArtifactWorkerMixin
from domain.collaboration import PlanningRequest
from domain.control_plane import ControlDomain, GlobalControlIntent, ObjectiveProfile
from domain.policy_plan import OperationIntent, PolicyPlanDraft
from utils.logger import log_event, log_timing

from .contracts import SingleAgentIntentDecision
from .prompts import SINGLE_AGENT_INTENT_PROMPT


class SingleControlAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "single_control"
    INTENT_GROUNDING_TOOLS = {
        "search_sm_flow_targets",
        "get_sm_ue_context",
        "get_sm_ue_flow_catalog",
        "get_am_policy_context",
        "search_am_policy_targets",
        "search_semantic_knowledge",
        "get_knowledge_by_key",
    }
    PLAN_GROUNDING_TOOLS = {
        "preview_qos_optimizer",
        "fetch_qos_network_status",
        "inspect_mobility_ue_policies",
        "search_semantic_knowledge",
        "get_knowledge_by_key",
    }

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        *,
        use_local_model: bool = False,
        rag_enabled: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "single_control"
        self.rag_enabled = rag_enabled
        self.intent_compiler = IntentCompiler()
        self.plan_compiler = OptimizationStrategyCompiler()
        self.initialize_agent_runtime(logger_color="\033[96m")

    def _build_intent_tools(self) -> List[Any]:
        tools: List[Any] = [
            search_sm_flow_targets,
            get_sm_ue_context,
            get_sm_ue_flow_catalog,
            get_am_policy_context,
            search_am_policy_targets,
        ]
        if self.rag_enabled:
            tools.extend([search_semantic_knowledge, get_knowledge_by_key])
        return tools

    def _build_plan_tools(self, planning_request: PlanningRequest) -> List[Any]:
        tools: List[Any] = [think]
        if self.rag_enabled:
            tools.extend([search_semantic_knowledge, get_knowledge_by_key])
        tools.extend(build_request_tools(planning_request))
        return tools

    def analyze_operation_intent(
        self,
        *,
        user_input: str,
        context: str = "",
        session_id: str = "",
        snapshot_id: str = "",
        allow_user_interaction: bool = False,
    ) -> tuple[GlobalControlIntent, OperationIntent]:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "single_control_intent_start")
        supi_match = re.search(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        supi = supi_match.group(1) if supi_match else ""
        evidence = self.intent_compiler.build_intent_evidence(
            user_input=str(user_input or "").strip(),
            supi=supi,
            main_directives={},
            catalog_payload={},
            semantic_candidates=[],
            am_context_payload={},
            am_policy_candidates=[],
        )
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            supi=supi or None,
            thread_id=session_id,
            allow_user_interaction=allow_user_interaction,
        )
        prompt = (
            "User request:\n"
            f"{user_input}\n\n"
            "Current evidence:\n"
            f"{json.dumps(evidence.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            "Retry / coordinator context:\n"
            f"{context or 'N/A'}\n\n"
            "Return one SingleAgentIntentDecision JSON object only."
        )
        advisor_agent = self.create_json_agent(
            tools=self._build_intent_tools(),
            system_prompt=SINGLE_AGENT_INTENT_PROMPT,
            response_model=SingleAgentIntentDecision,
            max_iterations=14,
        )
        result = advisor_agent.invoke({"messages": [{"role": "user", "content": prompt}]}, context=runtime_context)
        decision = coerce_structured_response(
            result,
            SingleAgentIntentDecision,
            error_message="Single control agent returned no intent decision",
        )
        refreshed_evidence = self._refresh_intent_evidence_from_tool_results(
            base_evidence=evidence,
            advisor_result=result,
            decision=decision,
        )
        grounding_tools = extract_grounding_tool_names(result, self.INTENT_GROUNDING_TOOLS)
        validation_errors = self.intent_compiler.validate_intent_grounding(
            evidence=refreshed_evidence,
            grounding_tools=grounding_tools,
        )
        if validation_errors:
            raise RuntimeError("Single agent intent grounding validation failed: " + "; ".join(validation_errors))

        operation_intent = self.intent_compiler.compile_operation_intent(
            evidence=refreshed_evidence,
            advisor_decision=self._to_intent_advisor_decision(decision),
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
            main_directives={
                "requested_domains": decision.requested_domains,
                "domain_evidence": decision.domain_evidence,
                "objective_profile_hint": decision.objective_profile_hint,
            },
        )
        global_intent = self._build_global_intent(
            decision=decision,
            operation_intent=operation_intent,
            session_id=session_id,
            snapshot_id=snapshot_id,
            user_input=user_input,
        )
        log_timing(self.logger, "single_control_intent_total", time.perf_counter() - total_start, status="success")
        return global_intent, operation_intent

    def generate_strategy(self, planning_request: PlanningRequest) -> PolicyPlanDraft:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "single_control_plan_start")
        optimizer_preview = self._run_joint_optimizer_direct(planning_request)
        planning_evidence = self.plan_compiler.build_planning_evidence(planning_request, optimizer_preview)
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=planning_request.context.session_id,
            snapshot_id=planning_request.context.snapshot_id,
            supi=planning_request.operation_intent.supi,
            thread_id=planning_request.context.session_id,
        )
        advisor_agent = self.create_json_agent(
            tools=self._build_plan_tools(planning_request),
            system_prompt=__import__("agents.optimization_strategy.prompts", fromlist=["OSA_SYSTEM_PROMPT"]).OSA_SYSTEM_PROMPT,
            response_model=OsaAdvisorOutput,
            max_iterations=14,
        )
        prompt = build_advisor_user_prompt(
            normalized_user_intent=planning_request.operation_intent.model_dump(mode="json"),
            coordination_context=planning_request.context.model_dump(mode="json"),
            planning_evidence=planning_evidence,
            optimizer_preview_summary=_summarize_optimizer_result(optimizer_preview),
        )
        result = advisor_agent.invoke({"messages": [{"role": "user", "content": prompt}]}, context=runtime_context)
        advisor_output = coerce_structured_response(
            result,
            OsaAdvisorOutput,
            error_message="Single control agent returned no policy decision",
        )
        grounding_tools = extract_grounding_tool_names(result, self.PLAN_GROUNDING_TOOLS)
        validation_errors = self.plan_compiler.validate_advisor_output(
            advisor_output=advisor_output,
            planning_request=planning_request,
            grounding_tools=grounding_tools,
            planning_evidence=planning_evidence,
        )
        if validation_errors:
            raise RuntimeError("Single agent policy grounding validation failed: " + "; ".join(validation_errors))
        plan = self.plan_compiler.assemble_policy_plan(
            advisor_output=advisor_output,
            planning_request=planning_request,
            optimizer_preview=optimizer_preview,
        )
        log_timing(self.logger, "single_control_plan_total", time.perf_counter() - total_start, status="success")
        return plan

    def _refresh_intent_evidence_from_tool_results(
        self,
        *,
        base_evidence: Any,
        advisor_result: Dict[str, Any],
        decision: SingleAgentIntentDecision,
    ) -> Any:
        tool_calls = {
            str(call.get("id") or "").strip(): call
            for call in extract_tool_calls(advisor_result.get("messages") or [])
            if str(call.get("id") or "").strip()
        }
        catalog_payload = dict(base_evidence.cached_catalog or {})
        semantic_candidates = list(base_evidence.cached_semantic_candidates or [])
        am_context_payload = dict(base_evidence.cached_am_context or {})
        am_policy_candidates = list(base_evidence.cached_am_policy_candidates or [])

        for result in extract_tool_results(advisor_result.get("messages") or []):
            tool_name = str(result.get("name") or "").strip()
            call_id = str(result.get("tool_call_id") or "").strip()
            call_args = tool_calls.get(call_id, {}).get("args") if call_id else {}
            if not isinstance(call_args, dict):
                call_args = {}
            if tool_name == "get_sm_ue_flow_catalog":
                payload = self.intent_compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="SM UE Flow Catalog Retrieved:",
                )
                if payload:
                    catalog_payload = dict(payload)
            elif tool_name == "search_sm_flow_targets":
                payload = self.intent_compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="SM Flow Target Search Retrieved:",
                )
                if payload:
                    semantic_candidates = list(payload.get("candidates") or [])
            elif tool_name == "get_am_policy_context":
                payload = self.intent_compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="AM Policy Context Retrieved:",
                )
                if payload:
                    am_context_payload = dict(payload)
            elif tool_name == "search_am_policy_targets":
                payload = self.intent_compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="AM Policy Target Search Retrieved:",
                )
                if payload:
                    am_policy_candidates = list(payload.get("candidates") or [])

        return self.intent_compiler.build_intent_evidence(
            user_input=base_evidence.user_input,
            supi=base_evidence.supi,
            main_directives={
                "requested_domains": decision.requested_domains,
                "domain_evidence": decision.domain_evidence,
                "objective_profile_hint": decision.objective_profile_hint,
            },
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )

    @staticmethod
    def _to_intent_advisor_decision(decision: SingleAgentIntentDecision) -> Any:
        from agents.intent_encoding.contracts import IntentAdvisorDecision

        return IntentAdvisorDecision(
            selected_app_id=decision.selected_app_id,
            selected_flow_id=decision.selected_flow_id,
            operation_type=decision.operation_type,
            raw_intent_summary=decision.raw_intent_summary,
            rationale=decision.rationale,
            mobility_intent=decision.mobility_intent,
            objective_profile_hint=decision.objective_profile_hint,
            flows=decision.flows,
        )

    @staticmethod
    def _required_evidence_from_domains(domains: List[str]) -> List[str]:
        required: List[str] = []
        normalized = {str(item or "").strip().lower() for item in domains if str(item or "").strip()}
        if "qos" in normalized:
            required.append("qos_runtime_evidence")
        if "mobility" in normalized:
            required.append("mobility_policy_context")
        return required

    def _build_global_intent(
        self,
        *,
        decision: SingleAgentIntentDecision,
        operation_intent: OperationIntent,
        session_id: str,
        snapshot_id: str,
        user_input: str,
    ) -> GlobalControlIntent:
        requested_domains = [ControlDomain(item) for item in decision.requested_domains]
        target_flow_ids = [
            str(flow.flow_id or "").strip()
            for flow in operation_intent.flows
            if str(flow.flow_id or "").strip()
        ]
        return GlobalControlIntent(
            session_id=session_id,
            snapshot_id=snapshot_id,
            raw_input=user_input,
            user_goal=str(decision.raw_intent_summary or user_input).strip(),
            operation_type=str(decision.operation_type or "modify").strip() or "modify",
            urgency="Normal",
            supi=str(operation_intent.supi or "").strip(),
            app_id=str(operation_intent.app_id or "").strip(),
            app_name=operation_intent.app_name,
            target_flow_ids=target_flow_ids,
            target_flow_names=[],
            next_agent="optimization_strategy",
            requested_domains=requested_domains,
            domain_evidence=decision.domain_evidence,
            objective_profile=ObjectiveProfile(profile_name=str(decision.objective_profile_hint or "balanced").strip() or "balanced"),
            mobility_triggers=[],
            active_constraints=[],
            required_evidence=self._required_evidence_from_domains(decision.requested_domains),
            forbidden_assumptions=[],
            prompt_injections={
                "optimization_strategy": "Preserve resolved identifiers and generate the minimum executable policy set.",
            },
        )

    @staticmethod
    def _run_joint_optimizer_direct(planning_request: PlanningRequest) -> Any:
        from agents.tools.optimizer import run_joint_control_optimizer as run_optimizer
        from agents.optimization_strategy.request_builder import build_joint_optimizer_request

        request = build_joint_optimizer_request(planning_request)
        return run_optimizer(request)


__all__ = ["SingleControlAgent"]
