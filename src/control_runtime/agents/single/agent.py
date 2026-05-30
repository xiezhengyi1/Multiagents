from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from shared.agents import BaseAgent, coerce_structured_response, extract_grounding_tool_names
from shared.runtime import ArtifactWorkerMixin, ContextPolicy, ToolLoopExecutionError
from shared.logging import log_event, log_timing

from ...domain.collaboration import PlanningContext, PlanningRequest
from ...domain.control_plane import ControlDomain, GlobalControlIntent, MainRoundStrategy, ObjectiveProfile
from ...domain.policy_plan import OperationIntent, PlanningRationale, PolicyDraft, PolicyPlanDraft
from ...integrations.storage import get_latest_snapshot_metadata
from ..common import project_intent_evidence_for_prompt, validate_and_compile_intent
from ..grounding.compiler import IntentCompiler
from ..grounding.contracts import IntentAdvisorDecision
from ..planning.policy_normalizer import normalize_policy_plan_draft
from ..grounding.tool_result_adapter import extract_grounding_tool_payloads
from ..planning.compiler import OptimizationStrategyCompiler
from ..planning.tool_result_adapter import extract_planning_tool_evidence
from .contracts import SingleAgentRoundDecision
from .prompts import SINGLE_AGENT_ROUND_PROMPT
from .tools import build_single_agent_tools


class _SingleAgentChatDeepSeekMixin:
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # DeepSeek thinking mode requires replaying the assistant reasoning_content
        # verbatim on subsequent turns. langchain_deepseek stores it on the
        # AIMessage, but the inherited OpenAI payload conversion drops it.
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return payload
        for source_message, payload_message in zip(messages, payload_messages):
            if not isinstance(source_message, AIMessage):
                continue
            if payload_message.get("role") != "assistant":
                continue
            reasoning_content = source_message.additional_kwargs.get("reasoning_content")
            if reasoning_content is not None:
                payload_message["reasoning_content"] = reasoning_content
        return payload


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
    MOBILITY_HINT_TOKENS = (
        "am policy",
        "allowed nssai",
        "target nssai",
        "rfsp",
        "service area",
        "access type",
        "registration",
        "handover",
        "mobility state",
        "移动性",
        "切换",
        "漫游",
        "注册",
    )
    QOS_HINT_TOKENS = (
        "slice",
        "sm policy",
        "throughput",
        "bandwidth",
        "latency",
        "delay",
        "jitter",
        "packet loss",
        "gbr",
        "5qi",
        "qos",
        "切片",
        "时延",
        "延迟",
        "吞吐",
        "带宽",
        "抖动",
        "丢包",
    )
    KNOWLEDGE_HINT_TOKENS = (
        "3gpp",
        "29.5",
        "29.507",
        "29.512",
        "npcf_",
        "smpolicydecision",
        "policyassociationupdaterequest",
        "ursp",
        "requesttrigger",
    )

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        *,
        use_local_model: bool = False,
        rag_enabled: bool = True,
    ) -> None:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
        if not use_local_model:
            if not deepseek_api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is required for SingleControlAgent")
            if not deepseek_base_url:
                raise RuntimeError("DEEPSEEK_BASE_URL is required for SingleControlAgent")
        super().__init__(
            model_name=model_name,
            use_local_model=use_local_model,
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
        )
        if not use_local_model:
            from langchain_deepseek import ChatDeepSeek
            class _SingleAgentChatDeepSeek(_SingleAgentChatDeepSeekMixin, ChatDeepSeek):
                pass
            raw_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "120")
            raw_max_retries = os.getenv("OPENAI_MAX_RETRIES", "2")
            self.llm = _SingleAgentChatDeepSeek(
                model=model_name,
                temperature=0,
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                timeout=float(raw_timeout),
                max_retries=int(raw_max_retries),
            )
        self.agent_name = "single_control"
        self.rag_enabled = rag_enabled
        self.intent_compiler = IntentCompiler()
        self.plan_compiler = OptimizationStrategyCompiler()
        self.initialize_agent_runtime(logger_color="\033[96m")

    def plan_round(
        self,
        *,
        user_input: str,
        context: str = "",
        session_id: str = "",
        snapshot_id: str = "",
        round_index: int = 1,
        feedback_context: str = "",
        round_traces: Optional[List[Dict[str, Any]]] = None,
        previous_mediator_decision: Optional[Dict[str, Any]] = None,
        allow_user_interaction: bool = False,
    ) -> tuple[GlobalControlIntent, OperationIntent, PolicyPlanDraft]:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "single_control_round_plan_start", session_id=session_id, round_index=round_index)

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
        token_budget, token_counter = self._resolve_token_context()
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            supi=supi or None,
            thread_id=session_id,
            allow_user_interaction=allow_user_interaction,
            token_budget=token_budget,
            token_counter=token_counter,
        )
        requested_domains = self._hint_requested_domains(user_input)
        allow_knowledge_tools = self._should_allow_knowledge_tools(user_input)
        advisor_agent = self.create_json_agent(
            tools=build_single_agent_tools(
                rag_enabled=self.rag_enabled,
                requested_domains=requested_domains,
                allow_knowledge_tools=allow_knowledge_tools,
            ),
            system_prompt=SINGLE_AGENT_ROUND_PROMPT,
            response_model=SingleAgentRoundDecision,
            max_iterations=12,
            tool_error_mode="raise",
            max_calls_per_tool=2,
            forbid_duplicate_tool_calls=True,
            tool_result_limits={
                "preview_qos_optimizer": 8000,
                "get_sm_ue_context": 4000,
                "get_sm_ue_flow_catalog": 8000,
                "get_am_policy_context": 4000,
                "fetch_qos_network_status": 4000,
                "inspect_mobility_ue_policies": 4000,
                "search_semantic_knowledge": 4000,
                "get_knowledge_by_key": 4000,
            },
            context_policy=ContextPolicy(
                default_tool_result_chars=4000,
                default_tool_result_tokens=1000,
                tool_result_char_limits={
                    "preview_qos_optimizer": 8000,
                    "get_sm_ue_context": 4000,
                    "get_sm_ue_flow_catalog": 8000,
                    "get_am_policy_context": 4000,
                    "fetch_qos_network_status": 4000,
                    "inspect_mobility_ue_policies": 4000,
                    "search_semantic_knowledge": 4000,
                    "get_knowledge_by_key": 4000,
                },
                recent_tool_results_per_tool=1,
                tool_history_keep_limits={
                    "search_sm_flow_targets": 2,
                    "search_am_policy_targets": 2,
                },
            ),
        )
        prompt = self._build_round_prompt(
            user_input=user_input,
            evidence=evidence.model_dump(mode="json"),
            context=context,
            feedback_context=feedback_context,
        )
        result = self._invoke_round_advisor(
            advisor_agent=advisor_agent,
            prompt=prompt,
            runtime_context=runtime_context,
        )
        decision = coerce_structured_response(
            result,
            SingleAgentRoundDecision,
            error_message="Single control agent returned no unified round decision",
        )

        refreshed_evidence = self._refresh_intent_evidence_from_tool_results(
            base_evidence=evidence,
            advisor_result=result,
            decision=decision,
        )
        intent_grounding_tools = extract_grounding_tool_names(result, self.INTENT_GROUNDING_TOOLS)
        intent_decision = self._to_intent_advisor_decision(decision)
        advisor_validation_errors, grounding_validation_errors, operation_intent = validate_and_compile_intent(
            compiler=self.intent_compiler,
            evidence=refreshed_evidence,
            decision=intent_decision,
            grounding_tools=intent_grounding_tools,
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
            main_directives=self._main_directives_from_decision(decision),
        )
        if advisor_validation_errors or grounding_validation_errors:
            problems = advisor_validation_errors + grounding_validation_errors
            raise RuntimeError("Single agent unified intent validation failed: " + "; ".join(problems))
        if operation_intent is None:
            raise RuntimeError("Single agent unified intent validation produced no OperationIntent")
        global_intent = self._build_global_intent(
            decision=decision,
            operation_intent=operation_intent,
            session_id=session_id,
            snapshot_id=snapshot_id,
            user_input=user_input,
        )
        planning_request = self._build_planning_request(
            global_intent=global_intent,
            operation_intent=operation_intent,
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_index=round_index,
            feedback_context=feedback_context,
            round_traces=round_traces or [],
            previous_mediator_decision=previous_mediator_decision or {},
        )
        planning_tool_evidence = extract_planning_tool_evidence(advisor_result=result)
        policy_plan = self._assemble_policy_plan_from_decision(
            decision=decision,
            planning_request=planning_request,
            planning_tool_evidence=planning_tool_evidence,
        )
        log_timing(self.logger, "single_control_round_plan_total", time.perf_counter() - total_start, status="success")
        return global_intent, operation_intent, policy_plan

    def _refresh_intent_evidence_from_tool_results(
        self,
        *,
        base_evidence: Any,
        advisor_result: Dict[str, Any],
        decision: SingleAgentRoundDecision,
    ) -> Any:
        catalog_payload = dict(base_evidence.cached_catalog or {})
        semantic_candidates = list(base_evidence.cached_semantic_candidates or [])
        am_context_payload = dict(base_evidence.cached_am_context or {})
        am_policy_candidates = list(base_evidence.cached_am_policy_candidates or [])

        for entry in extract_grounding_tool_payloads(advisor_result=advisor_result, compiler=self.intent_compiler):
            tool_name = str(entry.get("tool_name") or "").strip()
            payload = dict(entry.get("payload") or {})
            if tool_name == "get_sm_ue_flow_catalog":
                catalog_payload = dict(payload)
            elif tool_name == "search_sm_flow_targets":
                semantic_candidates = list(payload.get("candidates") or [])
            elif tool_name == "get_am_policy_context":
                am_context_payload = dict(payload)
            elif tool_name == "search_am_policy_targets":
                am_policy_candidates = list(payload.get("candidates") or [])

        resolved_supi = str(decision.supi or base_evidence.supi or "").strip()
        if not resolved_supi:
            raise ValueError("Single control round decision must provide a grounded supi")

        return self.intent_compiler.build_intent_evidence(
            user_input=base_evidence.user_input,
            supi=resolved_supi,
            main_directives=self._main_directives_from_decision(decision),
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )

    @staticmethod
    def _build_round_prompt(*, user_input: str, evidence: Dict[str, Any], context: str, feedback_context: str) -> str:
        feedback_block = ""
        if feedback_context and feedback_context not in context:
            feedback_block = (
                "Feedback context:\n"
                f"{feedback_context}\n\n"
            )
        return (
            "User request:\n"
            f"{user_input}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(project_intent_evidence_for_prompt(evidence), ensure_ascii=False)}\n\n"
            "Round context:\n"
            f"{context or 'N/A'}\n\n"
            f"{feedback_block}"
            "Task:\n"
            "- Stage 1: infer requested_domains and ground the required identifiers.\n"
            "- Stage 2: use planning tools in the same loop and return executable policy payloads.\n"
            "- If qos is active, flows and sm_policies must refer to the same grounded flow_ids.\n"
            "- If mobility is active, am_policy must be backed by grounded UE mobility evidence.\n"
            "- Return one SingleAgentRoundDecision JSON object only.\n"
        )

    @classmethod
    def _hint_requested_domains(cls, user_input: str) -> Optional[List[str]]:
        lowered = str(user_input or "").lower()
        has_mobility = any(token in lowered for token in cls.MOBILITY_HINT_TOKENS)
        has_qos = any(token in lowered for token in cls.QOS_HINT_TOKENS)
        if has_qos and has_mobility:
            return ["qos", "mobility"]
        if has_mobility:
            return ["mobility"]
        if has_qos:
            return ["qos"]
        return None

    @classmethod
    def _should_allow_knowledge_tools(cls, user_input: str) -> bool:
        lowered = str(user_input or "").lower()
        return any(token in lowered for token in cls.KNOWLEDGE_HINT_TOKENS)

    @staticmethod
    def _invoke_round_advisor(*, advisor_agent: Any, prompt: str, runtime_context: Any) -> Dict[str, Any]:
        try:
            return advisor_agent.invoke({"messages": [{"role": "user", "content": prompt}]}, context=runtime_context)
        except Exception as exc:
            if isinstance(exc, ToolLoopExecutionError):
                message = str(exc)
                if "max iterations" in message.lower():
                    raise RuntimeError(f"Single control round advisor did not converge: {message}") from exc
                if "SingleAgentRoundDecision" in message:
                    raise RuntimeError(
                        "Single control round advisor returned an invalid unified round decision. "
                        f"Validator error: {message}"
                    ) from exc
            raise RuntimeError(f"Single control round advisor invocation failed: {exc}") from exc

    @staticmethod
    def _validate_policy_presence(*, decision: SingleAgentRoundDecision, planning_request: PlanningRequest) -> None:
        requested_domains = {
            str(domain or "").strip().lower()
            for domain in (planning_request.operation_intent.requested_domains or [])
            if str(domain or "").strip()
        }
        has_sm = bool(decision.sm_policies)
        has_am = decision.am_policy is not None
        has_ursp = bool(decision.ursp_policies)

        if not has_sm and not has_am and not has_ursp:
            raise RuntimeError("Single control round advisor returned empty policy output")
        if "qos" in requested_domains and not has_sm:
            raise RuntimeError("Single control round advisor omitted sm_policies for a qos request")
        if "qos" not in requested_domains and has_sm:
            raise RuntimeError("Single control round advisor returned sm_policies for a non-qos request")
        if "mobility" in requested_domains and not has_am:
            raise RuntimeError("Single control round advisor omitted am_policy for a mobility request")
        if "mobility" not in requested_domains and has_am:
            raise RuntimeError("Single control round advisor returned am_policy for a non-mobility request")

    def _assemble_policy_plan_from_decision(
        self,
        *,
        decision: SingleAgentRoundDecision,
        planning_request: PlanningRequest,
        planning_tool_evidence: Dict[str, Any],
    ) -> PolicyPlanDraft:
        self._validate_policy_presence(decision=decision, planning_request=planning_request)
        optimizer_preview = self.plan_compiler._latest_optimizer_preview(planning_tool_evidence)
        mobility_context = self.plan_compiler._latest_mobility_context(planning_tool_evidence)

        active_domains = {
            str(item).strip().lower()
            for item in (planning_request.context.active_domains or [])
            if str(item).strip()
        }
        if ControlDomain.QOS.value in active_domains:
            self.plan_compiler._validate_optimizer_preview(optimizer_preview)
        if ControlDomain.MOBILITY.value in active_domains and decision.am_policy is not None and not mobility_context:
            raise ValueError("am_policy compilation requires grounded mobility context")

        planning_metadata = {
            "planning_mode": "single_agent_direct_pcf",
            "requested_domains": list(planning_request.context.active_domains or []),
            "main_retry_scope": str(planning_request.context.main_retry_scope or "").strip(),
            "objective_breakdown": dict(optimizer_preview.get("objective_breakdown") or {}) if isinstance(optimizer_preview, dict) else {},
            "advisor_rationale": decision.rationale,
            "advisor_metadata": decision.planning_metadata,
            "revision_requests": planning_request.context.revision_requests or [],
            "unified_constraints": planning_request.context.unified_constraints or {},
            "optimizer_cross_domain_verdicts": [
                item
                for item in ((optimizer_preview.get("cross_domain_verdicts") if isinstance(optimizer_preview, dict) else []) or [])
            ],
            "snapshot_writeback_patch": self.plan_compiler._build_snapshot_writeback_patch(optimizer_preview),
        }

        plan = PolicyPlanDraft(
            supi=str(planning_request.operation_intent.supi or "").strip(),
            session_id=str(planning_request.context.session_id or "").strip(),
            snapshot_id=str(planning_request.context.snapshot_id or "").strip(),
            planning_metadata=planning_metadata,
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.objective_profile.get("profile_name")
                    or planning_request.operation_intent.objective_profile_hint
                    or decision.objective_profile_hint
                    or ""
                ).strip(),
                objective_tradeoff_summary=str(decision.rationale or planning_metadata["objective_breakdown"] or "").strip(),
                decisive_evidence=[
                    item
                    for item in [
                        "tool:preview_qos_optimizer" if decision.sm_policies else "",
                        "tool:inspect_mobility_ue_policies" if decision.am_policy is not None else "",
                        "mediator_constraints" if planning_request.context.unified_constraints else "",
                        "revision_requests" if planning_request.context.revision_requests else "",
                    ]
                    if item
                ],
                active_constraints=[
                    str(item)
                    for item in self.plan_compiler.artifact_compiler._normalized_hard_constraints(
                        planning_request.context.unified_constraints
                    )
                    if str(item).strip()
                ],
                explanation=str(decision.rationale or "").strip(),
                rejected_alternatives=[],
            ),
            all_policies=[],
        )

        for index, policy_details in enumerate(decision.sm_policies, start=1):
            flow_id = self._resolve_pcflow_id(policy_details, decision=decision)
            flow_ctx = self.plan_compiler.artifact_compiler._resolve_flow(planning_request, flow_id)
            app_id = str(flow_ctx.app_id or decision.selected_app_id or planning_request.operation_intent.app_id or "").strip()
            resource_keys = self.plan_compiler.advisor_validator._extract_qos_resource_keys(optimizer_preview, flow_id=flow_id)
            if not resource_keys:
                raise ValueError(f"optimizer preview does not contain a grounded QoS assignment for flow_id={flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[f"Apply PCF SM policy for {flow_id}"],
                    supi=str(flow_ctx.supi or planning_request.operation_intent.supi or "").strip(),
                    app_id=app_id,
                    flow_id=flow_id,
                    target_type="flow",
                    policy_id=f"smp-{app_id or 'app'}-{flow_id or index}",
                    policy_type="SmPolicyDecision",
                    resource_keys=resource_keys,
                    policy_details=dict(policy_details),
                )
            )

        if decision.am_policy is not None:
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[str(decision.rationale or "").strip()] if str(decision.rationale or "").strip() else [],
                    supi=str(planning_request.operation_intent.supi or "").strip(),
                    app_id="",
                    flow_id=None,
                    target_type="ue",
                    policy_id=self.plan_compiler.artifact_compiler._resolve_am_association_id(
                        planning_request=planning_request,
                        mobility_context=mobility_context,
                    ),
                    policy_type="PcfAmPolicyControlPolicyAssociation",
                    policy_details=dict(decision.am_policy),
                )
            )

        for index, policy_details in enumerate(decision.ursp_policies, start=1):
            flow_id = self._resolve_pcflow_id(policy_details, decision=decision, allow_missing=True)
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(planning_request.operation_intent.supi or "").strip(),
                    app_id=str(decision.selected_app_id or planning_request.operation_intent.app_id or "").strip(),
                    flow_id=flow_id,
                    target_type="flow" if flow_id else "app",
                    policy_id=f"ursp-{decision.selected_app_id or 'app'}-{flow_id or index}",
                    policy_type="UrspRuleRequest",
                    policy_details=dict(policy_details),
                )
            )

        normalized = normalize_policy_plan_draft(plan)
        self.plan_compiler.plan_validator.validate_compiled_plan(normalized, planning_request)
        return normalized

    @staticmethod
    def _resolve_pcflow_id(
        policy_details: Dict[str, Any],
        *,
        decision: SingleAgentRoundDecision,
        allow_missing: bool = False,
    ) -> str:
        from ...domain.policy_compiler import PolicyCompiler

        inferred_flow_id = str(PolicyCompiler.extract_flow_id(policy_details) or "").strip()
        if not inferred_flow_id and not allow_missing:
            raise ValueError("PCF-style policy payload does not contain a grounded flow_id")
        if not inferred_flow_id:
            return ""

        grounded_flow_ids = {
            str(flow.flow_id or "").strip()
            for flow in decision.flows or []
            if str(flow.flow_id or "").strip()
        }
        if grounded_flow_ids and inferred_flow_id not in grounded_flow_ids:
            raise ValueError(
                f"PCF-style policy payload flow_id {inferred_flow_id} is not present in grounded decision.flows"
            )
        if decision.selected_flow_id and inferred_flow_id != str(decision.selected_flow_id).strip():
            raise ValueError(
                f"PCF-style policy payload flow_id {inferred_flow_id} does not match selected_flow_id "
                f"{str(decision.selected_flow_id).strip()}"
            )
        return inferred_flow_id

    @staticmethod
    def _main_directives_from_decision(decision: SingleAgentRoundDecision) -> Dict[str, Any]:
        return {
            "requested_domains": decision.requested_domains,
            "domain_evidence": decision.domain_evidence,
            "objective_profile_hint": decision.objective_profile_hint,
        }

    @staticmethod
    def _to_intent_advisor_decision(decision: SingleAgentRoundDecision) -> Any:
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
        decision: SingleAgentRoundDecision,
        operation_intent: OperationIntent,
        session_id: str,
        snapshot_id: str,
        user_input: str,
    ) -> GlobalControlIntent:
        requested_domains = [ControlDomain(item) for item in decision.requested_domains]
        return GlobalControlIntent(
            session_id=session_id,
            snapshot_id=snapshot_id,
            raw_input=user_input,
            supi=str(operation_intent.supi or "").strip(),
            round_strategy=MainRoundStrategy.INITIAL_GROUNDING,
            next_agent="optimization_strategy",
            requested_domains=requested_domains,
            domain_evidence=decision.domain_evidence,
            control_semantics=operation_intent.control_semantics,
            objective_profile=ObjectiveProfile(
                profile_name=str(decision.objective_profile_hint or "").strip()
            ),
            investigation_targets=[],
            uncertainty_flags=[],
            required_evidence=self._required_evidence_from_domains(decision.requested_domains),
            forbidden_assumptions=[],
        )

    @staticmethod
    def _build_planning_request(
        *,
        global_intent: GlobalControlIntent,
        operation_intent: OperationIntent,
        session_id: str,
        snapshot_id: str,
        round_index: int,
        feedback_context: str,
        round_traces: List[Dict[str, Any]],
        previous_mediator_decision: Dict[str, Any],
    ) -> PlanningRequest:
        return PlanningRequest(
            operation_intent=operation_intent,
            context=PlanningContext(
                round_index=round_index,
                session_id=session_id,
                snapshot_id=snapshot_id,
                snapshot_metadata=get_latest_snapshot_metadata() or {},
                feedback_context=feedback_context,
                handoff_history=round_traces[-2:],
                active_domains=[item.value for item in global_intent.requested_domains],
                main_round_strategy=global_intent.round_strategy.value,
                objective_profile=global_intent.objective_profile.model_dump(mode="json"),
                forbidden_assumptions=list(global_intent.forbidden_assumptions or []),
                required_evidence=list(global_intent.required_evidence or []),
                revision_requests=list((previous_mediator_decision or {}).get("revision_requests") or []),
                unified_constraints=dict((previous_mediator_decision or {}).get("unified_constraints") or {}),
            ),
        )


__all__ = ["SingleControlAgent"]
