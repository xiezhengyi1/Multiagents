from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from shared.agents import BaseAgent, coerce_structured_response
from shared.agents.base import resolve_agent_model_name
from shared.runtime import ArtifactWorkerMixin, ContextPolicy, RuntimePersistence, ToolLoopExecutionError, extract_tool_calls, extract_tool_results
from shared.logging import log_event, log_timing
from application_runtime.persistence import create_runtime_persistence

from ...tool_capabilities import CONTROL_RUNTIME_CAPABILITY_ALIASES
from ...context.prompts.single import render_single_system_prompt
from ..grounding.contracts import (
    FlowSelector,
    GroundingDecision,
    QosOperationConstraint,
    SliceMigrationAuthorization,
)
from ..main.contracts import ControlDomain, GlobalControlIntent, MainRoundStrategy, ObjectiveProfile
from ..planning.contracts import (
    PlanningContext,
    PlanningRequest,
    PolicyPlanDraft,
    SharedControlContext,
)
from ..planning.compiler import OptimizationStrategyCompiler
from ..planning.evidence import extract_planning_tool_evidence
from ..planning.response_models import OsaAdvisorOutput
from ...context.prompts.planning import build_planning_validation_retry_prompt
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
    DEEPSEEK_MODEL = "deepseek-v4-flash"
    LARGE_MODEL_TOOL_RESULT_LIMITS = {
        "preview_qos_optimizer": 32000,
        "get_sm_ue_context": 8000,
        "get_sm_ue_flow_catalog": 32000,
        "get_am_policy_context": 8000,
        "fetch_qos_network_status": 32000,
        "inspect_mobility_ue_policies": 8000,
        "search_semantic_knowledge": 8000,
        "get_knowledge_by_key": 8000,
    }
    SMALL_MODEL_TOOL_RESULT_LIMITS = {
        "preview_qos_optimizer": 12000,
        "get_sm_ue_context": 4000,
        "get_sm_ue_flow_catalog": 12000,
        "get_am_policy_context": 4000,
        "fetch_qos_network_status": 8000,
        "inspect_mobility_ue_policies": 6000,
        "search_semantic_knowledge": 3000,
        "get_knowledge_by_key": 3000,
    }
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
        model_name: str = DEEPSEEK_MODEL,
        *,
        use_local_model: bool = False,
        rag_enabled: bool = True,
        persistence: RuntimePersistence | None = None,
    ) -> None:
        resolved_model_name = resolve_agent_model_name(model_name, use_local_model=use_local_model)
        use_deepseek_model = str(resolved_model_name or "").strip().lower().startswith("deepseek")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") if use_deepseek_model else None
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL") if use_deepseek_model else None
        if not use_local_model and use_deepseek_model:
            if not deepseek_api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is required for SingleControlAgent")
            if not deepseek_base_url:
                raise RuntimeError("DEEPSEEK_BASE_URL is required for SingleControlAgent")
        super().__init__(
            model_name=resolved_model_name,
            use_local_model=use_local_model,
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
        )
        if not use_local_model and use_deepseek_model:
            from langchain_deepseek import ChatDeepSeek
            class _SingleAgentChatDeepSeek(_SingleAgentChatDeepSeekMixin, ChatDeepSeek):
                pass
            raw_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "120")
            raw_max_retries = os.getenv("OPENAI_MAX_RETRIES", "2")
            self.llm = _SingleAgentChatDeepSeek(
                model=resolved_model_name,
                temperature=0,
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                timeout=float(raw_timeout),
                max_retries=int(raw_max_retries),
            )
        self.agent_name = "single_control"
        self.rag_enabled = rag_enabled
        self.plan_compiler = OptimizationStrategyCompiler()
        self.last_round_detail: Dict[str, Any] = {}
        runtime_persistence = persistence if persistence is not None else create_runtime_persistence()
        self.initialize_agent_runtime(logger_color="\033[96m", persistence=runtime_persistence)

    @staticmethod
    def _uses_small_model_context(model_name: str) -> bool:
        normalized = str(model_name or "").strip().lower()
        return not normalized.startswith("deepseek")

    @classmethod
    def _tool_result_limits_for_model(cls, model_name: str) -> Dict[str, int]:
        if cls._uses_small_model_context(model_name):
            return dict(cls.SMALL_MODEL_TOOL_RESULT_LIMITS)
        return dict(cls.LARGE_MODEL_TOOL_RESULT_LIMITS)

    @classmethod
    def _tool_context_policy_for_model(cls, model_name: str) -> ContextPolicy:
        small_model = cls._uses_small_model_context(model_name)
        limits = cls._tool_result_limits_for_model(model_name)
        return ContextPolicy(
            default_tool_result_chars=4000 if small_model else 8000,
            default_tool_result_tokens=1000 if small_model else 2000,
            tool_result_char_limits=limits,
            tool_result_token_limits={
                name: 1000 if small_model else 2000
                for name in limits
            },
            recent_tool_results_per_tool=1,
        )

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
    ) -> PolicyPlanDraft:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        self.last_round_detail = {
            "agent": self.agent_name,
            "model_name": self.model_name,
            "rag_enabled": self.rag_enabled,
            "session_id": session_id,
            "snapshot_id": snapshot_id,
            "round_index": round_index,
            "attempts": [],
        }
        log_event(self.logger, "single_control_round_plan_start", session_id=session_id, round_index=round_index)

        supi_match = re.search(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        supi = supi_match.group(1) if supi_match else ""
        evidence = {
            "user_input": str(user_input or "").strip(),
            "supi": supi,
        }
        token_budget, token_counter = self._resolve_token_context()
        context_policy = self._tool_context_policy_for_model(self.model_name)
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
        base_prompt = self._build_round_prompt(
            user_input=user_input,
            evidence=evidence,
            context=context,
            feedback_context=feedback_context,
        )
        current_prompt = base_prompt
        advisor_output: Optional[OsaAdvisorOutput] = None
        result: Dict[str, Any] = {}
        planning_tool_evidence: Dict[str, Any] = {}
        planning_request: Optional[PlanningRequest] = None
        contract_errors: List[str] = []

        # Match OSA's bounded repair behaviour: every retry gets the original
        # contract plus focused validation feedback and prior tool evidence.
        for attempt_index in range(3):
            attempt_detail: Dict[str, Any] = {
                "attempt_index": attempt_index + 1,
                "prompt_mode": "initial" if attempt_index == 0 else "validation_retry",
                "retry_issues": list(contract_errors),
            }
            advisor_agent = self.create_json_agent(
                tools=build_single_agent_tools(
                    rag_enabled=self.rag_enabled,
                    requested_domains=requested_domains,
                    allow_knowledge_tools=allow_knowledge_tools,
                ),
                system_prompt=render_single_system_prompt(),
                response_model=OsaAdvisorOutput,
                max_iterations=12,
                tool_error_mode="raise",
                forbid_duplicate_tool_calls=True,
                max_tool_result_chars=context_policy.default_tool_result_chars,
                tool_result_limits=context_policy.tool_result_char_limits,
                max_tool_result_tokens=context_policy.default_tool_result_tokens,
                tool_result_token_limits=context_policy.tool_result_token_limits,
                context_policy=context_policy,
                capability_aliases=CONTROL_RUNTIME_CAPABILITY_ALIASES,
            )
            try:
                result = self._invoke_round_advisor(
                    advisor_agent=advisor_agent,
                    prompt=current_prompt,
                    runtime_context=runtime_context,
                )
                advisor_output = coerce_structured_response(
                    result,
                    OsaAdvisorOutput,
                    error_message="Single control agent returned no OsaAdvisorOutput",
                )
                planning_tool_evidence = extract_planning_tool_evidence(advisor_result=result)
                attempt_detail["tool_calls"] = extract_tool_calls(result.get("messages") or [])
                attempt_detail["tool_results"] = extract_tool_results(result.get("messages") or [])
                attempt_detail["osa_advisor_output"] = advisor_output.model_dump(mode="json")
                attempt_detail["planning_tool_evidence"] = planning_tool_evidence
                planning_request = self._build_single_planning_request(
                    advisor_output=advisor_output,
                    advisor_result=result,
                    user_input=user_input,
                    supi=supi,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    round_index=round_index,
                    feedback_context=feedback_context,
                    round_traces=round_traces or [],
                    previous_mediator_decision=previous_mediator_decision or {},
                )
                grounding_tools = [
                    str(item.get("name") or "").strip()
                    for item in extract_tool_results(result.get("messages") or [])
                    if str(item.get("name") or "").strip()
                ]
                contract_errors = self.plan_compiler.validate_advisor_output(
                    advisor_output=advisor_output,
                    planning_request=planning_request,
                    grounding_tools=grounding_tools,
                    planning_tool_evidence=planning_tool_evidence,
                )
                attempt_detail["materialized_planning_request"] = planning_request.model_dump(mode="json")
                attempt_detail["contract_errors"] = list(contract_errors)
                attempt_detail["status"] = "validated" if not contract_errors else "contract_rejected"
                self.last_round_detail["attempts"].append(attempt_detail)
                if not contract_errors:
                    break
            except Exception as exc:
                contract_errors = [str(exc)]
                attempt_detail["status"] = "invocation_or_processing_error"
                attempt_detail["error_type"] = type(exc).__name__
                attempt_detail["error"] = str(exc)
                if result:
                    attempt_detail["tool_calls"] = extract_tool_calls(result.get("messages") or [])
                    attempt_detail["tool_results"] = extract_tool_results(result.get("messages") or [])
                self.last_round_detail["attempts"].append(attempt_detail)

            if attempt_index == 2:
                self.last_round_detail["status"] = "failed"
                self.last_round_detail["failure_reason"] = "; ".join(contract_errors)
                raise RuntimeError(
                    "Single control OSA-equivalent contract validation failed: "
                    + "; ".join(contract_errors)
                )
            current_prompt = build_planning_validation_retry_prompt(
                base_prompt=base_prompt,
                issues=contract_errors,
                cached_planning_evidence=planning_tool_evidence,
            )

        if advisor_output is None or planning_request is None:
            raise RuntimeError("Single control agent returned no OSA-compatible planning result")
        try:
            policy_plan = self.plan_compiler.assemble_policy_plan(
                advisor_output=advisor_output,
                planning_request=planning_request,
                planning_tool_evidence=planning_tool_evidence,
            )
        except Exception as exc:
            self.last_round_detail.update(
                {
                    "status": "compiler_error",
                    "error_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "final_osa_advisor_output": advisor_output.model_dump(mode="json"),
                }
            )
            raise
        self.last_round_detail.update(
            {
                "status": "success",
                "final_osa_advisor_output": advisor_output.model_dump(mode="json"),
                "final_policy_plan": policy_plan.model_dump(mode="json"),
                "attempt_count": len(self.last_round_detail["attempts"]),
            }
        )
        log_timing(self.logger, "single_control_round_plan_total", time.perf_counter() - total_start, status="success")
        return policy_plan

    def _build_single_planning_request(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        advisor_result: Dict[str, Any],
        user_input: str,
        supi: str,
        session_id: str,
        snapshot_id: str,
        round_index: int,
        feedback_context: str,
        round_traces: List[Dict[str, Any]],
        previous_mediator_decision: Dict[str, Any],
    ) -> PlanningRequest:
        """Materialize the hidden IEA handoff used by the single-agent loop.

        The model never emits that handoff.  It is reconstructed only from
        catalog/tool evidence and the OSA-compatible output, which keeps the
        same ownership boundary as Main -> IEA -> OSA.
        """
        active_domains = self._single_output_domains(advisor_output, user_input)
        rows_by_id = self._catalog_rows_by_flow_id(advisor_result)
        requested_flow_ids = [str(item.flow_id or "").strip() for item in advisor_output.sm_policies]
        requested_flow_ids.extend(
            str(item.get("flow_id") or "").strip()
            for item in advisor_output.partial_policies
            if isinstance(item, dict)
        )
        requested_flow_ids = list(dict.fromkeys(item for item in requested_flow_ids if item))
        flows: List[FlowSelector] = []
        for flow_id in requested_flow_ids:
            row = rows_by_id.get(flow_id, {})
            if row:
                flows.append(FlowSelector.model_validate(
                    self._flow_selector_payload(supi=supi, flow_id=flow_id, row=row)
                ))
                continue
            matching = next((item for item in advisor_output.sm_policies if item.flow_id == flow_id), None)
            app_id = str(getattr(matching, "app_id", "") or "").strip()
            if flow_id and app_id:
                flows.append(FlowSelector(supi=supi, app_id=app_id, flow_id=flow_id, name=flow_id))

        retry_scope = "target_stable" if "retry_contract:" in feedback_context else "full_reground"
        profile_name = self._single_objective_profile(user_input)
        main_intent = GlobalControlIntent(
            raw_input=str(user_input or "").strip(),
            supi=supi,
            next_agent="optimization_strategy",
            round_strategy=(MainRoundStrategy.POLICY_REVISION if round_index > 1 else MainRoundStrategy.INITIAL_GROUNDING),
            requested_domains=[ControlDomain(item) for item in active_domains],
            objective_profile=ObjectiveProfile(profile_name=profile_name),
            retry_scope=retry_scope,
            required_evidence=self._required_evidence_from_domains(active_domains),
        )
        planning_context = PlanningContext(
            round_index=round_index,
            session_id=session_id,
            snapshot_id=snapshot_id,
            shared_context=SharedControlContext(main_intent=main_intent),
            feedback_context=feedback_context,
            handoff_history=round_traces[-2:],
            revision_requests=list(previous_mediator_decision.get("revision_requests") or []),
            unified_constraints=dict(previous_mediator_decision.get("unified_constraints") or {}),
        )
        migration_required = self._requires_serving_slice_change(user_input)
        subscription = self._single_subscription_evidence(advisor_result)
        authorized_snssais = [
            str(item or "").strip()
            for item in (subscription.get("authorized_snssais") or [])
            if str(item or "").strip()
        ]
        migration_decision = "not_applicable"
        if migration_required:
            migration_decision = "migration_authorized" if authorized_snssais else "evidence_missing"
        return PlanningRequest(
            grounding_decision=GroundingDecision(
                flows=flows,
                grounding_evidence={
                    "grounded_supi": supi,
                    "grounded_flows": [item.model_dump(mode="json") for item in flows],
                    "evidence_sources": {"single_agent": ["get_sm_ue_flow_catalog"] if flows else []},
                },
                qos_operation_constraints=[
                    QosOperationConstraint(
                        flow_id=str(flow.flow_id or "").strip(),
                        app_id=str(flow.app_id or "").strip(),
                        operation_type="slice_migration" if migration_required else "qos_reallocation",
                        require_slice_change=migration_required,
                        no_op_allowed=not migration_required,
                        rationale=["Single-agent hidden IEA migration decision"] if migration_required else [],
                    )
                    for flow in flows
                    if str(flow.flow_id or "").strip() and str(flow.app_id or "").strip()
                ],
                slice_migration_authorization=SliceMigrationAuthorization(
                    decision=migration_decision,
                    authority=str(subscription.get("authority") or "").strip(),
                    authorized_snssais=authorized_snssais,
                    rationale=["Read-only subscription entitlement evidence was observed"] if subscription else [],
                ),
            ),
            context=planning_context,
        )

    @staticmethod
    def _single_output_domains(advisor_output: OsaAdvisorOutput, user_input: str) -> List[str]:
        domains: List[str] = []
        if advisor_output.sm_policies or any(
            isinstance(item, dict) and str(item.get("policy_type") or "").lower() == "smpolicydecision"
            for item in advisor_output.partial_policies
        ):
            domains.append("qos")
        if advisor_output.am_policy is not None:
            domains.append("mobility")
        lowered = str(user_input or "").lower()
        if not domains:
            domains.append("qos")
        mobility_markers = ("am policy", "rfsp", "allowed nssai", "target nssai", "handover", "registration", "服务区域", "移动性策略")
        if any(marker in lowered for marker in mobility_markers) and not any(
            marker in lowered for marker in ("do not change mobility", "不要动 mobility", "别影响移动性")
        ):
            domains.append("mobility")
        if SingleControlAgent._requires_serving_slice_change(user_input):
            domains.append("mobility")
        return list(dict.fromkeys(domains))

    @staticmethod
    def _requires_serving_slice_change(user_input: str) -> bool:
        lowered = str(user_input or "").lower()
        explicit_markers = (
            "move off the current slice",
            "change serving snssai",
            "serving s-nssai",
            "target nssai",
            "allowed nssai after",
            "迁移到",
            "切换到",
            "更换服务切片",
            "目标切片",
        )
        return any(marker in lowered for marker in explicit_markers)

    @classmethod
    def _single_subscription_evidence(cls, advisor_result: Dict[str, Any]) -> Dict[str, Any]:
        for result in reversed(extract_tool_results((advisor_result or {}).get("messages") or [])):
            if str(result.get("name") or "").strip() != "get_ue_slice_subscription":
                continue
            payload = cls._parse_json_content(result.get("content"))
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _single_objective_profile(user_input: str) -> str:
        lowered = str(user_input or "").lower()
        if any(token in lowered for token in ("latency", "delay", "时延")):
            return "latency"
        if any(token in lowered for token in ("throughput", "bandwidth", "带宽", "吞吐")):
            return "throughput"
        if any(token in lowered for token in ("stability", "stable", "稳定", "jitter", "可靠")):
            return "stability"
        return "balanced"

    @classmethod
    def _catalog_rows_by_flow_id(cls, advisor_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        rows_by_flow_id: Dict[str, Dict[str, Any]] = {}
        for result in extract_tool_results((advisor_result or {}).get("messages") or []):
            if str(result.get("name") or "").strip() != "get_sm_ue_flow_catalog":
                continue
            payload = cls._parse_json_content(result.get("content"))
            if not isinstance(payload, dict):
                continue
            rows = payload.get("flow_catalog") if isinstance(payload.get("flow_catalog"), list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                flow_id = str(row.get("flow_id") or "").strip()
                if flow_id:
                    rows_by_flow_id[flow_id] = row
        return rows_by_flow_id

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
            f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
            "Round context:\n"
            f"{context or 'N/A'}\n\n"
            f"{feedback_block}"
            "Task:\n"
            "- Execute Main-equivalent routing, IEA-equivalent grounding, then OSA-equivalent planning in that order.\n"
            "- Keep the first two stages internal; only return the final OsaAdvisorOutput.\n"
            "- For QoS, sm_policies must use exact catalog-grounded flow_id/app_id pairs and current-round optimizer evidence.\n"
            "- For mobility, am_policy must use current-round mobility evidence.\n"
            "- Respect every retry_contract field as a binding repair constraint.\n\n"
        )

    @staticmethod
    def _parse_json_content(content: Any) -> Any:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                    parts.append(str(block.get("text") or ""))
            text = "".join(parts).strip()
        else:
            text = str(content or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if not text:
            return None
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
            fenced = match.group(1).strip()
            if not fenced:
                continue
            try:
                return json.loads(fenced)
            except Exception:
                continue
        try:
            return json.loads(text)
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                return parsed
            except Exception:
                continue
        return None

    @staticmethod
    def _flow_selector_payload(*, supi: str, flow_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        service = row.get("service") if isinstance(row.get("service"), dict) else {}
        sla = row.get("sla") if isinstance(row.get("sla"), dict) else {}
        allocation = row.get("allocation") if isinstance(row.get("allocation"), dict) else {}
        traffic = row.get("traffic") if isinstance(row.get("traffic"), dict) else {}
        flow_name = str(row.get("flow_name") or row.get("name") or flow_id).strip()
        return {
            "supi": supi,
            "app_id": str(row.get("app_id") or "").strip(),
            "app_name": str(row.get("app_name") or "").strip() or None,
            "flow_id": flow_id,
            "target_type": "flow",
            "name": flow_name,
            "service_type": str(service.get("service_type") or "").strip() or None,
            "service_type_id": service.get("service_type_id"),
            "bw_ul": sla.get("bandwidth_ul"),
            "bw_dl": sla.get("bandwidth_dl"),
            "gbr_ul": sla.get("guaranteed_bandwidth_ul"),
            "gbr_dl": sla.get("guaranteed_bandwidth_dl"),
            "lat": sla.get("latency"),
            "loss_req": sla.get("loss_rate"),
            "jitter_req": sla.get("jitter"),
            "priority": sla.get("priority"),
            "description": flow_name,
            "five_tuple": list(traffic.get("five_tuple")) if isinstance(traffic.get("five_tuple"), (list, tuple)) else None,
            "current_slice_snssai": str(allocation.get("current_slice_snssai") or "").strip() or None,
            "current_bw_ul": allocation.get("allocated_bandwidth_ul"),
            "current_bw_dl": allocation.get("allocated_bandwidth_dl"),
            "resolution_status": "resolved",
        }

    @classmethod
    def _hint_requested_domains(cls, user_input: str) -> Optional[List[str]]:
        # Keep single-agent domain routing model-driven.  The prompt owns the
        # domain boundary rules; pre-filtering tools here made natural-language
        # exclusions brittle and could expose the wrong tool subset.
        del user_input
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
            raise RuntimeError(f"Single control round advisor invocation failed: {exc}") from exc

    @staticmethod
    def _required_evidence_from_domains(domains: List[str]) -> List[str]:
        required: List[str] = []
        normalized = {str(item or "").strip().lower() for item in domains if str(item or "").strip()}
        if "qos" in normalized:
            required.append("qos_runtime_evidence")
        if "mobility" in normalized:
            required.append("mobility_policy_context")
        return required



__all__ = ["SingleControlAgent"]
