from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from agent_runtime import ArtifactEnvelope
from agent_runtime.messages import extract_tool_calls, extract_tool_results
from agents.BaseAgent import BaseAgent, coerce_structured_response, extract_grounding_tool_names
from agents.worker import ArtifactWorkerMixin
from agents.tools.knowledge_tool import get_knowledge_by_key, search_semantic_knowledge
from agents.tools.pcf_tools import (
    get_am_policy_context,
    get_sm_ue_context,
    get_sm_ue_flow_catalog,
    search_am_policy_targets,
    search_sm_flow_targets,
)
from domain.policy_plan import OperationIntent
from utils.logger import log_event, log_timing

from .compiler import IntentCompiler
from .contracts import IntentAdvisorDecision, IntentEvidence
from .prompts import IEA_SYSTEM_PROMPT


@dataclass
class IntentAdvisorInvocation:
    advisor_result: Dict[str, Any]
    trace_agent: Any
    trace_payload: Dict[str, Any]
    runtime_context: Any

    def write_final_trace(
        self,
        *,
        advisor_decision: IntentAdvisorDecision,
        operation_intent: OperationIntent | None,
        status: str,
        error: str | None = None,
    ) -> None:
        payload = dict(self.trace_payload)
        metadata = dict(payload.get("trace_metadata") or {})
        metadata["advisor_decision"] = advisor_decision.model_dump(mode="json")
        metadata["compiler_output"] = None if operation_intent is None else operation_intent.model_dump(mode="json")
        payload["trace_metadata"] = metadata
        self.trace_agent.write_trace(
            payload=payload,
            context=self.runtime_context,
            result=self.advisor_result,
            status=status,
            error=error,
            structured_response_override=metadata["compiler_output"],
        )


class IntentEncodingAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "intent_encoding"
    SM_GROUNDING_TOOLS = {"search_sm_flow_targets", "get_sm_ue_context", "get_sm_ue_flow_catalog"}
    AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}
    GROUNDING_TOOLS = SM_GROUNDING_TOOLS | AM_GROUNDING_TOOLS | {"search_semantic_knowledge", "get_knowledge_by_key"}

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507", use_local_model: bool = False) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "intent_encoding"
        self.compiler = IntentCompiler()
        self.initialize_agent_runtime(logger_color="\033[95m")
        self._ensure_intent_caches()
        self.tools = [
            search_sm_flow_targets,
            get_sm_ue_context,
            get_sm_ue_flow_catalog,
            get_am_policy_context,
            search_am_policy_targets,
            search_semantic_knowledge,
            get_knowledge_by_key,
        ]
        self.advisor_agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=IEA_SYSTEM_PROMPT,
            response_model=IntentAdvisorDecision,
            max_iterations=14,
        )

    def _ensure_intent_caches(self) -> None:
        if not hasattr(self, "compiler"):
            self.compiler = IntentCompiler()

    # 关键步骤：SM 与 AM grounding 缓存分桶，避免上一轮跨域工具结果污染当前 evidence。
    def _get_cached_sm_flow_catalog(self, supi: str, *, snapshot_id: str = "") -> Dict[str, Any]:
        self._ensure_intent_caches()
        return self.get_cached_runtime_value(
            "sm_ue_flow_catalog",
            str(supi or "").strip(),
            snapshot_id=snapshot_id,
            default={},
        ) or {}

    def _cache_sm_flow_catalog(self, supi: str, payload: Dict[str, Any], *, snapshot_id: str = "") -> None:
        normalized_supi = str(supi or payload.get("supi") or "").strip()
        if not normalized_supi or not isinstance(payload, dict):
            return
        normalized_payload = dict(payload)
        normalized_payload["supi"] = normalized_supi
        self.cache_runtime_value(
            "sm_ue_flow_catalog",
            normalized_supi,
            normalized_payload,
            snapshot_id=snapshot_id,
        )

    def _cache_sm_ue_context(self, supi: str, payload: Dict[str, Any], *, snapshot_id: str = "") -> None:
        normalized_supi = str(supi or "").strip()
        if not normalized_supi or not isinstance(payload, dict):
            return
        self.cache_runtime_value(
            "sm_ue_context",
            normalized_supi,
            dict(payload),
            snapshot_id=snapshot_id,
        )

    def _get_cached_am_policy_context(self, supi: str, *, snapshot_id: str = "") -> Dict[str, Any]:
        self._ensure_intent_caches()
        return self.get_cached_runtime_value(
            "am_policy_context",
            str(supi or "").strip(),
            snapshot_id=snapshot_id,
            default={},
        ) or {}

    def _cache_am_policy_context(self, supi: str, payload: Dict[str, Any], *, snapshot_id: str = "") -> None:
        normalized_supi = str(supi or payload.get("supi") or "").strip()
        if not normalized_supi or not isinstance(payload, dict):
            return
        normalized_payload = dict(payload)
        normalized_payload["supi"] = normalized_supi
        self.cache_runtime_value(
            "am_policy_context",
            normalized_supi,
            normalized_payload,
            snapshot_id=snapshot_id,
        )

    def _cache_sm_flow_search(
        self,
        *,
        snapshot_id: str,
        app_name: str,
        flow_name: str,
        limit: int,
        payload: Dict[str, Any],
    ) -> None:
        if not isinstance(payload, dict):
            return
        self.cache_runtime_value(
            "sm_flow_search",
            (
                str(app_name or "").strip().lower(),
                str(flow_name or "").strip().lower(),
                int(limit or 5),
            ),
            dict(payload),
            snapshot_id=snapshot_id,
        )

    def _cache_am_policy_search(
        self,
        *,
        snapshot_id: str,
        supi: str,
        association_id: str,
        allowed_snssai: str,
        target_snssai: str,
        service_area: str,
        rfsp: str,
        access_type: str,
        limit: int,
        payload: Dict[str, Any],
    ) -> None:
        if not isinstance(payload, dict):
            return
        self.cache_runtime_value(
            "am_policy_search",
            (
                str(supi or "").strip().lower(),
                str(association_id or "").strip().lower(),
                str(allowed_snssai or "").strip().lower(),
                str(target_snssai or "").strip().lower(),
                str(service_area or "").strip().lower(),
                str(rfsp or "").strip().lower(),
                str(access_type or "").strip().lower(),
                int(limit or 5),
            ),
            dict(payload),
            snapshot_id=snapshot_id,
        )

    def expected_request_type(self) -> str:
        return "OperationIntentRequest"

    def response_artifact_type(self) -> str:
        return "OperationIntent"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> OperationIntent:
        payload = envelope.payload or {}
        return self.analyze_operation_intent(
            user_input=str(payload.get("user_input") or ""),
            context=str(payload.get("context") or ""),
            conversation_messages=payload.get("messages"),
            allow_user_interaction=bool(payload.get("allow_user_interaction", False)),
            session_id=envelope.session_id,
            snapshot_id=envelope.snapshot_id,
            request_envelope=envelope,
        )

    def _cache_received_request(
        self,
        *,
        user_input: str,
        context: str,
        conversation_messages: List[Dict[str, Any]] | None,
        allow_user_interaction: bool,
        session_id: str,
        snapshot_id: str,
    ) -> ArtifactEnvelope:
        return self.cache_received_artifact(
            artifact_type="OperationIntentRequest",
            payload={
                "user_input": str(user_input),
                "context": str(context or ""),
                "messages": list(conversation_messages or []),
                "allow_user_interaction": bool(allow_user_interaction),
            },
            session_id=session_id,
            snapshot_id=snapshot_id,
        )

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        operation_intent: OperationIntent,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="OperationIntent",
            request_envelope=request_envelope,
            payload=operation_intent,
        )

    @staticmethod
    def _postprocess_operation_intent(result: OperationIntent, *, user_input: str = "") -> OperationIntent:
        if not isinstance(result, OperationIntent):
            raise TypeError("_postprocess_operation_intent expects an OperationIntent instance")
        payload = result.model_dump(mode="json")
        payload["raw_input"] = str(payload.get("raw_input") or user_input or "").strip()
        top_level_supi = str(payload.get("supi") or "").strip()
        flows = payload.get("flows") or []
        if not top_level_supi and isinstance(flows, list):
            for flow in flows:
                if not isinstance(flow, dict):
                    continue
                candidate = str(flow.get("supi") or "").strip()
                if candidate:
                    top_level_supi = candidate
                    break
        if top_level_supi:
            payload["supi"] = top_level_supi
            if isinstance(flows, list):
                for flow in flows:
                    if isinstance(flow, dict) and not str(flow.get("supi") or "").strip():
                        flow["supi"] = top_level_supi
        return OperationIntent.model_validate(payload)

    def analyze_operation_intent(
        self,
        user_input: str,
        context: str = "",
        conversation_messages: List[Dict[str, Any]] | None = None,
        *,
        session_id: str = "",
        snapshot_id: str = "",
        allow_user_interaction: bool = False,
        request_envelope: ArtifactEnvelope | None = None,
    ) -> OperationIntent:
        self.ensure_worker_runtime_initialized()
        if request_envelope is None:
            request_envelope = self._cache_received_request(
                user_input=user_input,
                context=context,
                conversation_messages=conversation_messages,
                allow_user_interaction=allow_user_interaction,
                session_id=session_id,
                snapshot_id=snapshot_id,
            )

        analyze_kwargs = {
            "context": context,
            "session_id": session_id,
            "snapshot_id": snapshot_id,
            "allow_user_interaction": allow_user_interaction,
        }

        operation_intent = self.analyze_intent(
            user_input,
            **analyze_kwargs,
        )
        self._cache_produced_result(
            request_envelope=request_envelope,
            operation_intent=operation_intent,
        )
        return operation_intent

    def analyze_intent(
        self,
        user_input: str,
        context: str = "",
        *,
        session_id: str = "",
        snapshot_id: str = "",
        allow_user_interaction: bool = False,
    ) -> OperationIntent:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "iea_analyze_start")
        main_directives = self.compiler.extract_main_directives(context)

        try:
            evidence = self._extract_intent_evidence(
                user_input=user_input,
                snapshot_id=snapshot_id,
                main_directives=main_directives,
            )
            runtime_context = self.build_runtime_context(
                agent_name=self.agent_name,
                session_id=session_id,
                snapshot_id=snapshot_id,
                supi=evidence.supi or None,
                thread_id=session_id,
                allow_user_interaction=allow_user_interaction,
            )
            advisor_invocation = self._invoke_intent_advisor(
                evidence=evidence,
                context=context,
                runtime_context=runtime_context,
            )
            advisor_result = advisor_invocation.advisor_result
            advisor_decision = coerce_structured_response(
                advisor_result,
                IntentAdvisorDecision,
                error_message="IEA advisor returned no structured_response",
            )
            advisor_validation_errors = self.compiler.validate_advisor_decision(
                evidence=evidence,
                decision=advisor_decision,
            )
            if advisor_validation_errors:
                raise RuntimeError("IEA advisor decision validation failed: " + "; ".join(advisor_validation_errors))
            grounding_tools = extract_grounding_tool_names(advisor_result, self.GROUNDING_TOOLS)
            evidence = self._refresh_intent_evidence_from_tool_results(
                evidence=evidence,
                advisor_result=advisor_result,
                main_directives=main_directives,
                snapshot_id=snapshot_id,
            )
            validation_errors = self.compiler.validate_intent_grounding(
                evidence=evidence,
                grounding_tools=grounding_tools,
            )
            if validation_errors:
                raise RuntimeError("IEA grounding validation failed: " + "; ".join(validation_errors))
            compiled = self.compiler.compile_operation_intent(
                evidence=evidence,
                advisor_decision=advisor_decision,
                user_input=user_input,
                session_id=session_id,
                snapshot_id=snapshot_id,
                main_directives=main_directives,
            )
            advisor_invocation.write_final_trace(
                advisor_decision=advisor_decision,
                operation_intent=compiled,
                status="success",
            )
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
            return compiled
        except Exception as exc:
            if "advisor_invocation" in locals() and "advisor_decision" in locals():
                advisor_invocation.write_final_trace(
                    advisor_decision=advisor_decision,
                    operation_intent=None,
                    status="error",
                    error=str(exc),
                )
            self.logger.error(f"Failed to analyze operation intent: {exc}")
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    def _invoke_intent_advisor(
        self,
        *,
        evidence: IntentEvidence,
        context: str,
        runtime_context: Any,
    ) -> IntentAdvisorInvocation:
        prompt = (
            "User request:\n"
            f"{evidence.user_input}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(evidence.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            "Coordinator context:\n"
            f"{context or 'N/A'}\n\n"
            "Resolve only the semantic choices that remain ambiguous."
        )
        self._pending_invoke_messages = [{"role": "user", "content": prompt}]
        base_trace_metadata = getattr(self, "_pending_trace_metadata", {}) or {}
        invoke_payload = {
            "messages": self._pending_invoke_messages,
            "trace_write_mode": "manual",
            "trace_metadata": {
                **base_trace_metadata,
                "path_label": "advisor_path",
            },
        }
        try:
            result = self.advisor_agent.invoke(invoke_payload, context=runtime_context)
        except Exception as exc:
            repair_prompt = (
                f"{prompt}\n\n"
                "Your previous response was not valid JSON for IntentAdvisorDecision.\n"
                f"Parser error: {exc}\n\n"
                "Return only one JSON object. Do not use markdown fences."
            )
            self._pending_invoke_messages = [{"role": "user", "content": repair_prompt}]
            invoke_payload = {
                "messages": self._pending_invoke_messages,
                "trace_write_mode": "manual",
                "trace_metadata": {
                    **base_trace_metadata,
                    "path_label": "advisor_path",
                },
            }
            result = self.advisor_agent.invoke(invoke_payload, context=runtime_context)
        return IntentAdvisorInvocation(
            advisor_result=result,
            trace_agent=self.advisor_agent,
            trace_payload=invoke_payload,
            runtime_context=runtime_context,
        )

    def _extract_intent_evidence(
        self,
        *,
        user_input: str,
        snapshot_id: str,
        main_directives: Dict[str, Any],
    ) -> IntentEvidence:
        self._ensure_intent_caches()
        normalized_input = str(user_input or "").strip()
        supi_match = re.search(r"(?i)(imsi-\d{5,})", normalized_input)
        supi = str(main_directives.get("supi") or "").strip() or (supi_match.group(1) if supi_match else "")
        requested_domains = list(main_directives.get("requested_domains") or [])
        return self.compiler.build_intent_evidence(
            user_input=normalized_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload=(
                self._get_cached_sm_flow_catalog(supi, snapshot_id=snapshot_id)
                if self.compiler.uses_sm_grounding(requested_domains)
                else {}
            ),
            semantic_candidates=[],
            am_context_payload=(
                self._get_cached_am_policy_context(supi, snapshot_id=snapshot_id)
                if self.compiler.uses_am_grounding(requested_domains)
                else {}
            ),
            am_policy_candidates=[],
        )

    def _build_intent_evidence(
        self,
        *,
        user_input: str,
        supi: str,
        main_directives: Dict[str, Any],
        catalog_payload: Dict[str, Any],
        semantic_candidates: List[Dict[str, Any]],
        am_context_payload: Dict[str, Any] | None = None,
        am_policy_candidates: List[Dict[str, Any]] | None = None,
    ) -> IntentEvidence:
        return self.compiler.build_intent_evidence(
            user_input=user_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )

    def _refresh_intent_evidence_from_tool_results(
        self,
        *,
        evidence: IntentEvidence,
        advisor_result: Dict[str, Any],
        main_directives: Dict[str, Any],
        snapshot_id: str,
    ) -> IntentEvidence:
        self._ensure_intent_caches()
        tool_calls = {
            str(call.get("id") or "").strip(): call
            for call in extract_tool_calls(advisor_result.get("messages") or [])
            if str(call.get("id") or "").strip()
        }
        semantic_candidates: List[Dict[str, Any]] = list(evidence.cached_semantic_candidates or [])
        catalog_payload = dict(evidence.cached_catalog or {})
        am_context_payload = dict(evidence.cached_am_context or {})
        am_policy_candidates: List[Dict[str, Any]] = list(evidence.cached_am_policy_candidates or [])
        requested_domains = list(evidence.requested_domains or [])

        for result in extract_tool_results(advisor_result.get("messages") or []):
            tool_name = str(result.get("name") or "").strip()
            call_id = str(result.get("tool_call_id") or "").strip()
            call_args = tool_calls.get(call_id, {}).get("args") if call_id else {}
            if not isinstance(call_args, dict):
                call_args = {}

            if tool_name == "get_sm_ue_flow_catalog" and self.compiler.uses_sm_grounding(requested_domains):
                payload = self.compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="SM UE Flow Catalog Retrieved:",
                )
                if payload:
                    cached_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                    self._cache_sm_flow_catalog(cached_supi, payload, snapshot_id=snapshot_id)
                    if cached_supi == str(evidence.supi or "").strip():
                        catalog_payload = dict(payload)
            elif tool_name == "get_sm_ue_context" and self.compiler.uses_sm_grounding(requested_domains):
                payload = self.compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="SM UE Context Retrieved:",
                )
                if payload:
                    cached_supi = str(call_args.get("supi") or evidence.supi or "").strip()
                    self._cache_sm_ue_context(cached_supi, payload, snapshot_id=snapshot_id)
            elif tool_name == "search_sm_flow_targets" and self.compiler.uses_sm_grounding(requested_domains):
                payload = self.compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="SM Flow Target Search Retrieved:",
                )
                if payload:
                    self._cache_sm_flow_search(
                        snapshot_id=snapshot_id,
                        app_name=str(call_args.get("app_name") or "").strip(),
                        flow_name=str(call_args.get("flow_name") or "").strip(),
                        limit=int(call_args.get("limit") or 5),
                        payload=payload,
                    )
                    semantic_candidates = list(payload.get("candidates") or [])
            elif tool_name == "get_am_policy_context" and self.compiler.uses_am_grounding(requested_domains):
                payload = self.compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="AM Policy Context Retrieved:",
                )
                if payload:
                    cached_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                    self._cache_am_policy_context(cached_supi, payload, snapshot_id=snapshot_id)
                    if cached_supi == str(evidence.supi or "").strip():
                        am_context_payload = dict(payload)
            elif tool_name == "search_am_policy_targets" and self.compiler.uses_am_grounding(requested_domains):
                payload = self.compiler.parse_json_payload_from_tool_result(
                    result.get("content"),
                    marker="AM Policy Target Search Retrieved:",
                )
                if payload:
                    self._cache_am_policy_search(
                        snapshot_id=snapshot_id,
                        supi=str(call_args.get("supi") or "").strip(),
                        association_id=str(call_args.get("association_id") or "").strip(),
                        allowed_snssai=str(call_args.get("allowed_snssai") or "").strip(),
                        target_snssai=str(call_args.get("target_snssai") or "").strip(),
                        service_area=str(call_args.get("service_area") or "").strip(),
                        rfsp=str(call_args.get("rfsp") or "").strip(),
                        access_type=str(call_args.get("access_type") or "").strip(),
                        limit=int(call_args.get("limit") or 5),
                        payload=payload,
                    )
                    am_policy_candidates = list(payload.get("candidates") or [])

        return self.compiler.build_intent_evidence(
            user_input=evidence.user_input,
            supi=evidence.supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )
