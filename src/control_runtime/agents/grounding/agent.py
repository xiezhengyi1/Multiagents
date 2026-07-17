from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from shared.runtime import ArtifactEnvelope
from shared.runtime import ContextPolicy
from shared.runtime import ToolLoopExecutionError
from shared.agents import BaseAgent, coerce_structured_response, extract_grounding_tool_names
from shared.runtime import ArtifactWorkerMixin
from knowledge_runtime.retrieval.raw import get_knowledge_by_key, search_semantic_knowledge
from ...integrations.pcf import (
    get_am_policy_context,
    get_sm_ue_context,
    get_sm_ue_flow_catalog,
    get_ue_slice_subscription,
    search_am_policy_targets,
    search_sm_flow_targets,
)
from ...context.projectors import project_intent_evidence_for_prompt
from ...context.observability import measure_context_components
from ...context.evidence.grounding import IntentEvidenceBuilder
from ...domain.policy_plan import GroundingDecision
from ...context.prompts import GroundingPromptBuilder, RetryPromptBuilder
from ..common import validate_grounding_decision
from shared.logging import log_event, log_timing

from .compiler import IntentCompiler
from .common import extract_requested_supis, merge_candidate_dicts, merge_catalog_payloads
from .contracts import IntentEvidence
from ...context.prompts import IEA_DYNAMIC_RULES
from .tool_result_adapter import extract_grounding_tool_payloads


@dataclass
class IntentAdvisorInvocation:
    advisor_result: Dict[str, Any]
    trace_agent: Any
    trace_payload: Dict[str, Any]
    runtime_context: Any

    def write_final_trace(
        self,
        *,
        grounding_decision: GroundingDecision | None,
        status: str,
        error: str | None = None,
    ) -> None:
        payload = dict(self.trace_payload)
        metadata = dict(payload.get("trace_metadata") or {})
        metadata["grounding_decision"] = None if grounding_decision is None else grounding_decision.model_dump(mode="json")
        payload["trace_metadata"] = metadata
        self.trace_agent.write_trace(
            payload=payload,
            context=self.runtime_context,
            result=self.advisor_result,
            status=status,
            error=error,
            structured_response_override=metadata["grounding_decision"],
        )


class IntentEncodingAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "intent_encoding"
    SM_GROUNDING_TOOLS = {"search_sm_flow_targets", "get_sm_ue_context", "get_sm_ue_flow_catalog"}
    AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}
    SUBSCRIPTION_GROUNDING_TOOLS = {"get_ue_slice_subscription"}
    GROUNDING_TOOLS = SM_GROUNDING_TOOLS | AM_GROUNDING_TOOLS | SUBSCRIPTION_GROUNDING_TOOLS | {"search_semantic_knowledge", "get_knowledge_by_key"}

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
        rag_enabled: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "intent_encoding"
        self.rag_enabled = rag_enabled
        self.compiler = IntentCompiler()
        self.initialize_agent_runtime(logger_color="\033[95m")
        self.tools = [
            search_sm_flow_targets,
            get_sm_ue_context,
            get_sm_ue_flow_catalog,
            get_am_policy_context,
            get_ue_slice_subscription,
            search_am_policy_targets,
        ]
        if self.rag_enabled:
            self.tools.extend([search_semantic_knowledge, get_knowledge_by_key])
        self.advisor_agent = self._create_advisor_agent(self.tools)
        self.last_failure_debug: Dict[str, Any] = {}

    def _create_advisor_agent(self, tools: List[Any]) -> Any:
        return self.create_json_agent(
            tools=tools,
            system_prompt=GroundingPromptBuilder().system_prompt(),
            response_model=GroundingDecision,
            max_iterations=14,
            max_calls_per_tool=3,
            tool_call_limits={
                "get_sm_ue_flow_catalog": 4,
                "search_sm_flow_targets": 4,
            },
            tool_result_limits={
                "get_sm_ue_flow_catalog": 16000,
                "get_sm_ue_context": 8000,
                "search_sm_flow_targets": 8000,
                "get_am_policy_context": 8000,
                "search_am_policy_targets": 8000,
                "get_ue_slice_subscription": 8000,
                "search_semantic_knowledge": 8000,
                "get_knowledge_by_key": 8000,
            },
            context_policy=ContextPolicy(
                default_tool_result_chars=8000,
                tool_result_char_limits={
                    "get_sm_ue_flow_catalog": 16000,
                    "get_sm_ue_context": 8000,
                    "search_sm_flow_targets": 8000,
                    "get_am_policy_context": 8000,
                    "search_am_policy_targets": 8000,
                    "get_ue_slice_subscription": 8000,
                    "search_semantic_knowledge": 8000,
                    "get_knowledge_by_key": 8000,
                },
                recent_tool_results_per_tool=1,
                tool_history_keep_limits={
                    "get_sm_ue_flow_catalog": 3,
                    "search_sm_flow_targets": 2,
                },
            ),
        )

    @classmethod
    def _filter_tools_for_domains(
        cls,
        tools: List[Any],
        requested_domains: List[str] | None,
    ) -> List[Any]:
        domains = {
            str(item or "").strip().lower()
            for item in (requested_domains or [])
            if str(item or "").strip()
        }
        if not domains:
            return list(tools)

        allowed = {"search_semantic_knowledge", "get_knowledge_by_key"}
        if "qos" in domains:
            allowed |= cls.SM_GROUNDING_TOOLS
            # Tool availability is a factual capability boundary, not an
            # operation classifier. IEA decides whether migration semantics
            # need this evidence; code later requires it only when IEA emits a
            # serving-slice change constraint.
            allowed |= cls.SUBSCRIPTION_GROUNDING_TOOLS
        if "mobility" in domains:
            allowed |= cls.AM_GROUNDING_TOOLS
        return [tool for tool in tools if str(getattr(tool, "name", "") or "").strip() in allowed]

    def handle_artifact(self, envelope: ArtifactEnvelope) -> GroundingDecision:
        payload = envelope.payload or {}
        return self.analyze_grounding_decision(
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
            artifact_type="GroundingDecisionRequest",
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
        grounding_decision: GroundingDecision,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="GroundingDecision",
            request_envelope=request_envelope,
            payload=grounding_decision,
        )

    def analyze_grounding_decision(
        self,
        user_input: str,
        context: str = "",
        conversation_messages: List[Dict[str, Any]] | None = None,
        *,
        session_id: str = "",
        snapshot_id: str = "",
        allow_user_interaction: bool = False,
        request_envelope: ArtifactEnvelope | None = None,
        trace_metadata: Dict[str, Any] | None = None,
    ) -> GroundingDecision:
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

        grounding_decision = self.analyze_intent(
            user_input,
            **analyze_kwargs,
            trace_metadata=trace_metadata,
        )
        self._cache_produced_result(
            request_envelope=request_envelope,
            grounding_decision=grounding_decision,
        )
        return grounding_decision

    def analyze_intent(
        self,
        user_input: str,
        context: str = "",
        *,
        session_id: str = "",
        snapshot_id: str = "",
        allow_user_interaction: bool = False,
        trace_metadata: Dict[str, Any] | None = None,
    ) -> GroundingDecision:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "iea_analyze_start")
        main_directives = self.compiler.extract_main_directives(context)
        self.last_failure_debug = {}

        try:
            evidence = self._extract_intent_evidence(
                user_input=user_input,
                snapshot_id=snapshot_id,
                main_directives=main_directives,
            )
            token_budget, token_counter = self._resolve_token_context()
            runtime_context = self.build_runtime_context(
                agent_name=self.agent_name,
                session_id=session_id,
                snapshot_id=snapshot_id,
                supi=evidence.supi or None,
                thread_id=session_id,
                allow_user_interaction=allow_user_interaction,
                token_budget=token_budget,
                token_counter=token_counter,
                trace_metadata=trace_metadata,
            )
            request_advisor_agent = self._create_advisor_agent(
                self._filter_tools_for_domains(
                    self.tools,
                    evidence.requested_domains,
                )
            )
            advisor_prompt = self._build_advisor_prompt(evidence=evidence, context=context)
            grounding_decision = None
            grounding_tools: List[str] = []
            validation_errors: List[str] = []
            advisor_validation_errors: List[str] = []
            invocation_error: str = ""
            for attempt_index in range(3):
                try:
                    advisor_invocation = self._invoke_intent_advisor(
                        prompt=advisor_prompt,
                        runtime_context=runtime_context,
                        advisor_agent=request_advisor_agent,
                    )
                except RuntimeError as exc:
                    invocation_error = str(exc)
                    if attempt_index == 2:
                        raise
                    # If the failed invocation produced tool results, extract them
                    # and merge into evidence so the retry prompt includes the data
                    # the LLM already collected — prevents burning tool quota re-querying.
                    raw_messages = list(getattr(exc, "_tool_output_messages", None) or [])
                    if raw_messages:
                        fake_result = {"messages": raw_messages}
                        evidence = self._refresh_intent_evidence_from_tool_results(
                            evidence=evidence,
                            advisor_result=fake_result,
                            main_directives=main_directives,
                            snapshot_id=snapshot_id,
                        )
                    log_event(self.logger, "iea_retry", attempt=attempt_index + 2, reason="invocation_error", error=invocation_error)
                    advisor_prompt = self._build_validation_retry_prompt(
                        base_prompt=self._build_advisor_prompt(evidence=evidence, context=context),
                        advisor_validation_errors=[],
                        grounding_validation_errors=[],
                        invocation_error=invocation_error,
                    )
                    continue
                advisor_result = advisor_invocation.advisor_result
                grounding_decision = coerce_structured_response(
                    advisor_result,
                    GroundingDecision,
                    error_message="IEA advisor returned no structured_response",
                )
                grounding_tools = extract_grounding_tool_names(advisor_result, self.GROUNDING_TOOLS)
                refreshed_evidence = self._refresh_intent_evidence_from_tool_results(
                    evidence=evidence,
                    advisor_result=advisor_result,
                    main_directives=main_directives,
                    snapshot_id=snapshot_id,
                )
                grounding_decision = self._attach_subscription_evidence(
                    grounding_decision=grounding_decision,
                    evidence=refreshed_evidence,
                )
                grounding_decision = self._materialize_catalog_facts(
                    grounding_decision=grounding_decision,
                    evidence=refreshed_evidence,
                )
                advisor_validation_errors, validation_errors, _ = validate_grounding_decision(
                    compiler=self.compiler,
                    evidence=refreshed_evidence,
                    grounding_decision=grounding_decision,
                    grounding_tools=grounding_tools,
                )
                self.last_failure_debug = {
                    "phase": "intent_encoding",
                    "attempt_index": attempt_index + 1,
                    "grounding_tools": list(grounding_tools or []),
                    "grounding_decision": grounding_decision.model_dump(mode="json"),
                    "advisor_validation_errors": list(advisor_validation_errors or []),
                    "grounding_validation_errors": list(validation_errors or []),
                    "evidence_snapshot": refreshed_evidence.model_dump(mode="json"),
                }
                evidence = refreshed_evidence
                if not advisor_validation_errors and not validation_errors:
                    break
                if attempt_index == 2:
                    if advisor_validation_errors:
                        raise RuntimeError("IEA advisor decision validation failed: " + "; ".join(advisor_validation_errors))
                    raise RuntimeError("IEA grounding validation failed: " + "; ".join(validation_errors))
                log_event(self.logger, "iea_retry", attempt=attempt_index + 2,
                          reason="validation_errors",
                          advisor_errors=advisor_validation_errors,
                          grounding_errors=validation_errors)
                advisor_prompt = self._build_validation_retry_prompt(
                    base_prompt=self._build_advisor_prompt(evidence=evidence, context=context),
                    advisor_validation_errors=advisor_validation_errors,
                    grounding_validation_errors=validation_errors,
                    invocation_error="",
                )
            if grounding_decision is None:
                raise RuntimeError("IEA advisor returned no GroundingDecision payload")
            _, _, validated_intent = validate_grounding_decision(
                compiler=self.compiler,
                evidence=evidence,
                grounding_decision=grounding_decision,
                grounding_tools=grounding_tools,
            )
            if validated_intent is None:
                raise RuntimeError("IEA GroundingDecision failed validation after retry loop")
            advisor_invocation.write_final_trace(
                grounding_decision=validated_intent,
                status="success",
            )
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
            self.last_failure_debug = {}
            return validated_intent
        except Exception as exc:
            if "advisor_invocation" in locals() and "grounding_decision" in locals():
                advisor_invocation.write_final_trace(
                    grounding_decision=grounding_decision,
                    status="error",
                    error=str(exc),
                )
            self.logger.error(f"Failed to analyze grounding decision: {exc}")
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    @staticmethod
    def _build_advisor_prompt(*, evidence: IntentEvidence, context: str) -> str:
        requested_domains = [str(item or "").strip() for item in (evidence.requested_domains or []) if str(item or "").strip()]
        domain_mode = ",".join(requested_domains) or "<empty>"
        qos_required = "qos" in requested_domains
        mobility_only = requested_domains == ["mobility"]
        domain_specific_rules: List[str] = [
            f"- Domain mode for this request: {domain_mode}.",
            "- Final answer must be exactly one raw JSON object with no markdown fence and no surrounding prose.",
            "- Main already selected the domain scope, semantic targets, and active stage. Return only grounded bindings and operation constraints for that stage.",
            "- `open_questions`, when needed, contains objects with owner_agent, question, blocking, and related_domains.",
        ]
        if qos_required:
            domain_specific_rules.extend(
                [
                    "- For each named QoS flow with a known SUPI, call get_sm_ue_flow_catalog before final JSON; when no exact candidate is already grounded, call search_sm_flow_targets first. Return a non-empty flows array.",
                    "- Mark a flow resolved only when its flow_id is in that catalog; otherwise preserve the named target as unresolved.",
                    "- A missing flow binding does not change Main's domain scope; preserve the named target as unresolved.",
                ]
            )
            domain_specific_rules.extend(
                [
                    "- Decide from the request and evidence whether a serving-S-NSSAI change is part of the operation. Only after flow grounding, if it is, call get_ue_slice_subscription once per target SUPI and return slice_migration_authorization.",
                    "- If you decide the operation is prioritization, deferment, congestion relief, or same-slice QoS tuning, do not call get_ue_slice_subscription and do not set require_slice_change=true.",
                    "- A target S-NSSAI may be used only when it is present in get_ue_slice_subscription.authorized_snssais. Do not invent authorization.",
                    "- If the requested target is not authorized and the user did not explicitly ask to add/change a subscription, block the migration and keep the serving slice; never emit an executable migration policy.",
                    "- Set subscription_change_required=true only when the user explicitly requests subscription provisioning or a subscription change.",
                ]
            )
            if evidence.candidate_flows:
                domain_specific_rules.extend(
                    [
                        "- candidate_flows contains a possible binding. Reuse an exact candidate after catalog confirmation; do not re-search it for reassurance.",
                    ]
                )
            elif str(evidence.explicit_flow_name or "").strip():
                domain_specific_rules.extend(
                    [
                        f"- No candidate exists for '{evidence.explicit_flow_name}'; search that exact target, then confirm it in the catalog.",
                    ]
                )
            explicit_target_names = [
                str(item.flow_name or "").strip()
                for item in (evidence.explicit_flow_targets or [])
                if str(item.flow_name or "").strip()
            ]
            if len(explicit_target_names) > 1:
                grounded_explicit_target_names = {
                    str(item.flow_name or "").strip()
                    for item in (evidence.candidate_flows or [])
                    if str(item.flow_name or "").strip() in explicit_target_names
                }
                unresolved_explicit_target_names = [
                    item for item in explicit_target_names
                    if item not in grounded_explicit_target_names
                ]
                domain_specific_rules.extend(
                    [
                        f"- Explicit QoS targets: {json.dumps(explicit_target_names, ensure_ascii=False)}. Return each as its exact resolved or unresolved entry; never substitute a neighbor.",
                    ]
                )
                if grounded_explicit_target_names and unresolved_explicit_target_names:
                    domain_specific_rules.extend(
                        [
                            f"- Already grounded: {json.dumps(sorted(grounded_explicit_target_names), ensure_ascii=False)}; unresolved: {json.dumps(unresolved_explicit_target_names, ensure_ascii=False)}. Return both sets.",
                        ]
                    )
        if mobility_only:
            domain_specific_rules.extend(
                [
                    "- This is mobility-only grounding. Final JSON must keep flows empty.",
                    "- Do not call any SM grounding tool: search_sm_flow_targets, get_sm_ue_context, or get_sm_ue_flow_catalog.",
                    "- Use only AM grounding if more evidence is needed.",
                ]
            )
        return (
            "User request:\n"
            f"{evidence.user_input}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(project_intent_evidence_for_prompt(evidence), ensure_ascii=False)}\n\n"
            "Coordinator context:\n"
            f"{context or 'N/A'}\n\n"
            f"{IEA_DYNAMIC_RULES.strip()}\n\n"
            "Task:\n"
            "- Resolve only choices that remain ambiguous; use already-grounded evidence directly.\n"
            "- Do not guess identifiers. An unresolved QoS target still appears in flows with its name and resolution_status='unresolved'.\n"
            f"{chr(10).join(domain_specific_rules)}\n"
            "- Return one GroundingDecision JSON object only."
        )

    @staticmethod
    def _build_validation_retry_prompt(
        *,
        base_prompt: str,
        advisor_validation_errors: List[str],
        grounding_validation_errors: List[str],
        invocation_error: str,
    ) -> str:
        return RetryPromptBuilder().build_grounding(
            base_prompt=base_prompt,
            advisor_validation_errors=advisor_validation_errors,
            grounding_validation_errors=grounding_validation_errors,
            invocation_error=invocation_error,
        )

    def _invoke_intent_advisor(
        self,
        *,
        prompt: str,
        runtime_context: Any,
        advisor_agent: Any | None = None,
    ) -> IntentAdvisorInvocation:
        self._pending_invoke_messages = [{"role": "user", "content": prompt}]
        base_trace_metadata = dict(getattr(runtime_context, "trace_metadata", {}) or {})
        active_advisor_agent = advisor_agent or self.advisor_agent
        context_components = measure_context_components(
            {
                "system_prompt": str(getattr(active_advisor_agent, "system_prompt", "") or ""),
                "dynamic_prompt": prompt,
            },
            token_counter=getattr(runtime_context, "token_counter", None),
        )
        log_event(self.logger, "iea_context_components", **context_components)
        invoke_payload = {
            "messages": self._pending_invoke_messages,
            "trace_write_mode": "manual",
            "trace_metadata": {
                **base_trace_metadata,
                "path_label": "advisor_path",
                "context_token_components": context_components,
            },
        }
        try:
            result = active_advisor_agent.invoke(invoke_payload, context=runtime_context)
        except Exception as exc:
            if isinstance(exc, ToolLoopExecutionError):
                failed_tool_call = exc.failed_tool_call or {}
                if failed_tool_call:
                    raise RuntimeError(
                        f"IEA tool call failed: {failed_tool_call.get('name') or '<unknown>'}: {exc}"
                    ) from exc
                message = str(exc)
                if "max iterations" in message.lower():
                    raise RuntimeError(f"IEA advisor did not converge to valid JSON: {message}") from exc
                # Preserve output_messages from ToolLoopExecutionError so the
                # retry loop can extract tool results and inject them into the
                # retry prompt — avoids burning tool quota re-querying the same data.
                invocation_error = RuntimeError(
                    f"IEA advisor invocation failed before structured output validation: {exc}"
                )
                invocation_error._tool_output_messages = list(exc.output_messages)
                raise invocation_error from exc
            raise RuntimeError(f"IEA advisor invocation failed before structured output validation: {exc}") from exc
        return IntentAdvisorInvocation(
            advisor_result=result,
            trace_agent=active_advisor_agent,
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
        normalized_input = str(user_input or "").strip()
        explicit_supis = extract_requested_supis(str(main_directives.get("supi") or ""), normalized_input)
        supi = str(main_directives.get("supi") or "").strip()
        if not supi and explicit_supis:
            supi = ", ".join(explicit_supis)
        return self.compiler.build_intent_evidence(
            user_input=normalized_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload={},
            catalog_evidence_observed=False,
            semantic_candidates=[],
            am_context_payload={},
            am_policy_candidates=[],
            subscription_payload={},
        )

    def _refresh_intent_evidence_from_tool_results(
        self,
        *,
        evidence: IntentEvidence,
        advisor_result: Dict[str, Any],
        main_directives: Dict[str, Any],
        snapshot_id: str,
    ) -> IntentEvidence:
        semantic_candidates: List[Dict[str, Any]] = list(evidence.semantic_candidates or [])
        catalog_payload: Dict[str, Any] = dict(evidence.catalog_payload or {})
        catalog_evidence_observed = bool(evidence.catalog_evidence_observed)
        am_context_payload: Dict[str, Any] = dict(evidence.am_context_payload or {})
        am_policy_candidates: List[Dict[str, Any]] = list(evidence.am_policy_candidates or [])
        subscription_payload: Dict[str, Any] = dict(evidence.subscription_payload or {})
        requested_domains = list(evidence.requested_domains or [])
        requested_supis = set(extract_requested_supis(evidence.supi, evidence.user_input))

        for entry in extract_grounding_tool_payloads(advisor_result=advisor_result):
            tool_name = str(entry.get("tool_name") or "").strip()
            call_args = dict(entry.get("call_args") or {})
            payload = dict(entry.get("payload") or {})
            if tool_name == "get_sm_ue_flow_catalog":
                payload_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                if not requested_supis or payload_supi in requested_supis:
                    catalog_evidence_observed = True
                    catalog_payload = merge_catalog_payloads(catalog_payload, payload)
            elif tool_name == "search_sm_flow_targets":
                semantic_candidates = merge_candidate_dicts(semantic_candidates, list(payload.get("candidates") or []))
            elif tool_name == "get_am_policy_context":
                payload_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                if not requested_supis or payload_supi in requested_supis:
                    am_context_payload = {**am_context_payload, **dict(payload)}
            elif tool_name == "search_am_policy_targets":
                am_policy_candidates = merge_candidate_dicts(am_policy_candidates, list(payload.get("candidates") or []))
            elif tool_name == "get_ue_slice_subscription":
                payload_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                if not requested_supis or payload_supi in requested_supis:
                    subscription_payload = {**subscription_payload, **payload}

        return self.compiler.build_intent_evidence(
            user_input=evidence.user_input,
            supi=evidence.supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            catalog_evidence_observed=catalog_evidence_observed,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
            subscription_payload=subscription_payload,
        )

    @staticmethod
    def _materialize_catalog_facts(
        *,
        grounding_decision: GroundingDecision,
        evidence: IntentEvidence,
    ) -> GroundingDecision:
        """Replace LLM-copied QoS baselines with the authoritative catalog facts."""
        catalog_by_flow_id = {
            str(item.get("flow_id") or "").strip(): item
            for item in (evidence.catalog_payload or {}).get("flow_catalog") or []
            if isinstance(item, dict) and str(item.get("flow_id") or "").strip()
        }
        if not catalog_by_flow_id:
            return grounding_decision

        enriched = grounding_decision.model_copy(deep=True)
        resolved_flows = []
        for flow in enriched.flows or []:
            is_resolved = str(flow.resolution_status or "resolved").strip().lower() == "resolved"
            catalog_entry = catalog_by_flow_id.get(str(flow.flow_id or "").strip()) if is_resolved else None
            if catalog_entry is None:
                resolved_flows.append(flow)
                continue
            # This projection is a code-owned fact boundary. The LLM chooses
            # only the catalog identity; it does not transcribe SLA facts.
            resolved_flows.append(IntentEvidenceBuilder._build_flow_selector_from_catalog(catalog_entry))

        enriched.flows = resolved_flows
        return enriched

    @staticmethod
    def _attach_subscription_evidence(
        *,
        grounding_decision: GroundingDecision,
        evidence: IntentEvidence,
    ) -> GroundingDecision:
        subscription = dict(evidence.subscription_summary or {})
        if not subscription:
            return grounding_decision
        enriched = grounding_decision.model_copy(deep=True)
        mobility_targets = dict(enriched.grounding_evidence.grounded_mobility_targets or {})
        mobility_targets["subscription_entitlement"] = subscription
        enriched.grounding_evidence.grounded_mobility_targets = mobility_targets
        sources = dict(enriched.grounding_evidence.evidence_sources or {})
        source_list = list(sources.get("ue_slice_subscription") or [])
        authority = str(subscription.get("authority") or "postgresql_ue_context").strip()
        if authority and authority not in source_list:
            source_list.append(authority)
        sources["ue_slice_subscription"] = source_list
        enriched.grounding_evidence.evidence_sources = sources
        return enriched

