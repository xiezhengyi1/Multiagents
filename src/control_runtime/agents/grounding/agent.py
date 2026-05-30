from __future__ import annotations

import json
import re
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
    search_am_policy_targets,
    search_sm_flow_targets,
)
from ...domain.policy_plan import OperationIntent
from ..common import project_intent_evidence_for_prompt, validate_and_compile_intent
from shared.logging import log_event, log_timing

from .compiler import IntentCompiler
from .common import extract_requested_supis, merge_candidate_dicts, merge_catalog_payloads
from .contracts import IntentAdvisorDecision, IntentEvidence
from .prompts import IEA_DYNAMIC_RULES, IEA_SYSTEM_PROMPT
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
            search_am_policy_targets,
        ]
        if self.rag_enabled:
            self.tools.extend([search_semantic_knowledge, get_knowledge_by_key])
        self.advisor_agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=IEA_SYSTEM_PROMPT,
            response_model=IntentAdvisorDecision,
            max_iterations=14,
            tool_result_limits={
                "get_sm_ue_flow_catalog": 8000,
                "get_sm_ue_context": 4000,
                "search_sm_flow_targets": 4000,
                "get_am_policy_context": 4000,
                "search_am_policy_targets": 4000,
                "search_semantic_knowledge": 4000,
                "get_knowledge_by_key": 4000,
            },
            context_policy=ContextPolicy(
                default_tool_result_chars=4000,
                default_tool_result_tokens=1000,
                tool_result_char_limits={
                    "get_sm_ue_flow_catalog": 8000,
                    "get_sm_ue_context": 4000,
                    "search_sm_flow_targets": 4000,
                    "get_am_policy_context": 4000,
                    "search_am_policy_targets": 4000,
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
        self.last_failure_debug: Dict[str, Any] = {}

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
        trace_metadata: Dict[str, Any] | None = None,
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
            trace_metadata=trace_metadata,
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
        trace_metadata: Dict[str, Any] | None = None,
    ) -> OperationIntent:
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
            advisor_prompt = self._build_advisor_prompt(evidence=evidence, context=context)
            advisor_decision = None
            grounding_tools: List[str] = []
            validation_errors: List[str] = []
            advisor_validation_errors: List[str] = []
            invocation_error: str = ""
            for attempt_index in range(3):
                try:
                    advisor_invocation = self._invoke_intent_advisor(
                        prompt=advisor_prompt,
                        runtime_context=runtime_context,
                    )
                except RuntimeError as exc:
                    invocation_error = str(exc)
                    if attempt_index == 2:
                        raise
                    log_event(self.logger, "iea_retry", attempt=attempt_index + 2, reason="invocation_error", error=invocation_error)
                    advisor_prompt = self._build_validation_retry_prompt(
                        base_prompt=self._build_advisor_prompt(evidence=evidence, context=context),
                        advisor_validation_errors=[],
                        grounding_validation_errors=[],
                        invocation_error=invocation_error,
                    )
                    continue
                advisor_result = advisor_invocation.advisor_result
                advisor_decision = coerce_structured_response(
                    advisor_result,
                    IntentAdvisorDecision,
                    error_message="IEA advisor returned no structured_response",
                )
                grounding_tools = extract_grounding_tool_names(advisor_result, self.GROUNDING_TOOLS)
                refreshed_evidence = self._refresh_intent_evidence_from_tool_results(
                    evidence=evidence,
                    advisor_result=advisor_result,
                    main_directives=main_directives,
                    snapshot_id=snapshot_id,
                )
                advisor_validation_errors, validation_errors, _ = validate_and_compile_intent(
                    compiler=self.compiler,
                    evidence=refreshed_evidence,
                    decision=advisor_decision,
                    grounding_tools=grounding_tools,
                    user_input=user_input,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    main_directives=main_directives,
                )
                self.last_failure_debug = {
                    "phase": "intent_encoding",
                    "attempt_index": attempt_index + 1,
                    "grounding_tools": list(grounding_tools or []),
                    "advisor_decision": advisor_decision.model_dump(mode="json"),
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
            if advisor_decision is None:
                raise RuntimeError("IEA advisor returned no decision payload")
            _, _, compiled = validate_and_compile_intent(
                compiler=self.compiler,
                evidence=evidence,
                decision=advisor_decision,
                grounding_tools=grounding_tools,
                user_input=user_input,
                session_id=session_id,
                snapshot_id=snapshot_id,
                main_directives=main_directives,
            )
            if compiled is None:
                raise RuntimeError("IEA could not compile OperationIntent after validation")
            advisor_invocation.write_final_trace(
                advisor_decision=advisor_decision,
                operation_intent=compiled,
                status="success",
            )
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
            self.last_failure_debug = {}
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

    @staticmethod
    def _build_advisor_prompt(*, evidence: IntentEvidence, context: str) -> str:
        requested_domains = [str(item or "").strip() for item in (evidence.requested_domains or []) if str(item or "").strip()]
        domain_mode = ",".join(requested_domains) or "<empty>"
        qos_required = "qos" in requested_domains
        mobility_only = requested_domains == ["mobility"]
        domain_specific_rules: List[str] = [
            f"- Domain mode for this request: {domain_mode}.",
            "- Final answer must be exactly one raw JSON object with no markdown fence and no surrounding prose.",
            "- `domain_resolution` must be one scalar string value, never an object.",
        ]
        if qos_required:
            domain_specific_rules.extend(
                [
                    "- This request includes QoS grounding. Final JSON must contain a non-empty flows array.",
                    "- Every resolved QoS flow must include grounded app_id and grounded flow_id.",
                    "- If the current evidence does not already ground the QoS target, keep using SM grounding tools until flows is populated or the target is explicitly unresolved.",
                    "- Do not stop at selected_app_id / selected_flow_id alone; the grounded binding must appear inside flows.",
                ]
            )
            if evidence.candidate_flows:
                domain_specific_rules.extend(
                    [
                        "- Current evidence already contains candidate_flows with grounded identifiers (flow_id, app_id). Use those identifiers in flows.",
                        "- Do not call search_sm_flow_targets or get_sm_ue_context again to reconfirm an already unique exact candidate.",
                        "- Do not leave flows empty when candidate_flows is already non-empty.",
                        "- IMPORTANT: candidate_flows only carries identifiers, NOT SLA parameters (latency, bandwidth). You must call get_sm_ue_flow_catalog to fetch the SLA baseline before finalizing.",
                    ]
                )
            elif str(evidence.explicit_flow_name or "").strip():
                domain_specific_rules.extend(
                    [
                        f"- No grounded candidate_flows currently exist for the explicit QoS target '{evidence.explicit_flow_name}'.",
                        "- Before final JSON, call search_sm_flow_targets for that explicit flow target.",
                        "- After search returns a grounded exact match, copy that app_id + flow_id into flows immediately.",
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
                        "- This request names multiple QoS flow targets.",
                        f"- Explicit QoS targets in this request: {json.dumps(explicit_target_names, ensure_ascii=False)}.",
                        "- Every resolved flow in `flows` must correspond to one of those explicit targets and be grounded by catalog/search evidence for that exact target.",
                        "- When a resolved flow corresponds to an explicit target, keep `flows[].name` equal to that explicit flow name.",
                        "- If candidate_flows does not already cover all explicit targets, search unresolved explicit targets individually before finalizing.",
                        "- If any explicit target remains ungrounded, do not substitute a nearby flow name.",
                    ]
                )
                if grounded_explicit_target_names and unresolved_explicit_target_names:
                    domain_specific_rules.extend(
                        [
                            f"- Evidence already grounds these explicit QoS targets: {json.dumps(sorted(grounded_explicit_target_names), ensure_ascii=False)}.",
                            f"- These explicit QoS targets are still unresolved in current evidence: {json.dumps(unresolved_explicit_target_names, ensure_ascii=False)}.",
                            "- The next answer must return a mixed flows array: resolved entries for grounded explicit targets, plus unresolved entries for still-unresolved explicit targets.",
                            "- Never leave flows empty when at least one explicit target is already grounded.",
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
            "- Resolve only the semantic choices that remain ambiguous.\n"
            "- Use tools only when the structured evidence does not already ground the required target.\n"
            "- You may revise Main's requested domain boundary when grounding evidence proves it is too narrow, too wide, or cannot be confirmed.\n"
            "- If you revise the domain boundary, populate grounded_requested_domains, domain_resolution, domain_revision_needed, and domain_revision_rationale explicitly.\n"
            "- For every QoS flow with resolution_status='resolved', include grounded flow_id and app_id in the final JSON.\n"
            "- If a QoS target is not fully grounded to flow_id + app_id, do not mark it resolved.\n"
            "- If the structured evidence already contains the grounded answer, finalize from that evidence without extra tool calls.\n"
            f"{chr(10).join(domain_specific_rules)}\n"
            "- Return one IntentAdvisorDecision JSON object only."
        )

    @staticmethod
    def _build_validation_retry_prompt(
        *,
        base_prompt: str,
        advisor_validation_errors: List[str],
        grounding_validation_errors: List[str],
        invocation_error: str,
    ) -> str:
        issues: List[str] = []
        if invocation_error:
            issues.append(invocation_error)
        if advisor_validation_errors:
            issues.extend(advisor_validation_errors)
        if grounding_validation_errors:
            issues.extend(grounding_validation_errors)
        repair_rules: List[str] = [
            "Return one corrected IntentAdvisorDecision JSON object only.",
            "Do not guess missing identifiers, and do not rely on downstream compilation to fill them.",
            "Return raw JSON only, with no markdown fence and no prose outside the JSON object.",
            "`domain_resolution` must be a scalar string, not an object.",
        ]
        joined = " | ".join(issues)
        if "QoS advisor decision must include grounded target flows." in joined:
            repair_rules.extend([
                "This retry is specifically failing because your previous JSON omitted flows.",
                "For the next answer, flows must be non-empty.",
                "If you already have a grounded QoS candidate in evidence, copy it into flows and finalize.",
                "If only some explicit QoS targets are grounded, return resolved entries for those grounded targets and unresolved entries for the remaining explicit targets.",
                "If you still do not have a grounded QoS candidate, do not return an empty object; call the required SM grounding tool and then return either a resolved or explicitly unresolved flow entry.",
                "Do not spend another tool call to reconfirm a single exact candidate that is already grounded in evidence.",
            ])
        if "domain_resolution must be confirmed, narrowed, widened, or cannot_confirm" in joined:
            repair_rules.extend([
                "Set `domain_resolution` to exactly one of: confirmed, narrowed, widened, cannot_confirm.",
                "Do not output a nested object under `domain_resolution`.",
            ])
        if "cannot_confirm domain resolution requires domain_revision_rationale" in joined:
            repair_rules.extend([
                "If you set `domain_resolution` to `cannot_confirm`, you must include a non-empty `domain_revision_rationale`.",
                "If you can confirm the domain boundary from evidence, use `confirmed` instead.",
            ])
        if (
            "explicitly named QoS flow '" in joined
            and (
                "was not grounded by catalog/search evidence" in joined
                or "must appear in advisor decision flows as resolved or unresolved" in joined
            )
        ):
            repair_rules.extend([
                "For each explicitly named QoS flow, either ground it via catalog/search evidence or leave it unresolved.",
                "When a flow is resolved, set `flows[].name` to the explicit flow name that the resolved binding satisfies.",
                "Do not return a resolved flow binding for any name that is missing from catalog/search evidence.",
            ])
        if "mobility-only intent must not call SM grounding tools" in joined:
            repair_rules.extend([
                "This retry is mobility-only.",
                "Do not call search_sm_flow_targets, get_sm_ue_context, or get_sm_ue_flow_catalog.",
            ])
        if "QoS-only intent must not call AM grounding tools" in joined:
            repair_rules.extend([
                "This retry is QoS-only.",
                "Do not call get_am_policy_context or search_am_policy_targets.",
            ])

        import re
        cleaned = re.sub(
            r'\n\nRetry feedback \(attempt \d+\).*$',
            '',
            base_prompt,
            flags=re.DOTALL,
        )
        cleaned = re.sub(
            r'\n\nYour previous attempt failed validation.*$',
            '',
            cleaned,
            flags=re.DOTALL,
        )

        return (
            f"{cleaned}\n\n"
            "Retry feedback:\n"
            + "\n".join(f"- {rule}" for rule in repair_rules)
            + "\n\nValidation errors:\n- "
            + "\n- ".join(issues)
        )

    def _invoke_intent_advisor(
        self,
        *,
        prompt: str,
        runtime_context: Any,
    ) -> IntentAdvisorInvocation:
        self._pending_invoke_messages = [{"role": "user", "content": prompt}]
        base_trace_metadata = dict(getattr(runtime_context, "trace_metadata", {}) or {})
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
            if isinstance(exc, ToolLoopExecutionError):
                failed_tool_call = exc.failed_tool_call or {}
                if failed_tool_call:
                    raise RuntimeError(
                        f"IEA tool call failed: {failed_tool_call.get('name') or '<unknown>'}: {exc}"
                    ) from exc
                message = str(exc)
                if "max iterations" in message.lower():
                    raise RuntimeError(f"IEA advisor did not converge to valid JSON: {message}") from exc
            raise RuntimeError(f"IEA advisor invocation failed before structured output validation: {exc}") from exc
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
            semantic_candidates=[],
            am_context_payload={},
            am_policy_candidates=[],
        )

    def _refresh_intent_evidence_from_tool_results(
        self,
        *,
        evidence: IntentEvidence,
        advisor_result: Dict[str, Any],
        main_directives: Dict[str, Any],
        snapshot_id: str,
    ) -> IntentEvidence:
        semantic_candidates: List[Dict[str, Any]] = []
        catalog_payload: Dict[str, Any] = {}
        am_context_payload: Dict[str, Any] = {}
        am_policy_candidates: List[Dict[str, Any]] = []
        requested_domains = list(evidence.requested_domains or [])
        requested_supis = set(extract_requested_supis(evidence.supi, evidence.user_input))

        for entry in extract_grounding_tool_payloads(advisor_result=advisor_result):
            tool_name = str(entry.get("tool_name") or "").strip()
            call_args = dict(entry.get("call_args") or {})
            payload = dict(entry.get("payload") or {})
            if tool_name == "get_sm_ue_flow_catalog" and self.compiler.uses_sm_grounding(requested_domains):
                payload_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                if not requested_supis or payload_supi in requested_supis:
                    catalog_payload = merge_catalog_payloads(catalog_payload, payload)
            elif tool_name == "search_sm_flow_targets" and self.compiler.uses_sm_grounding(requested_domains):
                semantic_candidates = merge_candidate_dicts(semantic_candidates, list(payload.get("candidates") or []))
            elif tool_name == "get_am_policy_context" and self.compiler.uses_am_grounding(requested_domains):
                payload_supi = str(call_args.get("supi") or payload.get("supi") or evidence.supi or "").strip()
                if not requested_supis or payload_supi in requested_supis:
                    am_context_payload = dict(payload)
            elif tool_name == "search_am_policy_targets" and self.compiler.uses_am_grounding(requested_domains):
                am_policy_candidates = merge_candidate_dicts(am_policy_candidates, list(payload.get("candidates") or []))

        return self.compiler.build_intent_evidence(
            user_input=evidence.user_input,
            supi=evidence.supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )

