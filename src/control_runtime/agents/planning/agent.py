from __future__ import annotations

import copy
import time
from typing import Any

from shared.runtime import ArtifactEnvelope
from shared.runtime import ContextPolicy
from shared.agents import BaseAgent, coerce_structured_response, extract_grounding_tool_names
from knowledge_runtime.retrieval.raw import get_knowledge_by_key, search_semantic_knowledge
from shared.runtime import ToolLoopExecutionError
from shared.runtime import ArtifactWorkerMixin
from ...domain.collaboration import PlanningRequest
from ...domain.policy_plan import PolicyPlanDraft
from ...context.projectors import project_collaboration_context_for_prompt, project_operation_intent_for_prompt
from ...context.prompts import PlanningPromptBuilder, RetryPromptBuilder
from shared.logging import log_event, log_timing

from .compiler import OptimizationStrategyCompiler
from .planning_validation import normalize_app_id as _normalize_app_id
from .response_models import OsaAdvisorOutput
from .tool_result_adapter import extract_planning_tool_evidence
from .tools import build_request_tools


def _build_lean_operation_intent(operation_intent: Any) -> dict:
    return project_operation_intent_for_prompt(operation_intent)


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
    _RAG_TOOLS = [search_semantic_knowledge, get_knowledge_by_key]

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
        rag_enabled: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "optimization_strategy"
        self.rag_enabled = rag_enabled
        self.compiler = OptimizationStrategyCompiler()
        self.initialize_agent_runtime(logger_color="\033[94m")
        self.last_failure_debug: dict[str, Any] = {}

    def handle_artifact(self, envelope: ArtifactEnvelope) -> PolicyPlanDraft:
        planning_request = PlanningRequest.model_validate(envelope.payload)
        return self._generate_strategy(planning_request=planning_request, request_envelope=envelope)

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

    def generate_strategy(self, planning_request: PlanningRequest, *, trace_metadata: dict[str, Any] | None = None) -> PolicyPlanDraft:
        self.ensure_worker_runtime_initialized()
        if not isinstance(planning_request, PlanningRequest):
            raise TypeError("generate_strategy expects a PlanningRequest instance")
        request_envelope = self._cache_received_request(planning_request)
        return self._generate_strategy(
            planning_request=planning_request,
            request_envelope=request_envelope,
            trace_metadata=trace_metadata,
        )

    def _generate_strategy(
        self,
        *,
        planning_request: PlanningRequest,
        request_envelope: ArtifactEnvelope,
        trace_metadata: dict[str, Any] | None = None,
    ) -> PolicyPlanDraft:
        try:
            effective_request = self._effective_planning_request(planning_request)
        except ValueError as exc:
            return self.compiler.build_upstream_reground_plan(
                planning_request=planning_request,
                reason=str(exc),
            )
        operation_intent = effective_request.operation_intent
        normalized_user_intent = _build_lean_operation_intent(operation_intent)
        normalized_user_intent["app_id"] = _normalize_app_id(normalized_user_intent.get("app_id"))
        coordination_context = project_collaboration_context_for_prompt(effective_request.context)

        total_start = time.perf_counter()
        log_event(self.logger, "osa_generate_start")
        advisor_trace = None
        self.last_failure_debug = {}
        try:
            planning_evidence = self.compiler.build_planning_evidence(effective_request)
            prompt_planning_tools, _ = build_request_tools(effective_request)
            available_tool_names = [
                str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
                for tool in [*([] if not self.rag_enabled else self._RAG_TOOLS), *prompt_planning_tools]
                if str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
            ]
            prompt_builder = PlanningPromptBuilder()
            base_prompt = prompt_builder.advisor_user_prompt(
                normalized_user_intent=normalized_user_intent,
                coordination_context=coordination_context,
                planning_evidence=planning_evidence,
                available_tool_names=available_tool_names,
            )
            current_prompt = base_prompt
            advisor_output = None
            advisor_result = None
            planning_tool_evidence: dict[str, Any] = {}
            grounding_tools: list[str] = []
            contract_errors: list[str] = []
            invocation_error = ""
            for attempt_index in range(3):
                try:
                    advisor_output, advisor_result, advisor_trace = self._invoke_strategy_advisor(
                        planning_request=effective_request,
                        prompt=current_prompt,
                        trace_metadata=trace_metadata,
                    )
                except RuntimeError as exc:
                    invocation_error = str(exc)
                    self.last_failure_debug = {
                        "phase": "optimization_strategy",
                        "attempt_index": attempt_index + 1,
                        "invocation_error": invocation_error,
                    }
                    # Merge tool results from the failed invocation into the
                    # tools_cache so retry prompt can include them.
                    raw_messages = list(getattr(exc, "_tool_output_messages", None) or [])
                    if raw_messages:
                        tools_cache = getattr(self, "_tools_cache", {}) if hasattr(self, "_tools_cache") else {}
                        cached_preview = tools_cache.get("latest_optimizer_preview") if isinstance(tools_cache, dict) else None
                        if cached_preview is not None:
                            tools_cache["latest_optimizer_preview"] = dict(cached_preview)
                    if attempt_index == 2:
                        raise
                    log_event(self.logger, "osa_retry", attempt=attempt_index + 2, reason="invocation_error", error=invocation_error)
                    current_prompt = RetryPromptBuilder().build_osa(
                        base_prompt=base_prompt,
                        issues=[invocation_error],
                        cached_planning_evidence=self._cached_tool_evidence_for_retry(),
                    )
                    continue

                planning_tool_evidence = extract_planning_tool_evidence(
                    advisor_result=advisor_result,
                    tool_payload_cache=getattr(self, "_tools_cache", None),
                )
                grounding_tools = extract_grounding_tool_names(
                    advisor_result,
                    self.GROUNDING_TOOLS,
                )
                contract_errors = self.compiler.validate_advisor_output(
                    advisor_output=advisor_output,
                    planning_request=effective_request,
                    grounding_tools=grounding_tools,
                    planning_tool_evidence=planning_tool_evidence,
                )
                self.last_failure_debug = {
                    "phase": "optimization_strategy",
                    "attempt_index": attempt_index + 1,
                    "grounding_tools": list(grounding_tools or []),
                    "advisor_output": advisor_output.model_dump(mode="json"),
                    "contract_errors": list(contract_errors or []),
                    "planning_tool_evidence": planning_tool_evidence,
                }
                if not contract_errors:
                    break
                if attempt_index == 2:
                    raise RuntimeError("OSA advisor contract validation failed: " + "; ".join(contract_errors))
                log_event(self.logger, "osa_retry", attempt=attempt_index + 2,
                          reason="contract_errors", errors=contract_errors)
                current_prompt = RetryPromptBuilder().build_osa(
                    base_prompt=base_prompt,
                    issues=list(contract_errors),
                    cached_planning_evidence=self._cached_tool_evidence_for_retry(),
                )

            if advisor_output is None or advisor_result is None:
                raise RuntimeError("OSA advisor returned no structured_response")

            final_output = self.compiler.assemble_policy_plan(
                advisor_output=advisor_output,
                planning_request=effective_request,
                planning_tool_evidence=planning_tool_evidence,
            )
            self._write_advisor_trace(
                advisor_trace=advisor_trace,
                advisor_output=advisor_output,
                compiler_output=final_output.model_dump(mode="json"),
                status="success",
            )
            self._cache_produced_result(
                request_envelope=request_envelope,
                policy_plan=final_output,
            )
            self.last_failure_debug = {}
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="success")
            return final_output
        except Exception as exc:
            if advisor_trace is not None:
                self._write_advisor_trace(
                    advisor_trace=advisor_trace,
                    advisor_output=advisor_output if "advisor_output" in locals() else None,
                    compiler_output=None,
                    status="error",
                    error=str(exc),
                )
            self.logger.exception(f"Failed to generate optimization strategy: {exc}")
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    @staticmethod
    def _effective_planning_request(planning_request: PlanningRequest) -> PlanningRequest:
        semantics = planning_request.operation_intent.control_semantics
        stages = semantics.stages or []
        if not stages:
            return planning_request
        current_stage_index = max(1, int(semantics.current_stage or 1))
        current_stage = next(
            (stage for stage in stages if int(stage.stage_index or 0) == current_stage_index),
            stages[0],
        )
        active_flow_ids = {
            str(flow_id or "").strip()
            for flow_id in (current_stage.active_flow_ids or [])
            if str(flow_id or "").strip()
        }
        # Mobility-only requests have no QoS flows — active_flow_ids is expected
        # to be empty because flows are populated by IEA only for QoS targets.
        is_mobility_only = (
            planning_request.operation_intent.requested_domains
            == ["mobility"]
        )
        if not active_flow_ids and is_mobility_only:
            return planning_request
        if not active_flow_ids:
            raise ValueError(
                "current control stage has no grounded active_flow_ids; OSA must request upstream reground instead of expanding to all flows"
            )
        filtered_intent = planning_request.operation_intent.model_copy(deep=True)
        filtered_intent.flows = [
            flow for flow in filtered_intent.flows
            if str(flow.flow_id or "").strip() in active_flow_ids
        ]
        filtered_intent.qos_target_envelopes = [
            envelope for envelope in filtered_intent.qos_target_envelopes
            if str(envelope.flow_id or "").strip() in active_flow_ids
        ]
        return planning_request.model_copy(update={"operation_intent": filtered_intent}, deep=True)

    def _invoke_strategy_advisor(
        self,
        *,
        planning_request: PlanningRequest,
        prompt: str,
        trace_metadata: dict[str, Any] | None = None,
    ) -> tuple[OsaAdvisorOutput, dict[str, Any], dict[str, Any]]:
        token_budget, token_counter = self._resolve_token_context()
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=planning_request.context.session_id,
            snapshot_id=planning_request.context.snapshot_id,
            supi=planning_request.operation_intent.supi,
            thread_id=planning_request.context.session_id,
            token_budget=token_budget,
            token_counter=token_counter,
            trace_metadata=trace_metadata,
        )
        planning_tools, tools_cache = build_request_tools(planning_request)
        self._tools_cache = tools_cache
        advisor_agent = self.create_json_agent(
            tools=[
                *([] if not self.rag_enabled else self._RAG_TOOLS),
                *planning_tools,
            ],
            system_prompt=PlanningPromptBuilder().system_prompt(),
            response_model=OsaAdvisorOutput,
            max_iterations=12,
            max_calls_per_tool=2,
            tool_call_limits={
                "get_knowledge_by_key": 1,
                "search_semantic_knowledge": 1,
            },
            tool_result_limits={
                "preview_qos_optimizer": 32000,
                "fetch_qos_network_status": 32000,
                "inspect_mobility_ue_policies": 8000,
                "search_semantic_knowledge": 8000,
                "get_knowledge_by_key": 8000,
            },
            context_policy=ContextPolicy(
                default_tool_result_chars=8000,
                tool_result_char_limits={
                    "preview_qos_optimizer": 32000,
                    "fetch_qos_network_status": 32000,
                    "inspect_mobility_ue_policies": 8000,
                    "search_semantic_knowledge": 8000,
                    "get_knowledge_by_key": 8000,
                },
                recent_tool_results_per_tool=1,
            ),
        )
        messages = [{"role": "user", "content": prompt}]
        invoke_payload = {
            "messages": messages,
            "trace_write_mode": "manual",
            "trace_metadata": {
                **(trace_metadata or {}),
                "path_label": "strategy_advisor",
            },
        }
        self._pending_invoke_messages = messages
        try:
            result = advisor_agent.invoke(invoke_payload, context=runtime_context)
        except Exception as exc:
            if isinstance(exc, ToolLoopExecutionError):
                failed_tool_call = exc.failed_tool_call or {}
                if failed_tool_call:
                    raise RuntimeError(
                        f"OSA advisor tool call failed: {failed_tool_call.get('name') or '<unknown>'}: {exc}"
                    ) from exc
                if "max iterations" in str(exc).lower():
                    invocation_error = RuntimeError(f"OSA advisor did not converge: {exc}")
                    invocation_error._tool_output_messages = list(exc.output_messages)
                    raise invocation_error from exc
                error_text = str(exc)
                if "Input should be a valid dictionary or instance of OsaAdvisorOutput" in error_text:
                    raise RuntimeError(
                        "OSA advisor returned an invalid planning payload: expected one OsaAdvisorOutput JSON object, "
                        "but received a bare array or non-object payload"
                    ) from exc
                if "Extra inputs are not permitted" in error_text and "flow_id" in error_text:
                    raise RuntimeError(
                        "OSA advisor returned a policy item without the required top-level planning object: "
                        "a bare SmPolicySpec appeared instead of OsaAdvisorOutput.sm_policies"
                    ) from exc
                # Preserve output_messages for tool-limit errors so the retry loop
                # can inject cached evidence into the prompt.
                invocation_error = RuntimeError(f"OSA advisor invocation failed: {exc}")
                invocation_error._tool_output_messages = list(exc.output_messages)
                raise invocation_error from exc
            raise
        advisor_output = coerce_structured_response(
            result,
            OsaAdvisorOutput,
            error_message="OSA advisor returned no structured_response",
        )
        return advisor_output, result, {
            "agent": advisor_agent,
            "payload": invoke_payload,
            "runtime_context": runtime_context,
            "result": result,
        }

    def _cached_tool_evidence_for_retry(self) -> dict[str, Any]:
        tools_cache = getattr(self, "_tools_cache", None)
        if not isinstance(tools_cache, dict):
            return {}
        return {
            str(key): copy.deepcopy(value)
            for key, value in tools_cache.items()
            if value not in (None, "", [], {})
        }

    @staticmethod
    def _write_advisor_trace(
        *,
        advisor_trace: dict[str, Any],
        advisor_output: OsaAdvisorOutput | None,
        compiler_output: Any,
        status: str,
        error: str | None = None,
    ) -> None:
        payload = copy.deepcopy(advisor_trace["payload"])
        metadata = dict(payload.get("trace_metadata") or {})
        metadata["advisor_decision"] = (
            advisor_output.model_dump(mode="json")
            if advisor_output is not None
            else None
        )
        metadata["compiler_output"] = compiler_output
        payload["trace_metadata"] = metadata
        advisor_trace["agent"].write_trace(
            payload=payload,
            context=advisor_trace["runtime_context"],
            result=advisor_trace["result"],
            status=status,
            error=error,
            structured_response_override=compiler_output,
        )


__all__ = ["OptimizationStrategyAgent"]

