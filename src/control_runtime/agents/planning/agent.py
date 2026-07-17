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
from ...context.projectors import project_collaboration_context_for_prompt, project_grounding_decision_for_prompt
from ...context.observability import measure_context_components
from ...context.prompts import PlanningPromptBuilder
from shared.logging import log_event, log_timing

from .compiler import OptimizationStrategyCompiler
from .response_models import OsaAdvisorOutput
from .tool_result_adapter import extract_planning_tool_evidence
from .tools import build_request_tools


def _build_lean_grounding_decision(grounding_decision: Any) -> dict:
    return project_grounding_decision_for_prompt(grounding_decision)


def _planning_supi(planning_request: PlanningRequest) -> str:
    main_supi = str(planning_request.context.shared_context.main_intent.supi or "").strip()
    if main_supi:
        return main_supi
    return next(
        (
            str(flow.supi or "").strip()
            for flow in (planning_request.grounding_decision.flows or [])
            if str(flow.supi or "").strip()
        ),
        "",
    )


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

    @staticmethod
    def _active_domain_names(planning_request: PlanningRequest) -> set[str]:
        domain_values = planning_request.context.shared_context.main_intent.requested_domains or []
        return {
            str(item.value if hasattr(item, "value") else item or "").strip().lower()
            for item in domain_values
            if str(item.value if hasattr(item, "value") else item or "").strip()
        }

    def _include_knowledge_tools(self, planning_request: PlanningRequest) -> bool:
        if not self.rag_enabled:
            return False
        # Standards retrieval is exceptional: local optimizer, UE context, and
        # IEA evidence own normal planning facts. Do not make RAG callable just
        # because mobility participates in the plan.
        question_text = " ".join(
            " ".join(
                [
                    str(question.owner_agent or ""),
                    str(question.question or ""),
                ]
            )
            for question in (planning_request.grounding_decision.open_questions or [])
        ).lower()
        return any(
            marker in question_text
            for marker in ("3gpp", "ts 23.", "ts 29.", "standard", "specification", "标准", "规范")
        )

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
        grounding_decision = effective_request.grounding_decision
        normalized_grounding_decision = _build_lean_grounding_decision(grounding_decision)
        coordination_context = project_collaboration_context_for_prompt(effective_request.context)

        total_start = time.perf_counter()
        log_event(self.logger, "osa_generate_start")
        advisor_trace = None
        self.last_failure_debug = {}
        try:
            planning_evidence = self.compiler.build_planning_evidence(effective_request)
            prompt_planning_tools, _ = build_request_tools(effective_request)
            include_knowledge_tools = self._include_knowledge_tools(effective_request)
            available_tool_names = [
                str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
                for tool in [*(self._RAG_TOOLS if include_knowledge_tools else []), *prompt_planning_tools]
                if str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
            ]
            prompt_builder = PlanningPromptBuilder()
            base_prompt = prompt_builder.advisor_user_prompt(
                normalized_user_intent=normalized_grounding_decision,
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
            retry_tool_evidence: dict[str, Any] = {}
            disable_knowledge_tools = False
            for attempt_index in range(3):
                try:
                    advisor_output, advisor_result, advisor_trace = self._invoke_strategy_advisor(
                        planning_request=effective_request,
                        prompt=current_prompt,
                        trace_metadata=trace_metadata,
                        cached_tool_evidence=retry_tool_evidence,
                        disable_knowledge_tools=disable_knowledge_tools,
                    )
                except RuntimeError as exc:
                    invocation_error = str(exc)
                    self.last_failure_debug = {
                        "phase": "optimization_strategy",
                        "attempt_index": attempt_index + 1,
                        "invocation_error": invocation_error,
                    }
                    retry_tool_evidence = self._cached_tool_evidence_for_retry()
                    if "get_knowledge_by_key exceeded" in invocation_error or "search_semantic_knowledge exceeded" in invocation_error:
                        disable_knowledge_tools = True
                    if attempt_index == 2:
                        raise
                    log_event(self.logger, "osa_retry", attempt=attempt_index + 2, reason="invocation_error", error=invocation_error)
                    current_prompt = prompt_builder.validation_retry_prompt(
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
                advisor_output = self._enforce_entitlement_limited_execution_boundary(
                    advisor_output=advisor_output,
                    planning_request=effective_request,
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
                retry_tool_evidence = self._cached_tool_evidence_for_retry()
                log_event(self.logger, "osa_retry", attempt=attempt_index + 2,
                          reason="contract_errors", errors=contract_errors)
                current_prompt = prompt_builder.validation_retry_prompt(
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
        domains = OptimizationStrategyAgent._active_domain_names(planning_request)
        if "qos" in domains and not planning_request.grounding_decision.flows:
            raise ValueError(
                "Main selected QoS for this stage but IEA returned no grounded flows; request upstream reground"
            )
        # IEA is invoked with one Main-selected stage and returns bindings for
        # that stage only. No second stage model or flow filtering belongs in
        # OSA; it consumes that handoff as-is.
        return planning_request

    @staticmethod
    def _enforce_entitlement_limited_execution_boundary(
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
    ) -> OsaAdvisorOutput:
        """Remove policy types prohibited by IEA's entitlement evidence.

        OSA still chooses the QoS strategy. This is only the final execution
        boundary: AM/URSP cannot provision or bypass a target S-NSSAI when IEA
        has grounded that entitlement as unavailable.
        """
        decision = str(
            planning_request.grounding_decision.slice_migration_authorization.decision or ""
        ).strip()
        if decision not in {
            "blocked_by_subscription_entitlement",
            "blocked_requires_subscription_provisioning",
            "evidence_missing",
        }:
            return advisor_output

        bounded = advisor_output.model_copy(deep=True)
        bounded.am_policy = None
        bounded.ursp_policies = []
        if bounded.sm_policies or bounded.partial_policies:
            bounded.planning_status = "partial_plan"
        if not bounded.blocked_targets:
            bounded.blocked_targets = ["requested target slice migration"]
        if not bounded.missing_evidence:
            bounded.missing_evidence = ["authorized target S-NSSAI after subscription provisioning"]
        if not bounded.planner_conflicts:
            bounded.planner_conflicts = [
                "Subscription entitlement blocks the target slice change; only current-slice QoS delivery is executable."
            ]
        return bounded

    def _invoke_strategy_advisor(
        self,
        *,
        planning_request: PlanningRequest,
        prompt: str,
        trace_metadata: dict[str, Any] | None = None,
        cached_tool_evidence: dict[str, Any] | None = None,
        disable_knowledge_tools: bool = False,
    ) -> tuple[OsaAdvisorOutput, dict[str, Any], dict[str, Any]]:
        token_budget, token_counter = self._resolve_token_context()
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=planning_request.context.session_id,
            snapshot_id=planning_request.context.snapshot_id,
            supi=_planning_supi(planning_request),
            thread_id=planning_request.context.session_id,
            token_budget=token_budget,
            token_counter=token_counter,
            trace_metadata=trace_metadata,
        )
        planning_tools, tools_cache = build_request_tools(
            planning_request,
            cached_tool_evidence=cached_tool_evidence,
        )
        self._tools_cache = tools_cache
        include_knowledge_tools = (
            not disable_knowledge_tools and self._include_knowledge_tools(planning_request)
        )
        advisor_agent = self.create_json_agent(
            tools=[
                *(self._RAG_TOOLS if include_knowledge_tools else []),
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
        context_components = measure_context_components(
            {
                "system_prompt": str(getattr(advisor_agent, "system_prompt", "") or ""),
                "dynamic_prompt": prompt,
            },
            token_counter=getattr(runtime_context, "token_counter", None),
        )
        log_event(self.logger, "osa_context_components", **context_components)
        invoke_payload = {
            "messages": messages,
            "trace_write_mode": "manual",
            "trace_metadata": {
                **(trace_metadata or {}),
                "path_label": "strategy_advisor",
                "context_token_components": context_components,
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

