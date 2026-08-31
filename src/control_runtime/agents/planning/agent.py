from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from shared.contracts import (
    AgentInvocation,
    AgentMessage,
    AgentRunResult,
    ExecutionScope,
    TraceMetadata,
    TraceTarget,
)
from shared.runtime import ArtifactEnvelope, RuntimePersistence
from shared.runtime import ContextPolicy
from shared.agents import (
    BaseAgent,
    RetryContext,
    RetryPhase,
    RetryPolicy,
    StageAttempt,
    StageRunner,
    coerce_structured_response,
    extract_grounding_tool_names,
)
from shared.runtime import ToolLoopExecutionError
from shared.runtime import ArtifactWorkerMixin
from application_runtime.persistence import create_runtime_persistence
from ...tool_capabilities import CONTROL_RUNTIME_CAPABILITY_ALIASES
from .contracts import PlanningRequest, PolicyPlanDraft
from ...context.projectors import project_collaboration_context_for_prompt, project_grounding_decision_for_prompt
from ...context.observability import measure_context_components
from ...context.prompts.planning import (
    build_planning_validation_retry_prompt,
    build_planning_user_prompt,
    render_planning_system_prompt,
)
from shared.logging import log_event, log_timing

from .compiler import OptimizationStrategyCompiler
from .evidence import extract_planning_tool_evidence
from .response_models import OsaAdvisorOutput
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


@dataclass(slots=True)
class _OsaAdvisorStageState:
    """Inputs, response, and diagnostics retained while one advisor stage retries."""

    current_prompt: str
    retry_tool_evidence: dict[str, Any] = field(default_factory=dict)
    invocation_error: str = ""
    advisor_output: OsaAdvisorOutput | None = None
    advisor_result: AgentRunResult | None = None
    advisor_trace: dict[str, Any] | None = None
    planning_tool_evidence: dict[str, Any] = field(default_factory=dict)
    grounding_tools: list[str] = field(default_factory=list)
    contract_errors: list[str] = field(default_factory=list)
    trace_recorded: bool = False


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
    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
        rag_enabled: bool = True,
        *,
        persistence: RuntimePersistence | None = None,
    ) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "optimization_strategy"
        self.rag_enabled = rag_enabled
        self.compiler = OptimizationStrategyCompiler()
        runtime_persistence = persistence if persistence is not None else create_runtime_persistence()
        self.initialize_agent_runtime(logger_color="\033[94m", persistence=runtime_persistence)
        self.last_failure_debug: dict[str, Any] = {}

    @staticmethod
    def _rag_tools() -> list[Any]:
        from knowledge_runtime.retrieval.raw import get_knowledge_by_key, search_semantic_knowledge

        return [search_semantic_knowledge, get_knowledge_by_key]

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
            payload_schema_name=PlanningRequest.__name__,
            payload=planning_request,
            scope=ExecutionScope(
                session_id=planning_request.context.session_id,
                thread_id=planning_request.context.session_id,
                snapshot_id=planning_request.context.snapshot_id or None,
                round_index=planning_request.context.round_index,
                supi=_planning_supi(planning_request),
            ),
        )

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        policy_plan: PolicyPlanDraft,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="PolicyPlanDraft",
            payload_schema_name=PolicyPlanDraft.__name__,
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
        advisor_trace: dict[str, Any] | None = None
        advisor_output: OsaAdvisorOutput | None = None
        advisor_result: AgentRunResult | None = None
        stage_state: _OsaAdvisorStageState | None = None
        self.last_failure_debug = {}
        try:
            planning_evidence = self.compiler.build_planning_evidence(effective_request)
            prompt_planning_tools, _ = build_request_tools(effective_request)
            include_knowledge_tools = self._include_knowledge_tools(effective_request)
            available_tool_names = [
                str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
                for tool in [*(self._rag_tools() if include_knowledge_tools else []), *prompt_planning_tools]
                if str(getattr(tool, "name", "") or getattr(tool, "__name__", "")).strip()
            ]
            base_prompt = build_planning_user_prompt(
                normalized_user_intent=normalized_grounding_decision,
                coordination_context=coordination_context,
                planning_evidence=planning_evidence,
                available_tool_names=available_tool_names,
            )
            stage_state = _OsaAdvisorStageState(current_prompt=base_prompt)
            completed_stage = self._run_advisor_stage(
                planning_request=effective_request,
                base_prompt=base_prompt,
                trace_metadata=trace_metadata,
                state=stage_state,
            )
            advisor_output = completed_stage.advisor_output
            advisor_result = completed_stage.advisor_result
            advisor_trace = completed_stage.advisor_trace
            planning_tool_evidence = completed_stage.planning_tool_evidence
            grounding_tools = completed_stage.grounding_tools

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
            )
            self._cache_produced_result(
                request_envelope=request_envelope,
                policy_plan=final_output,
            )
            self.last_failure_debug = {}
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="success")
            return final_output
        except Exception as exc:
            failed_advisor_trace = advisor_trace
            failed_advisor_output = advisor_output
            if stage_state is not None:
                if failed_advisor_trace is None:
                    failed_advisor_trace = stage_state.advisor_trace
                if failed_advisor_output is None:
                    failed_advisor_output = stage_state.advisor_output
            if failed_advisor_trace is not None and not stage_state.trace_recorded:
                self._write_advisor_trace(
                    advisor_trace=failed_advisor_trace,
                    advisor_output=failed_advisor_output,
                    compiler_output=None,
                    error=str(exc),
                )
            self.logger.exception(f"Failed to generate optimization strategy: {exc}")
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    def _run_advisor_stage(
        self,
        *,
        planning_request: PlanningRequest,
        base_prompt: str,
        trace_metadata: dict[str, Any] | None,
        state: _OsaAdvisorStageState,
    ) -> _OsaAdvisorStageState:
        """Run OSA's existing advisor/repair/validation lifecycle once or with retries."""

        def invoke_advisor(_: StageAttempt) -> _OsaAdvisorStageState:
            state.advisor_output = None
            state.advisor_result = None
            state.advisor_trace = None
            state.trace_recorded = False
            advisor_output, advisor_result, advisor_trace = self._invoke_strategy_advisor(
                planning_request=planning_request,
                prompt=state.current_prompt,
                trace_metadata=trace_metadata,
                cached_tool_evidence=state.retry_tool_evidence,
            )
            state.advisor_output = advisor_output
            state.advisor_result = advisor_result
            state.advisor_trace = advisor_trace
            return state

        def repair_advisor(
            stage_value: _OsaAdvisorStageState,
            _: StageAttempt,
        ) -> _OsaAdvisorStageState:
            stage_value.planning_tool_evidence = extract_planning_tool_evidence(
                advisor_result=stage_value.advisor_result,
                tool_payload_cache=getattr(self, "_tools_cache", None),
            )
            stage_value.grounding_tools = extract_grounding_tool_names(
                stage_value.advisor_result,
                self.GROUNDING_TOOLS,
            )
            stage_value.advisor_output = self._enforce_entitlement_limited_execution_boundary(
                advisor_output=stage_value.advisor_output,
                planning_request=planning_request,
            )
            return stage_value

        def validate_advisor(
            stage_value: _OsaAdvisorStageState,
            attempt: StageAttempt,
        ) -> list[str]:
            contract_errors = self.compiler.validate_advisor_output(
                advisor_output=stage_value.advisor_output,
                planning_request=planning_request,
                grounding_tools=stage_value.grounding_tools,
                planning_tool_evidence=stage_value.planning_tool_evidence,
            )
            state.contract_errors = contract_errors
            self.last_failure_debug = {
                "phase": "optimization_strategy",
                "attempt_index": attempt.attempt,
                "grounding_tools": list(stage_value.grounding_tools or []),
                "advisor_output": stage_value.advisor_output.model_dump(mode="json"),
                "contract_errors": list(contract_errors or []),
                "planning_tool_evidence": stage_value.planning_tool_evidence,
            }
            return contract_errors

        def record_advisor_failure(context: RetryContext) -> None:
            if context.phase is RetryPhase.VALIDATION and state.advisor_trace is not None:
                self._write_advisor_trace(
                    advisor_trace=state.advisor_trace,
                    advisor_output=state.advisor_output,
                    compiler_output=None,
                    error=str(context.error),
                )
                state.trace_recorded = True
            if context.phase is not RetryPhase.INVOKE or not isinstance(context.error, RuntimeError):
                return
            state.invocation_error = str(context.error)
            self.last_failure_debug = {
                "phase": "optimization_strategy",
                "attempt_index": context.attempt,
                "invocation_error": state.invocation_error,
            }
            state.retry_tool_evidence = self._cached_tool_evidence_for_retry()

        def prepare_advisor_retry(context: RetryContext) -> None:
            if context.phase is RetryPhase.INVOKE:
                retry_issues = [str(context.error)]
                log_event(
                    self.logger,
                    "osa_retry",
                    attempt=context.attempt + 1,
                    reason="invocation_error",
                    error=retry_issues[0],
                )
            else:
                retry_issues = list(context.issues)
                state.retry_tool_evidence = self._cached_tool_evidence_for_retry()
                log_event(
                    self.logger,
                    "osa_retry",
                    attempt=context.attempt + 1,
                    reason="contract_errors",
                    errors=retry_issues,
                )
            state.current_prompt = build_planning_validation_retry_prompt(
                base_prompt=base_prompt,
                issues=retry_issues,
                cached_planning_evidence=self._cached_tool_evidence_for_retry(),
            )

        runner: StageRunner[_OsaAdvisorStageState] = StageRunner(
            retry_policy=RetryPolicy(
                max_attempts=3,
                retryable_exceptions=(RuntimeError,),
                retryable_phases=frozenset({RetryPhase.INVOKE, RetryPhase.VALIDATION}),
            )
        )
        return runner.run(
            invoke=invoke_advisor,
            repair=repair_advisor,
            validate=validate_advisor,
            validation_error=lambda issues: RuntimeError(
                "OSA advisor contract validation failed: " + "; ".join(issues)
            ),
            on_failure=record_advisor_failure,
            on_retry=prepare_advisor_retry,
        )

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
    ) -> tuple[OsaAdvisorOutput, dict[str, Any], dict[str, Any]]:
        token_budget, token_counter = self._resolve_token_context()
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            scope=ExecutionScope(
                session_id=planning_request.context.session_id,
                thread_id=planning_request.context.session_id,
                snapshot_id=planning_request.context.snapshot_id or None,
                round_index=planning_request.context.round_index,
                scenario_id=str((trace_metadata or {}).get("scenario_id") or "").strip() or None,
                scenario_tags=list((trace_metadata or {}).get("scenario_tags") or []),
                supi=_planning_supi(planning_request),
            ),
            token_budget=token_budget,
            token_counter=token_counter,
            trace_attributes=dict(trace_metadata or {}),
        )
        planning_tools, tools_cache = build_request_tools(
            planning_request,
            cached_tool_evidence=cached_tool_evidence,
        )
        self._tools_cache = tools_cache
        include_knowledge_tools = self._include_knowledge_tools(planning_request)
        advisor_agent = self.create_json_agent(
            tools=[
                *(self._rag_tools() if include_knowledge_tools else []),
                *planning_tools,
            ],
            system_prompt=render_planning_system_prompt(),
            prompt_name="optimization_strategy_advisor",
            prompt_version="2.0.0",
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
            capability_aliases=CONTROL_RUNTIME_CAPABILITY_ALIASES,
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
        invocation = AgentInvocation(
            messages=[AgentMessage(role="user", content=prompt)],
            metadata=TraceMetadata(
                path_label="strategy_advisor",
                prompt_name="optimization_strategy_advisor",
                prompt_version="2.0.0",
                attributes={
                    **{
                        key: value
                        for key, value in (trace_metadata or {}).items()
                        if key not in {"path_label", "scenario_id", "scenario_tags"}
                    },
                    "context_token_components": context_components,
                },
            ),
        )
        self._pending_invoke_messages = messages
        try:
            result = advisor_agent.invoke(
                invocation,
                context=runtime_context,
                defer_trace=True,
            )
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
            "invocation": invocation,
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
        error: str | None = None,
    ) -> None:
        derived_targets = []
        if compiler_output is not None and error is None:
            derived_targets.append(
                TraceTarget(
                    kind="compiled_output",
                    producer="compiler",
                    schema_name=PolicyPlanDraft.__name__,
                    output_schema=PolicyPlanDraft.model_json_schema(),
                    value=compiler_output,
                    sft_eligible=False,
                )
            )
        advisor_trace["agent"].write_trace(
            invocation=advisor_trace["invocation"],
            context=advisor_trace["runtime_context"],
            result=advisor_trace["result"],
            derived_targets=derived_targets,
            error=error,
        )


__all__ = ["OptimizationStrategyAgent"]

