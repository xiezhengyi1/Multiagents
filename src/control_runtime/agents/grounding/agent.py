from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from pydantic import ValidationError

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
from shared.runtime import ToolLoopExecutionError
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
from shared.runtime import ArtifactWorkerMixin
from application_runtime.persistence import create_runtime_persistence
from ...tool_capabilities import CONTROL_RUNTIME_CAPABILITY_ALIASES
from ...integrations.pcf import (
    get_am_policy_context,
    get_sm_ue_flow_catalog,
    get_ue_slice_subscription,
    search_am_policy_targets,
    search_sm_flow_targets,
)
from ...context.projectors import project_intent_evidence_for_prompt
from ...context.observability import measure_context_components
from .contracts import (
    FlowSelector,
    GroundingAdvisorOutput,
    GroundingDecision,
    GroundingEvidenceBundle,
    IntentEncodingDirectives,
    IntentEvidence,
    QosOperationConstraint,
    SliceMigrationAuthorization,
)
from ...context.prompts.grounding import (
    build_grounding_validation_retry_prompt,
    render_grounding_system_prompt,
)
from shared.logging import log_event, log_timing

from .evidence import (
    IntentEvidenceBuilder,
    extract_grounding_tool_payloads,
    extract_requested_supis,
    merge_candidate_dicts,
    merge_catalog_payloads,
)
from .validator import IntentGroundingValidator, validate_grounding


@dataclass
class IntentAdvisorInvocation:
    advisor_result: AgentRunResult
    trace_agent: Any
    invocation: AgentInvocation
    runtime_context: Any

    def write_final_trace(
        self,
        *,
        grounding_decision: GroundingDecision | None,
        error: str | None = None,
    ) -> None:
        derived_targets = []
        if grounding_decision is not None and error is None:
            derived_targets.append(
                TraceTarget(
                    kind="validated_output",
                    producer="validator",
                    schema_name=GroundingDecision.__name__,
                    output_schema=GroundingDecision.model_json_schema(),
                    value=grounding_decision.model_dump(mode="json"),
                )
            )
        self.trace_agent.write_trace(
            invocation=self.invocation,
            context=self.runtime_context,
            result=self.advisor_result,
            derived_targets=derived_targets,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class _IeaInvocationFailure:
    """A retryable transport/tool-loop failure from one advisor invocation."""

    error: RuntimeError


@dataclass(frozen=True, slots=True)
class _IeaAdvisorContractFailure:
    """A retryable malformed structured advisor response."""

    error: RuntimeError | ValidationError
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IeaValidationOutcome:
    """Keep decision and grounding validation failures distinct for retry/exit."""

    advisor_errors: tuple[str, ...]
    grounding_errors: tuple[str, ...]

    @property
    def issues(self) -> tuple[str, ...]:
        return (*self.advisor_errors, *self.grounding_errors)

    def terminal_error(self) -> RuntimeError:
        if self.advisor_errors:
            return RuntimeError(
                "IEA advisor decision validation failed: " + "; ".join(self.advisor_errors)
            )
        return RuntimeError(
            "IEA grounding validation failed: " + "; ".join(self.grounding_errors)
        )


@dataclass(slots=True)
class _IeaAdvisorStageState:
    """Mutable IEA advisor lifecycle state retained across bounded retries."""

    current_prompt: str
    evidence: IntentEvidence
    main_directives: Dict[str, Any]
    snapshot_id: str
    context: str
    runtime_context: Any
    advisor_agent: Any
    advisor_invocation: IntentAdvisorInvocation | None = None
    refreshed_evidence: IntentEvidence | None = None
    advisor_output: GroundingAdvisorOutput | None = None
    grounding_decision: GroundingDecision | None = None
    grounding_tools: List[str] = field(default_factory=list)
    invocation_failure: _IeaInvocationFailure | None = None
    advisor_contract_failure: _IeaAdvisorContractFailure | None = None
    validation_outcome: _IeaValidationOutcome | None = None
    trace_recorded: bool = False


class IntentEncodingAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "intent_encoding"
    SM_GROUNDING_TOOLS = {"search_sm_flow_targets", "get_sm_ue_flow_catalog"}
    AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}
    SUBSCRIPTION_GROUNDING_TOOLS = {"get_ue_slice_subscription"}
    GROUNDING_TOOLS = SM_GROUNDING_TOOLS | AM_GROUNDING_TOOLS | SUBSCRIPTION_GROUNDING_TOOLS

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
        *,
        persistence: RuntimePersistence | None = None,
    ) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "intent_encoding"
        self.evidence_builder = IntentEvidenceBuilder()
        self.validator = IntentGroundingValidator()
        runtime_persistence = persistence if persistence is not None else create_runtime_persistence()
        self.initialize_agent_runtime(logger_color="\033[95m", persistence=runtime_persistence)
        self.tools = [
            search_sm_flow_targets,
            get_sm_ue_flow_catalog,
            get_am_policy_context,
            get_ue_slice_subscription,
            search_am_policy_targets,
        ]
        self.advisor_agent = self._create_advisor_agent(self.tools)
        self.last_failure_debug: Dict[str, Any] = {}

    def _create_advisor_agent(self, tools: List[Any]) -> Any:
        return self.create_json_agent(
            tools=tools,
            system_prompt=render_grounding_system_prompt(),
            prompt_name="intent_grounding_advisor",
            prompt_version="2.0.0",
            response_model=GroundingAdvisorOutput,
            max_iterations=14,
            max_calls_per_tool=3,
            forbid_duplicate_tool_calls=True,
            tool_call_limits={
                "get_sm_ue_flow_catalog": 4,
                "search_sm_flow_targets": 4,
            },
            context_policy=ContextPolicy(
                default_tool_result_chars=8000,
                tool_result_char_limits={
                    "get_sm_ue_flow_catalog": 16000,
                    "search_sm_flow_targets": 8000,
                    "get_am_policy_context": 8000,
                    "search_am_policy_targets": 8000,
                    "get_ue_slice_subscription": 8000,
                },
            ),
            capability_aliases=CONTROL_RUNTIME_CAPABILITY_ALIASES,
        )

    @classmethod
    def _filter_tools_for_domains(
        cls,
        tools: List[Any],
        requested_domains: List[str] | None,
        *,
        include_subscription: bool = True,
    ) -> List[Any]:
        domains = {
            str(item or "").strip().lower()
            for item in (requested_domains or [])
            if str(item or "").strip()
        }
        if not domains:
            return list(tools)

        allowed: set[str] = set()
        if "qos" in domains:
            allowed |= cls.SM_GROUNDING_TOOLS
            if include_subscription:
                allowed |= cls.SUBSCRIPTION_GROUNDING_TOOLS
        if "mobility" in domains:
            allowed |= cls.AM_GROUNDING_TOOLS
        return [tool for tool in tools if str(getattr(tool, "name", "") or "").strip() in allowed]

    @staticmethod
    def _should_offer_subscription_tool(*, evidence: IntentEvidence) -> bool:
        """Expose entitlement lookup only for Main's typed migration decision.

        User prose and coordinator free text are deliberately excluded: Main
        owns the semantic classification and must set the explicit boolean.
        """
        domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or [])}
        return (
            domains == {"qos", "mobility"}
            and evidence.serving_slice_change_requested is True
            and not bool(evidence.subscription_summary)
        )

    def handle_artifact(self, envelope: ArtifactEnvelope) -> GroundingDecision:
        payload = envelope.payload or {}
        return self.analyze_grounding_decision(
            user_input=str(payload.get("user_input") or ""),
            context=str(payload.get("context") or ""),
            conversation_messages=payload.get("messages"),
            allow_user_interaction=bool(payload.get("allow_user_interaction", False)),
            session_id=envelope.scope.session_id,
            snapshot_id=envelope.scope.snapshot_id or "",
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
            payload_schema_name="GroundingDecisionRequest",
            payload={
                "user_input": str(user_input),
                "context": str(context or ""),
                "messages": list(conversation_messages or []),
                "allow_user_interaction": bool(allow_user_interaction),
            },
            scope=ExecutionScope(
                session_id=session_id,
                thread_id=session_id,
                snapshot_id=snapshot_id or None,
            ),
        )

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        grounding_decision: GroundingDecision,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="GroundingDecision",
            payload_schema_name=GroundingDecision.__name__,
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
        main_directives = IntentEncodingDirectives.from_context(context).model_dump()
        self.last_failure_debug = {}
        stage_state: _IeaAdvisorStageState | None = None

        try:
            evidence = self._extract_intent_evidence(
                user_input=user_input,
                snapshot_id=snapshot_id,
                main_directives=main_directives,
            )
            token_budget, token_counter = self._resolve_token_context()
            runtime_context = self.build_runtime_context(
                agent_name=self.agent_name,
                scope=ExecutionScope(
                    session_id=session_id,
                    thread_id=session_id,
                    snapshot_id=snapshot_id or None,
                    scenario_id=str((trace_metadata or {}).get("scenario_id") or "").strip() or None,
                    scenario_tags=list((trace_metadata or {}).get("scenario_tags") or []),
                    supi=evidence.supi or None,
                ),
                allow_user_interaction=allow_user_interaction,
                token_budget=token_budget,
                token_counter=token_counter,
                trace_attributes=dict(trace_metadata or {}),
            )
            request_advisor_agent = self._create_advisor_agent(
                self._filter_tools_for_domains(
                    self.tools,
                    evidence.requested_domains,
                    include_subscription=self._should_offer_subscription_tool(
                        evidence=evidence,
                    ),
                )
            )
            stage_state = _IeaAdvisorStageState(
                current_prompt=self._build_advisor_prompt(evidence=evidence, context=context),
                evidence=evidence,
                main_directives=main_directives,
                snapshot_id=snapshot_id,
                context=context,
                runtime_context=runtime_context,
                advisor_agent=request_advisor_agent,
            )
            completed_stage = self._run_advisor_stage(state=stage_state)
            grounding_decision = completed_stage.grounding_decision
            advisor_invocation = completed_stage.advisor_invocation
            grounding_tools = completed_stage.grounding_tools
            evidence = completed_stage.evidence
            if grounding_decision is None or advisor_invocation is None:
                raise RuntimeError("IEA advisor returned no GroundingDecision payload")
            _, _, validated_intent = validate_grounding(
                validator=self.validator,
                evidence=evidence,
                grounding_decision=grounding_decision,
                grounding_tools=grounding_tools,
            )
            if validated_intent is None:
                raise RuntimeError("IEA GroundingDecision failed validation after retry loop")
            advisor_invocation.write_final_trace(
                grounding_decision=validated_intent,
            )
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
            self.last_failure_debug = {}
            return validated_intent
        except Exception as exc:
            if (
                stage_state is not None
                and stage_state.advisor_invocation is not None
                and not stage_state.trace_recorded
            ):
                stage_state.advisor_invocation.write_final_trace(
                    grounding_decision=None,
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
        requested_domains = [
            str(item or "").strip()
            for item in (evidence.requested_domains or [])
            if str(item or "").strip()
        ]
        domain_mode = ",".join(requested_domains) or "<empty>"
        qos_required = "qos" in requested_domains
        mobility_only = requested_domains == ["mobility"]
        subscription_eligible = IntentEncodingAgent._should_offer_subscription_tool(
            evidence=evidence,
        )
        explicit_target_names = [
            str(item.flow_name or "").strip()
            for item in (evidence.explicit_flow_targets or [])
            if str(item.flow_name or "").strip()
        ]
        candidate_names = {
            str(item.flow_name or "").strip()
            for item in (evidence.candidate_flows or [])
            if str(item.flow_name or "").strip()
        }
        missing_target_names = [
            name for name in explicit_target_names if name not in candidate_names
        ]
        domain_specific_rules: List[str] = [
            f"- Domain mode for this request: {domain_mode}.",
            "- Final answer must be exactly one raw JSON object with no markdown fence and no surrounding prose.",
            "- Main already selected the domain scope, semantic targets, and active stage. Return only grounded bindings and operation constraints for that stage.",
            "- `open_questions`, when needed, contains objects with owner_agent, question, blocking, and related_domains.",
        ]
        if qos_required:
            domain_specific_rules.extend(
                [
                    "- Binding and feasibility are separate: every named QoS target gets exactly one flows entry even when migration is blocked or no target slice is known.",
                    "- An exact catalog/search match is resolved; an absent match is {app_id:'',flow_id:'',name,resolution_status:'unresolved'}. Runtime attaches the Main-owned SUPI. Never use flows:[] for a QoS target.",
                    "- Runtime fills SLA, current-slice, traffic, and provenance facts after parsing. Output only the compact flow binding fields shown by the schema.",
                ]
            )
            if not evidence.catalog_evidence_observed:
                domain_specific_rules.extend(
                    [
                        "- For every known target SUPI that has no catalog evidence yet, call get_sm_ue_flow_catalog once before final JSON.",
                    ]
                )
            elif evidence.candidate_flows and not missing_target_names:
                domain_specific_rules.extend(
                    [
                        "- Catalog evidence already contains the exact binding. Reuse candidate_flows and do not call get_sm_ue_flow_catalog or search_sm_flow_targets again.",
                    ]
                )
            elif evidence.search_evidence_observed:
                domain_specific_rules.extend(
                    [
                        "- Catalog/search evidence has already been collected and did not bind this target. Do not call either tool again; return the named flow as unresolved.",
                    ]
                )
            elif missing_target_names:
                domain_specific_rules.extend(
                    [
                        f"- Catalog has no exact binding for {json.dumps(missing_target_names, ensure_ascii=False)}. Call search_sm_flow_targets once per missing exact target, then resolve or return unresolved.",
                    ]
                )
            elif str(evidence.explicit_app_name or "").strip():
                domain_specific_rules.extend(
                    [
                        f"- The app '{evidence.explicit_app_name}' is known but its requested flow is generic. Call search_sm_flow_targets once with that app and SUPI; if it returns no unique candidate, return an unresolved named flow.",
                    ]
                )
            else:
                domain_specific_rules.extend(
                    [
                        "- No exact flow binding is present after catalog lookup. Call search_sm_flow_targets once for the explicit target; if it has no exact result, return unresolved.",
                    ]
                )
            if explicit_target_names:
                domain_specific_rules.extend(
                    [
                        f"- Explicit QoS targets: {json.dumps(explicit_target_names, ensure_ascii=False)}. Return each as its exact resolved or unresolved entry; never substitute a neighbor.",
                    ]
                )
            if evidence.subscription_summary:
                domain_specific_rules.extend(
                    [
                        "- Subscription entitlement evidence is already present. Do not call get_ue_slice_subscription again; runtime copies its authority and authorized S-NSSAI list into the handoff.",
                    ]
                )
            elif not subscription_eligible:
                domain_specific_rules.extend(
                    [
                        "- Main did not set the typed serving_slice_change_requested flag together with qos+mobility domains. This is same-slice QoS work: set require_slice_change=false, decision=not_applicable, and do not request subscription evidence. The subscription tool is intentionally unavailable.",
                    ]
                )
            else:
                domain_specific_rules.extend(
                    [
                        "- Main supplied the typed serving_slice_change_requested flag with qos+mobility domains. Only after flow grounding, call get_ue_slice_subscription exactly once for each target SUPI. Reuse that UE-level result for every flow and comparison; a different rationale never permits another call.",
                    ]
                )
            domain_specific_rules.extend(
                [
                    "- For same-slice QoS tuning, set require_slice_change=false and do not request subscription evidence.",
                    "- For a required slice change, decision is exactly one of migration_authorized, blocked_by_subscription_entitlement, blocked_requires_subscription_provisioning, or evidence_missing; never write migration_blocked.",
                    "- If no different authorized S-NSSAI exists and the user did not ask to change a subscription, use blocked_by_subscription_entitlement with subscription_change_required=false; do not invent a target slice.",
                    "- Set subscription_change_required=true only when the user explicitly requests subscription provisioning or a subscription change.",
                ]
            )
            if evidence.search_evidence_observed and missing_target_names:
                domain_specific_rules.extend(
                    [
                        "- The required search already returned no binding. Its correct final form is unresolved, not another search or an empty flows array.",
                    ]
                )
        if mobility_only:
            domain_specific_rules.extend(
                [
                    "- This is mobility-only grounding. Final JSON must keep flows empty.",
                    "- Do not call any SM grounding tool: search_sm_flow_targets or get_sm_ue_flow_catalog.",
                    "- Use only AM grounding if more evidence is needed.",
                ]
            )
        # Keep the mandatory next action at the end of the dynamic prompt.  This
        # is a rule for the advisor, not an automatic tool invocation: other
        # request types can still finish from their own evidence contract.
        if qos_required and not evidence.catalog_evidence_observed:
            domain_specific_rules.append(
                "- FINAL TOOL GATE: do not return JSON yet. Your next action must be one get_sm_ue_flow_catalog call for each known target SUPI."
            )
        elif qos_required and missing_target_names and not evidence.search_evidence_observed:
            domain_specific_rules.append(
                "- FINAL TOOL GATE: do not return JSON yet. Your next action must be one search_sm_flow_targets call for each missing exact target."
            )
        domain_specific_rules.append(
            "- FINAL JSON TYPE GATE: `open_questions[].blocking` is JSON true/false (never a string); operation_type is only qos_reallocation or slice_migration."
        )
        return (
            "User request:\n"
            f"{evidence.user_input}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(project_intent_evidence_for_prompt(evidence), ensure_ascii=False)}\n\n"
            "Coordinator context:\n"
            f"{context or 'N/A'}\n\n"
            "Task:\n"
            "- Resolve only choices that remain ambiguous; use already-grounded evidence directly.\n"
            "- Do not guess identifiers. An unresolved QoS target still appears in flows with its name and resolution_status='unresolved'.\n"
            f"{chr(10).join(domain_specific_rules)}\n"
            "- Return one raw IEA decision JSON object only."
        )

    def _run_advisor_stage(
        self,
        *,
        state: _IeaAdvisorStageState,
    ) -> _IeaAdvisorStageState:
        """Run IEA's advisor parse/repair/validation lifecycle with old retry semantics."""

        retry_policy = RetryPolicy(
            max_attempts=3,
            retryable_exceptions_by_phase=(
                (RetryPhase.INVOKE, (RuntimeError,)),
                (RetryPhase.PARSE, (RuntimeError, ValidationError)),
                (RetryPhase.VALIDATION, (RuntimeError,)),
            ),
        )

        def invoke_advisor(_: StageAttempt) -> _IeaAdvisorStageState:
            state.advisor_invocation = None
            state.advisor_output = None
            state.grounding_decision = None
            state.trace_recorded = False
            state.advisor_invocation = self._invoke_intent_advisor(
                prompt=state.current_prompt,
                runtime_context=state.runtime_context,
                advisor_agent=state.advisor_agent,
            )
            state.invocation_failure = None
            return state

        def parse_advisor(
            stage_value: _IeaAdvisorStageState,
            attempt: StageAttempt,
        ) -> _IeaAdvisorStageState:
            invocation = stage_value.advisor_invocation
            if invocation is None:
                raise AssertionError("IEA advisor invocation must exist before structured parsing")
            try:
                stage_value.advisor_output = coerce_structured_response(
                    invocation.advisor_result,
                    GroundingAdvisorOutput,
                    error_message="IEA advisor returned no structured_response",
                )
            except (RuntimeError, ValidationError) as exc:
                failure = _IeaAdvisorContractFailure(
                    error=exc,
                    errors=(str(exc),),
                )
                stage_value.advisor_contract_failure = failure
                stage_value.validation_outcome = None
                if attempt.is_final_attempt:
                    self._capture_advisor_contract_failure(
                        state=stage_value,
                        failure=failure,
                        attempt_index=attempt.attempt,
                    )
                    raise RuntimeError(
                        "IEA advisor output contract validation failed: "
                        + "; ".join(failure.errors)
                    ) from exc
                raise
            stage_value.advisor_contract_failure = None
            return stage_value

        def repair_advisor(
            stage_value: _IeaAdvisorStageState,
            _: StageAttempt,
        ) -> _IeaAdvisorStageState:
            advisor_output = stage_value.advisor_output
            if advisor_output is None:
                raise RuntimeError("IEA advisor returned no structured_response")
            refreshed_evidence = self._refresh_advisor_stage_evidence(state=stage_value)
            grounding_decision = self._compile_advisor_output(
                advisor_output=advisor_output,
                evidence=refreshed_evidence,
            )
            grounding_decision = self._materialize_catalog_facts(
                grounding_decision=grounding_decision,
                evidence=refreshed_evidence,
            )
            grounding_decision = self._materialize_qos_constraint_facts(
                grounding_decision=grounding_decision,
            )
            stage_value.grounding_decision = self._attach_subscription_evidence(
                grounding_decision=grounding_decision,
                evidence=refreshed_evidence,
            )
            return stage_value

        def validate_advisor(
            stage_value: _IeaAdvisorStageState,
            attempt: StageAttempt,
        ) -> tuple[str, ...]:
            grounding_decision = stage_value.grounding_decision
            refreshed_evidence = stage_value.refreshed_evidence
            if grounding_decision is None or refreshed_evidence is None:
                raise RuntimeError("IEA advisor returned no GroundingDecision payload")
            stage_value.validation_outcome = None
            advisor_errors, grounding_errors, _ = validate_grounding(
                validator=self.validator,
                evidence=refreshed_evidence,
                grounding_decision=grounding_decision,
                grounding_tools=stage_value.grounding_tools,
            )
            outcome = _IeaValidationOutcome(
                advisor_errors=tuple(advisor_errors),
                grounding_errors=tuple(grounding_errors),
            )
            stage_value.validation_outcome = outcome
            self.last_failure_debug = {
                "phase": "intent_encoding",
                "attempt_index": attempt.attempt,
                "grounding_tools": list(stage_value.grounding_tools or []),
                "grounding_decision": grounding_decision.model_dump(mode="json"),
                "advisor_validation_errors": list(outcome.advisor_errors),
                "grounding_validation_errors": list(outcome.grounding_errors),
                "evidence_snapshot": refreshed_evidence.model_dump(mode="json"),
            }
            stage_value.evidence = refreshed_evidence
            return outcome.issues

        def terminal_validation_error(issues: tuple[str, ...]) -> RuntimeError:
            outcome = state.validation_outcome
            if outcome is None or outcome.issues != issues:
                raise RuntimeError("IEA validation outcome was not recorded")
            return outcome.terminal_error()

        def record_stage_failure(retry_context: RetryContext) -> None:
            # Final/non-retryable failures retain the old no-extra-side-effect
            # behavior.  A non-final malformed advisor response is the one
            # path that must retain its tool evidence before prompt repair.
            if retry_context.phase in {RetryPhase.PARSE, RetryPhase.VALIDATION}:
                if state.advisor_invocation is None:
                    raise RuntimeError("IEA failed after invocation without trace context")
                state.advisor_invocation.write_final_trace(
                    grounding_decision=None,
                    error=str(retry_context.error),
                )
                state.trace_recorded = True
            if retry_context.phase is not RetryPhase.PARSE:
                return
            if not retry_policy.should_retry(retry_context):
                return
            failure = state.advisor_contract_failure
            if failure is None or failure.error is not retry_context.error:
                return
            self._capture_advisor_contract_failure(
                state=state,
                failure=failure,
                attempt_index=retry_context.attempt,
            )

        def prepare_advisor_retry(retry_context: RetryContext) -> None:
            if retry_context.phase is RetryPhase.INVOKE:
                error = retry_context.error
                if not isinstance(error, RuntimeError):
                    return
                failure = _IeaInvocationFailure(error=error)
                state.invocation_failure = failure
                # If the failed invocation produced tool results, extract them
                # and merge them into evidence so the retry avoids re-querying
                # catalog/search data and burning its tool quota.
                raw_messages = list(getattr(failure.error, "_tool_output_messages", None) or [])
                if raw_messages:
                    state.evidence = self._refresh_intent_evidence_from_tool_results(
                        evidence=state.evidence,
                        advisor_result={"messages": raw_messages},
                        main_directives=state.main_directives,
                        snapshot_id=state.snapshot_id,
                    )
                log_event(
                    self.logger,
                    "iea_retry",
                    attempt=retry_context.attempt + 1,
                    reason="invocation_error",
                    error=str(failure.error),
                )
                state.current_prompt = build_grounding_validation_retry_prompt(
                    base_prompt=self._build_advisor_prompt(
                        evidence=state.evidence,
                        context=state.context,
                    ),
                    advisor_validation_errors=[],
                    grounding_validation_errors=[],
                    invocation_error=str(failure.error),
                )
                return

            if retry_context.phase is RetryPhase.PARSE:
                failure = state.advisor_contract_failure
                if failure is None:
                    return
                log_event(
                    self.logger,
                    "iea_retry",
                    attempt=retry_context.attempt + 1,
                    reason="advisor_output_contract",
                    advisor_errors=list(failure.errors),
                )
                state.current_prompt = build_grounding_validation_retry_prompt(
                    base_prompt=self._build_advisor_prompt(
                        evidence=state.evidence,
                        context=state.context,
                    ),
                    advisor_validation_errors=list(failure.errors),
                    grounding_validation_errors=[],
                    invocation_error="",
                )
                return

            if retry_context.phase is RetryPhase.VALIDATION:
                outcome = state.validation_outcome
                if outcome is None:
                    return
                log_event(
                    self.logger,
                    "iea_retry",
                    attempt=retry_context.attempt + 1,
                    reason="validation_errors",
                    advisor_errors=list(outcome.advisor_errors),
                    grounding_errors=list(outcome.grounding_errors),
                )
                state.current_prompt = build_grounding_validation_retry_prompt(
                    base_prompt=self._build_advisor_prompt(
                        evidence=state.evidence,
                        context=state.context,
                    ),
                    advisor_validation_errors=list(outcome.advisor_errors),
                    grounding_validation_errors=list(outcome.grounding_errors),
                    invocation_error="",
                )

        runner: StageRunner[_IeaAdvisorStageState] = StageRunner(retry_policy=retry_policy)
        return runner.run(
            invoke=invoke_advisor,
            parse=parse_advisor,
            repair=repair_advisor,
            validate=validate_advisor,
            validation_error=terminal_validation_error,
            on_failure=record_stage_failure,
            on_retry=prepare_advisor_retry,
        )

    def _refresh_advisor_stage_evidence(
        self,
        *,
        state: _IeaAdvisorStageState,
    ) -> IntentEvidence:
        invocation = state.advisor_invocation
        if invocation is None:
            raise RuntimeError("IEA advisor returned no invocation payload")
        advisor_result = invocation.advisor_result
        state.grounding_tools = extract_grounding_tool_names(advisor_result, self.GROUNDING_TOOLS)
        refreshed_evidence = self._refresh_intent_evidence_from_tool_results(
            evidence=state.evidence,
            advisor_result=advisor_result,
            main_directives=state.main_directives,
            snapshot_id=state.snapshot_id,
        )
        state.refreshed_evidence = refreshed_evidence
        return refreshed_evidence

    def _capture_advisor_contract_failure(
        self,
        *,
        state: _IeaAdvisorStageState,
        failure: _IeaAdvisorContractFailure,
        attempt_index: int,
    ) -> None:
        # A narrow raw schema is deliberately strict.  Retain authoritative
        # tool evidence collected before the malformed object so a repair turn
        # can finish from the data without spending its tool quota again.
        evidence = self._refresh_advisor_stage_evidence(state=state)
        state.evidence = evidence
        self.last_failure_debug = {
            "phase": "intent_encoding",
            "attempt_index": attempt_index,
            "grounding_tools": list(state.grounding_tools or []),
            "advisor_validation_errors": list(failure.errors),
            "grounding_validation_errors": [],
            "evidence_snapshot": evidence.model_dump(mode="json"),
        }

    def _invoke_intent_advisor(
        self,
        *,
        prompt: str,
        runtime_context: Any,
        advisor_agent: Any | None = None,
    ) -> IntentAdvisorInvocation:
        self._pending_invoke_messages = [{"role": "user", "content": prompt}]
        base_trace_metadata = dict(runtime_context.trace_attributes)
        active_advisor_agent = advisor_agent or self.advisor_agent
        context_components = measure_context_components(
            {
                "system_prompt": str(getattr(active_advisor_agent, "system_prompt", "") or ""),
                "dynamic_prompt": prompt,
            },
            token_counter=getattr(runtime_context, "token_counter", None),
        )
        log_event(self.logger, "iea_context_components", **context_components)
        invocation = AgentInvocation(
            messages=[AgentMessage(role="user", content=prompt)],
            metadata=TraceMetadata(
                path_label="advisor_path",
                prompt_name="intent_grounding_advisor",
                prompt_version="2.0.0",
                attributes={
                    **{
                        key: value
                        for key, value in base_trace_metadata.items()
                        if key not in {"path_label", "scenario_id", "scenario_tags"}
                    },
                    "context_token_components": context_components,
                },
            ),
        )
        try:
            result = active_advisor_agent.invoke(
                invocation,
                context=runtime_context,
                defer_trace=True,
            )
        except Exception as exc:
            if isinstance(exc, ToolLoopExecutionError):
                failed_tool_call = exc.failed_tool_call or {}
                if failed_tool_call:
                    invocation_error = RuntimeError(
                        f"IEA tool call failed: {failed_tool_call.get('name') or '<unknown>'}: {exc}"
                    )
                    # Retain successful calls made before the failed one.  A
                    # retry can then finish from their evidence rather than
                    # re-running catalog/search queries and hitting another
                    # tool-limit loop.
                    invocation_error._tool_output_messages = list(exc.output_messages)
                    raise invocation_error from exc
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
            invocation=invocation,
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
        return self.evidence_builder.build_intent_evidence(
            user_input=normalized_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload={},
            catalog_evidence_observed=False,
            search_evidence_observed=False,
            semantic_candidates=[],
            am_context_payload={},
            am_policy_candidates=[],
            subscription_payload={},
        )

    def _refresh_intent_evidence_from_tool_results(
        self,
        *,
        evidence: IntentEvidence,
        advisor_result: AgentRunResult,
        main_directives: Dict[str, Any],
        snapshot_id: str,
    ) -> IntentEvidence:
        semantic_candidates: List[Dict[str, Any]] = list(evidence.semantic_candidates or [])
        catalog_payload: Dict[str, Any] = dict(evidence.catalog_payload or {})
        catalog_evidence_observed = bool(evidence.catalog_evidence_observed)
        search_evidence_observed = bool(evidence.search_evidence_observed)
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
                search_evidence_observed = True
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

        return self.evidence_builder.build_intent_evidence(
            user_input=evidence.user_input,
            supi=evidence.supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            catalog_evidence_observed=catalog_evidence_observed,
            search_evidence_observed=search_evidence_observed,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
            subscription_payload=subscription_payload,
        )

    @staticmethod
    def _compile_advisor_output(
        *,
        advisor_output: GroundingAdvisorOutput,
        evidence: IntentEvidence,
    ) -> GroundingDecision:
        """Build the OSA handoff from narrow LLM choices and tool-owned facts."""
        subscription = dict(evidence.subscription_summary or {})
        authorized_snssais = [
            str(item or "").strip()
            for item in (subscription.get("authorized_snssais") or [])
            if str(item or "").strip()
        ]
        evidence_sources: Dict[str, List[str]] = {}
        if evidence.catalog_evidence_observed:
            evidence_sources["get_sm_ue_flow_catalog"] = ["runtime"]
        if evidence.search_evidence_observed:
            evidence_sources["search_sm_flow_targets"] = ["runtime"]
        if evidence.am_context_summary:
            evidence_sources["get_am_policy_context"] = ["runtime"]
        if evidence.am_policy_candidates:
            evidence_sources["search_am_policy_targets"] = ["runtime"]
        mobility_targets: Dict[str, Any] = {}
        if evidence.am_context_summary:
            mobility_targets["summary"] = dict(evidence.am_context_summary)

        return GroundingDecision(
            flows=[
                FlowSelector(
                    supi=IntentEncodingAgent._binding_supi_from_evidence(flow=flow, evidence=evidence),
                    app_id=str(flow.app_id or "").strip(),
                    flow_id=str(flow.flow_id or "").strip() or None,
                    name=str(flow.name or "").strip(),
                    resolution_status=str(flow.resolution_status or "resolved").strip(),
                )
                for flow in (advisor_output.flows or [])
            ],
            mobility_intent=dict(advisor_output.mobility_intent or {}),
            grounding_evidence=GroundingEvidenceBundle(
                grounded_supi=str(evidence.supi or "").strip(),
                grounded_apps=[dict(item) for item in (evidence.candidate_apps or []) if isinstance(item, dict)],
                grounded_flows=[item.model_dump(mode="json") for item in (evidence.candidate_flows or [])],
                grounded_mobility_targets=mobility_targets,
                evidence_sources=evidence_sources,
            ),
            qos_operation_constraints=[
                QosOperationConstraint(
                    flow_id=str(constraint.flow_id or "").strip(),
                    app_id=str(constraint.app_id or "").strip(),
                    operation_type=str(constraint.operation_type or "").strip(),
                    require_slice_change=bool(constraint.require_slice_change),
                    target_slice_preference=str(constraint.target_slice_preference or "").strip(),
                    no_op_allowed=bool(constraint.no_op_allowed),
                    rationale=list(constraint.rationale or []),
                )
                for constraint in (advisor_output.qos_operation_constraints or [])
            ],
            slice_migration_authorization=SliceMigrationAuthorization(
                decision=advisor_output.slice_migration_authorization.decision,
                authority=str(subscription.get("authority") or "").strip(),
                authorized_snssais=authorized_snssais,
                target_snssais=list(advisor_output.slice_migration_authorization.target_snssais or []),
                subscription_change_required=bool(
                    advisor_output.slice_migration_authorization.subscription_change_required
                ),
                rationale=list(advisor_output.slice_migration_authorization.rationale or []),
            ),
            open_questions=list(advisor_output.open_questions or []),
        )

    @staticmethod
    def _binding_supi_from_evidence(*, flow: Any, evidence: IntentEvidence) -> str:
        """Attach Main/tool-owned UE identity without asking IEA to echo it."""
        flow_id = str(getattr(flow, "flow_id", "") or "").strip()
        flow_name = str(getattr(flow, "name", "") or "").strip()
        for item in (evidence.catalog_payload or {}).get("flow_catalog") or []:
            if isinstance(item, dict) and flow_id and str(item.get("flow_id") or "").strip() == flow_id:
                return str(item.get("supi") or "").strip()
        for item in (evidence.semantic_candidates or []):
            if not isinstance(item, dict):
                continue
            if flow_id and str(item.get("flow_id") or "").strip() == flow_id:
                return str(item.get("supi") or "").strip()
        for target in (evidence.explicit_flow_targets or []):
            if str(target.flow_name or "").strip() == flow_name and str(target.supi or "").strip():
                return str(target.supi or "").strip()
        requested_supis = extract_requested_supis(evidence.supi, evidence.user_input)
        return requested_supis[0] if len(requested_supis) == 1 else ""

    @staticmethod
    def _materialize_qos_constraint_facts(
        *,
        grounding_decision: GroundingDecision,
    ) -> GroundingDecision:
        """Derive source/excluded slice facts from the materialized flow binding."""
        by_flow_id = {
            str(flow.flow_id or "").strip(): flow
            for flow in (grounding_decision.flows or [])
            if str(flow.flow_id or "").strip()
        }
        enriched = grounding_decision.model_copy(deep=True)
        constraints: List[QosOperationConstraint] = []
        for constraint in enriched.qos_operation_constraints or []:
            source_slice_snssai = constraint.source_slice_snssai
            if constraint.require_slice_change and not str(source_slice_snssai or "").strip():
                flow = by_flow_id.get(str(constraint.flow_id or "").strip())
                source_slice_snssai = getattr(flow, "current_slice_snssai", None) if flow is not None else None
            constraint_payload = constraint.model_dump(mode="json")
            constraint_payload["source_slice_snssai"] = source_slice_snssai
            constraints.append(
                QosOperationConstraint(**constraint_payload)
            )
        enriched.qos_operation_constraints = constraints
        return enriched

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
        requires_slice_change = any(
            bool(constraint.require_slice_change)
            for constraint in (enriched.qos_operation_constraints or [])
        )
        if requires_slice_change:
            authorization = enriched.slice_migration_authorization.model_copy(
                update={
                    "authority": str(subscription.get("authority") or "").strip(),
                    "authorized_snssais": [
                        str(item or "").strip()
                        for item in (subscription.get("authorized_snssais") or [])
                        if str(item or "").strip()
                    ],
                }
            )
            enriched.slice_migration_authorization = authorization
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

