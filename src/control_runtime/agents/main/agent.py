from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared.runtime import ToolLoopExecutionError
from shared.agents import (
    BaseAgent,
    RetryContext,
    RetryPhase,
    RetryPolicy,
    StageAttempt,
    StageRunner,
)
from ...tool_capabilities import CONTROL_RUNTIME_CAPABILITY_ALIASES
from .contracts import (
    GlobalControlIntent,
    MainControlDecisionWire,
)
from .compiler import (
    normalize_semantic_target_scope,
    validate_global_intent as validate_main_global_intent,
    validate_global_intent_result,
)
from shared.logging import log_event, setup_logger

from ...context.prompts.main import (
    build_main_user_prompt,
    build_main_validation_retry_prompt,
    render_main_system_prompt,
)


@dataclass
class MainControlInvocation:
    raw_result: Dict[str, Any]
    trace_agent: Any
    trace_payload: Dict[str, Any]
    runtime_context: Any

    def write_final_trace(
        self,
        *,
        status: str,
        structured_response: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.trace_agent.write_trace(
            payload=self.trace_payload,
            context=self.runtime_context,
            result=self.raw_result,
            status=status,
            error=error,
            structured_response_override=structured_response,
        )


@dataclass
class _MainIntentStageState:
    """Mutable state retained across Main's bounded intent-generation attempts."""

    base_prompt: str
    current_prompt: str
    user_input: str
    session_id: str
    snapshot_id: str
    round_index: int
    runtime_context: Any
    trace_metadata: Dict[str, Any]
    invocation: Optional[MainControlInvocation] = None
    intent: Optional[GlobalControlIntent] = None
    projected_fields: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    invocation_error: str = ""


class MainControlAgent(BaseAgent):
    agent_name = "main_control"

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "main_control"
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[93m")
        self.tools = []
        self.agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=render_main_system_prompt(),
            response_model=MainControlDecisionWire,
            max_iterations=6,
            capability_aliases=CONTROL_RUNTIME_CAPABILITY_ALIASES,
        )
        self.reset_schema_projection_metrics()

    def reset_schema_projection_metrics(self) -> None:
        self._schema_projection_count = 0
        self._schema_projected_fields: List[str] = []

    @property
    def schema_projection_metrics(self) -> Dict[str, Any]:
        return {
            "count": self._schema_projection_count,
            "field_count": len(self._schema_projected_fields),
            "fields": list(dict.fromkeys(self._schema_projected_fields)),
        }

    def _record_schema_projection(self, fields: List[str]) -> None:
        if not fields:
            return
        self._schema_projection_count += 1
        self._schema_projected_fields.extend(fields)
        log_event(
            self.logger,
            "main_control_schema_projected",
            projected_field_count=len(fields),
            projected_fields=",".join(fields),
        )

    def analyze_global_intent(
        self,
        *,
        user_input: str,
        session_id: str = "",
        snapshot_id: str = "",
        context: str = "",
        round_index: int = 1,
        trace_metadata: Dict[str, Any] | None = None,
    ) -> GlobalControlIntent:
        log_event(self.logger, "main_control_start")
        payload = {
            "role": "user",
            "content": build_main_user_prompt(
                user_input=user_input,
                coordinator_context=context,
            ),
        }
        token_budget, token_counter = self._resolve_token_context()
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            thread_id=session_id,
            token_budget=token_budget,
            token_counter=token_counter,
            trace_metadata=trace_metadata,
        )
        self._pending_invoke_messages = [payload]
        base_trace_metadata = {
            **(trace_metadata or {}),
            "path_label": "global_intent_advisor",
        }
        state = _MainIntentStageState(
            base_prompt=payload["content"],
            current_prompt=payload["content"],
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_index=round_index,
            runtime_context=runtime_context,
            trace_metadata=base_trace_metadata,
        )
        try:
            state = self._run_global_intent_stage(state=state)
            if state.intent is None:
                raise RuntimeError("Main Agent returned no intent payload")
            if state.invocation is None:
                raise RuntimeError("Main Agent finished without a trace invocation")
            state.invocation.write_final_trace(
                status="success",
                structured_response=state.intent.model_dump(mode="json"),
            )
            return state.intent
        except Exception as exc:
            if state.invocation is not None:
                state.invocation.write_final_trace(
                    status="error",
                    structured_response=None if state.intent is None else state.intent.model_dump(mode="json"),
                    error=str(exc),
                )
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    def _run_global_intent_stage(self, *, state: _MainIntentStageState) -> _MainIntentStageState:
        """Run Main's existing invoke/parse/validation lifecycle via StageRunner."""

        retry_policy = RetryPolicy(
            max_attempts=3,
            retryable_exceptions_by_phase=(
                (RetryPhase.INVOKE, (RuntimeError,)),
                (RetryPhase.PARSE, (RuntimeError,)),
                (RetryPhase.VALIDATION, (RuntimeError,)),
            ),
        )

        def before_global_intent_attempt(attempt: StageAttempt) -> None:
            log_event(
                self.logger,
                "main_control_attempt_start",
                attempt=attempt.attempt,
                session_id=state.session_id,
                snapshot_id=state.snapshot_id,
            )
            state.projected_fields = []

        def invoke_global_intent(_: StageAttempt) -> _MainIntentStageState:
            state.invocation = self._invoke_global_intent_result(
                state.current_prompt,
                runtime_context=state.runtime_context,
                trace_metadata=state.trace_metadata,
            )
            return state

        def parse_global_intent(
            stage_value: _MainIntentStageState,
            _: StageAttempt,
        ) -> _MainIntentStageState:
            invocation = stage_value.invocation
            if invocation is None:
                raise AssertionError("Main intent invocation must exist before structured parsing")
            stage_value.intent = validate_global_intent_result(
                invocation.raw_result,
                user_input=stage_value.user_input,
                session_id=stage_value.session_id,
                snapshot_id=stage_value.snapshot_id,
                round_index=stage_value.round_index,
                projected_fields=stage_value.projected_fields,
            )
            return stage_value

        def validate_global_intent(
            stage_value: _MainIntentStageState,
            attempt: StageAttempt,
        ) -> List[str]:
            self._record_schema_projection(stage_value.projected_fields)
            intent = stage_value.intent
            if intent is None:
                raise AssertionError("Main intent must exist before semantic validation")
            normalize_semantic_target_scope(
                intent,
                user_input=stage_value.user_input,
            )
            stage_value.invocation_error = ""
            stage_value.validation_errors = validate_main_global_intent(
                intent,
                user_input=stage_value.user_input,
                round_index=stage_value.round_index,
            )
            if not stage_value.validation_errors:
                log_event(
                    self.logger,
                    "main_control_attempt_success",
                    attempt=attempt.attempt,
                    requested_domains=",".join(item.value for item in intent.requested_domains),
                    supi=intent.supi,
                )
            return stage_value.validation_errors

        def terminal_validation_error(issues: tuple[str, ...]) -> RuntimeError:
            return RuntimeError(
                "Main Agent could not produce a valid MainControlDecision: "
                + "; ".join(issues)
            )

        def record_stage_failure(retry_context: RetryContext) -> None:
            if retry_context.phase in {RetryPhase.INVOKE, RetryPhase.PARSE}:
                if not isinstance(retry_context.error, RuntimeError):
                    return
                self._record_schema_projection(state.projected_fields)
                state.invocation_error = str(retry_context.error)
                log_event(
                    self.logger,
                    "main_control_validation_failed",
                    attempt=retry_context.attempt,
                    validation_errors=state.invocation_error,
                    supi="<empty>",
                )
                return
            if retry_context.phase is RetryPhase.VALIDATION:
                intent = state.intent
                log_event(
                    self.logger,
                    "main_control_validation_failed",
                    attempt=retry_context.attempt,
                    validation_errors=" || ".join(retry_context.issues),
                    supi=(str(intent.supi or "").strip() or "<empty>") if intent is not None else "<empty>",
                )

        def prepare_stage_retry(retry_context: RetryContext) -> None:
            if retry_context.phase in {RetryPhase.INVOKE, RetryPhase.PARSE}:
                state.current_prompt = build_main_validation_retry_prompt(
                    base_prompt=state.base_prompt,
                    validation_errors=[],
                    invocation_error=str(retry_context.error),
                )
                return
            if retry_context.phase is RetryPhase.VALIDATION:
                state.current_prompt = build_main_validation_retry_prompt(
                    base_prompt=state.base_prompt,
                    validation_errors=list(retry_context.issues),
                    invocation_error="",
                )

        runner: StageRunner[_MainIntentStageState] = StageRunner(retry_policy=retry_policy)
        return runner.run(
            invoke=invoke_global_intent,
            before_attempt=before_global_intent_attempt,
            parse=parse_global_intent,
            validate=validate_global_intent,
            validation_error=terminal_validation_error,
            on_failure=record_stage_failure,
            on_retry=prepare_stage_retry,
        )

    def _invoke_global_intent_result(self, user_prompt: str, *, runtime_context: Any, trace_metadata: Dict[str, Any]) -> MainControlInvocation:
        self._pending_invoke_messages = [{"role": "user", "content": user_prompt}]
        payload = {
            "messages": self._pending_invoke_messages,
            "trace_write_mode": "manual",
            "trace_metadata": dict(trace_metadata or {}),
        }
        try:
            result = self.agent.invoke(payload, context=runtime_context)
        except Exception as exc:
            if isinstance(exc, ToolLoopExecutionError):
                failed_tool_call = exc.failed_tool_call or {}
                if failed_tool_call:
                    raise RuntimeError(
                        f"Main Agent tool call failed: {failed_tool_call.get('name') or '<unknown>'}: {exc}"
                    ) from exc
                message = str(exc)
                if "max iterations" in message.lower():
                    raise RuntimeError(f"Main Agent did not converge to valid JSON: {message}") from exc
            raise RuntimeError(f"Main Agent invocation failed before structured output validation: {exc}") from exc
        return MainControlInvocation(
            raw_result=result,
            trace_agent=self.agent,
            trace_payload=payload,
            runtime_context=runtime_context,
        )
