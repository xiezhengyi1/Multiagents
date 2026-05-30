from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.runtime import ArtifactEnvelope
from shared.runtime import ToolLoopExecutionError
from shared.agents import BaseAgent, coerce_structured_response
from shared.runtime import ArtifactWorkerMixin
from ...domain.control_plane import (
    GlobalControlIntent,
    MainRetryScope,
)
from shared.logging import log_event

from .prompts import MAIN_CONTROL_DYNAMIC_RULES, MAIN_CONTROL_SYSTEM_PROMPT


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


class MainControlAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "main_control"

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507", use_local_model: bool = False) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "main_control"
        self.initialize_agent_runtime(logger_color="\033[93m")
        self.tools = []
        self.agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=MAIN_CONTROL_SYSTEM_PROMPT,
            response_model=GlobalControlIntent,
            max_iterations=6,
        )

    def analyze_global_intent(
        self,
        *,
        user_input: str,
        session_id: str = "",
        snapshot_id: str = "",
        context: str = "",
        trace_metadata: Dict[str, Any] | None = None,
    ) -> GlobalControlIntent:
        self.ensure_worker_runtime_initialized()
        log_event(self.logger, "main_control_start")
        payload = {
            "role": "user",
            "content": (
                f"User input:\n{user_input}\n\n"
                f"Coordinator context:\n{context or 'N/A'}\n\n"
                f"{MAIN_CONTROL_DYNAMIC_RULES.strip()}\n\n"
                "Resolve only the round-level domain routing, retry scope, explicit SUPI already present in the request, and keep intent_encoding_guidance empty unless non-empty routing guidance is strictly necessary."
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
        try:
            current_prompt = payload["content"]
            invocation: Optional[MainControlInvocation] = None
            intent: Optional[GlobalControlIntent] = None
            validation_errors: List[str] = []
            invocation_error: str = ""
            for attempt_index in range(3):
                log_event(
                    self.logger,
                    "main_control_attempt_start",
                    attempt=attempt_index + 1,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                )
                try:
                    invocation = self._invoke_global_intent_result(
                        current_prompt,
                        runtime_context=runtime_context,
                        trace_metadata=base_trace_metadata,
                    )
                    parsed_intent = self._validate_global_intent_result(invocation.raw_result)
                except RuntimeError as exc:
                    invocation_error = str(exc)
                    log_event(
                        self.logger,
                        "main_control_validation_failed",
                        attempt=attempt_index + 1,
                        validation_errors=invocation_error,
                        supi="<empty>",
                    )
                    if attempt_index == 2:
                        raise
                    current_prompt = self._build_validation_retry_prompt(
                        base_prompt=payload["content"],
                        validation_errors=[],
                        invocation_error=invocation_error,
                    )
                    continue
                intent = parsed_intent
                invocation_error = ""
                validation_errors = self._validate_global_intent(
                    intent,
                    user_input=user_input,
                    context=context,
                )
                if not validation_errors:
                    log_event(
                        self.logger,
                        "main_control_attempt_success",
                        attempt=attempt_index + 1,
                        requested_domains=",".join(item.value for item in intent.requested_domains),
                        supi=intent.supi,
                    )
                    break
                log_event(
                    self.logger,
                    "main_control_validation_failed",
                    attempt=attempt_index + 1,
                    validation_errors=" || ".join(validation_errors),
                    supi=str(intent.supi or "").strip() or "<empty>",
                )
                if attempt_index == 2:
                    raise RuntimeError(
                        "Main Agent could not produce a valid GlobalControlIntent: "
                        + "; ".join(validation_errors)
                    )
                current_prompt = self._build_validation_retry_prompt(
                    base_prompt=payload["content"],
                    validation_errors=validation_errors,
                    invocation_error="",
                )
            if intent is None:
                raise RuntimeError("Main Agent returned no intent payload")
            if not intent.session_id:
                intent.session_id = session_id
            if not intent.snapshot_id:
                intent.snapshot_id = snapshot_id
            if not intent.raw_input:
                intent.raw_input = user_input
            self._enrich_global_intent_contract(
                intent,
                context=context,
                user_input=user_input,
            )
            if invocation is None:
                raise RuntimeError("Main Agent finished without a trace invocation")
            invocation.write_final_trace(
                status="success",
                structured_response=intent.model_dump(mode="json"),
            )
            return intent
        except Exception as exc:
            if "invocation" in locals() and invocation is not None:
                invocation.write_final_trace(
                    status="error",
                    structured_response=None if "intent" not in locals() or intent is None else intent.model_dump(mode="json"),
                    error=str(exc),
                )
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

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
                # Print the last AI message so we can see what the model actually output
                for msg in reversed(exc.output_messages or []):
                    content = getattr(msg, "content", None)
                    if content and getattr(msg, "type", "") in ("ai", "AIMessage") or msg.__class__.__name__ == "AIMessage":
                        print(f"[DEBUG] Main Agent last AI output:\n{content}")
                        break
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

    @staticmethod
    def _build_validation_retry_prompt(*, base_prompt: str, validation_errors: List[str], invocation_error: str) -> str:
        issues: List[str] = []
        if invocation_error:
            issues.append(invocation_error)
        if validation_errors:
            issues.extend(validation_errors)

        cleaned = re.sub(
            r'\n\nRetry feedback \(attempt \d+\).*$',
            '',
            base_prompt,
            flags=re.DOTALL,
        )
        cleaned = re.sub(
            r'\n\nRetry feedback:.*$',
            '',
            cleaned,
            flags=re.DOTALL,
        )
        cleaned = re.sub(
            r'\n\nYour previous draft failed validation.*$',
            '',
            cleaned,
            flags=re.DOTALL,
        )
        required_keys = (
            "supi, round_strategy, next_agent, requested_domains, domain_evidence, "
            "control_semantics, objective_profile, investigation_targets, uncertainty_flags, "
            "retry_scope, required_evidence, forbidden_assumptions, intent_encoding_guidance, "
            "routing_decision, routing_rationale, routing_confidence, reuse_contract, "
            "handoff_expectations"
        )

        return (
            f"{cleaned}\n\n"
            "Retry feedback:\n"
            "Correct the output now.\n"
            "Return exactly one MainControlInvocation raw_result-shaped JSON object.\n"
            "Top-level output must be an object, never a list.\n"
            "Do not return markdown, prose, bullets, arrays, or partial sub-objects.\n"
            "Put the full GlobalControlIntent object under structured_response.\n"
            "Use messages=[] in the structure example; do not put routing fields under messages.\n"
            f"Required structured_response keys: {required_keys}.\n"
            "If a previous draft was a list, convert the intended route into structured_response.\n"
            "If a previous draft only contained reuse_contract fields, keep those fields nested under structured_response.reuse_contract and add all missing structured_response routing fields.\n\n"
            "Validation errors:\n- " + "\n- ".join(issues)
        )

    @staticmethod
    def _validate_global_intent_result(result: Dict[str, Any]) -> GlobalControlIntent:
        try:
            intent = coerce_structured_response(
                result,
                GlobalControlIntent,
                error_message="Main Agent returned no structured_response",
            )
            return intent
        except Exception as exc:
            raise RuntimeError(f"Main Agent returned invalid GlobalControlIntent payload: {exc}") from exc

    @staticmethod
    def _validate_global_intent(
        intent: GlobalControlIntent,
        *,
        user_input: str,
        context: str = "",
    ) -> List[str]:
        errors: List[str] = []
        allowed_round_strategies = {"initial_grounding", "regrounding", "policy_revision", "joint_replan"}
        allowed_investigation_targets = {
            "domain_boundary",
            "ue_binding",
            "qos_flow_binding",
            "mobility_target_binding",
            "policy_feasibility",
            "cross_domain_consistency",
            "assurance_gap",
        }
        allowed_uncertainty_flags = {
            "domain_ambiguous",
            "identifier_risk",
            "runtime_evidence_missing",
            "execution_feedback_incomplete",
            "conflict_signal_present",
        }
        allowed_retry_scopes = {
            "full_reground",
            "partial_reground",
            "target_stable",
            "execution_retry_forbidden",
            "",
        }
        round_index = 0
        round_match = re.search(r"(?im)^\s*-\s*round_index:\s*(\d+)\s*$", str(context or ""))
        if round_match:
            round_index = int(round_match.group(1))
        else:
            try:
                parsed_context = json.loads(str(context or "").strip()) if str(context or "").strip() else {}
            except Exception:
                parsed_context = {}
            if isinstance(parsed_context, dict):
                try:
                    round_index = int(parsed_context.get("round_index") or 0)
                except (TypeError, ValueError):
                    round_index = 0
        if not intent.requested_domains:
            errors.append("requested_domains is empty")
        else:
            values = [item.value for item in intent.requested_domains]
            if any(item not in {"qos", "mobility"} for item in values):
                errors.append(f"requested_domains contains unsupported values: {values}")
            if str(intent.next_agent or "").strip() not in {"intent_encoding", "optimization_strategy"}:
                errors.append("next_agent must be either intent_encoding or optimization_strategy")
            if round_index <= 1 and str(intent.next_agent or "").strip() != "intent_encoding":
                errors.append("round-1 main routing must set next_agent=intent_encoding; optimization_strategy is retry-only")
        round_strategy = str(intent.round_strategy.value if hasattr(intent.round_strategy, "value") else intent.round_strategy or "").strip()
        if round_strategy not in allowed_round_strategies:
            errors.append(f"round_strategy contains unsupported value: {round_strategy or '<empty>'}")
        elif round_index <= 1 and round_strategy != "initial_grounding":
            errors.append("round-1 main routing must set round_strategy=initial_grounding")
        explicit_supis = re.findall(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        unique_explicit_supis = list(dict.fromkeys(explicit_supis))
        if len(unique_explicit_supis) == 1:
            explicit_supi = unique_explicit_supis[0]
            if str(intent.supi or "").strip() != explicit_supi:
                errors.append(f"supi must equal explicit user-provided identifier {explicit_supi}")
        if isinstance(intent.domain_evidence, dict) and any(intent.domain_evidence.values()):
            unknown_domains = [key for key in intent.domain_evidence.keys() if key not in {"qos", "mobility"}]
            if unknown_domains:
                errors.append(f"domain_evidence contains unsupported keys: {unknown_domains}")
            requested = {item.value for item in intent.requested_domains}
            evidence_keys = {str(key).strip().lower() for key, values in intent.domain_evidence.items() if values}
            if not requested.issubset(evidence_keys):
                errors.append(
                    f"domain_evidence must cover every requested domain: requested={sorted(requested)} evidence={sorted(evidence_keys)}"
                )
        investigation_targets = [
            item.value if hasattr(item, "value") else str(item or "").strip()
            for item in (intent.investigation_targets or [])
        ]
        unknown_investigation_targets = [
            item for item in investigation_targets if item not in allowed_investigation_targets
        ]
        if unknown_investigation_targets:
            errors.append(
                f"investigation_targets contains unsupported values: {sorted(set(unknown_investigation_targets))}"
            )
        uncertainty_flags = [
            item.value if hasattr(item, "value") else str(item or "").strip()
            for item in (intent.uncertainty_flags or [])
        ]
        unknown_uncertainty_flags = [
            item for item in uncertainty_flags if item not in allowed_uncertainty_flags
        ]
        if unknown_uncertainty_flags:
            errors.append(
                f"uncertainty_flags contains unsupported values: {sorted(set(unknown_uncertainty_flags))}"
            )
        retry_scope = (
            intent.retry_scope.value
            if getattr(intent, "retry_scope", None) is not None and hasattr(intent.retry_scope, "value")
            else str(getattr(intent, "retry_scope", "") or "").strip()
        )
        if retry_scope not in allowed_retry_scopes:
            errors.append(f"retry_scope contains unsupported value: {retry_scope or '<empty>'}")
        if round_index <= 1 and retry_scope and retry_scope != "full_reground":
            intent.retry_scope = MainRetryScope.FULL_REGROUND
            retry_scope = "full_reground"
        next_agent = str(intent.next_agent or "").strip()
        if not str(intent.routing_decision or "").strip():
            errors.append("routing_decision must not be empty")
        if not str(intent.routing_rationale or "").strip():
            errors.append("routing_rationale must not be empty")
        if next_agent == "optimization_strategy" and not bool(intent.reuse_contract.allowed):
            errors.append("next_agent=optimization_strategy requires reuse_contract.allowed=true")
        if next_agent == "intent_encoding" and intent.reuse_contract.allowed:
            errors.append("reuse_contract.allowed must be false when next_agent=intent_encoding")
        if next_agent == "intent_encoding" and not str(intent.intent_encoding_guidance or "").strip() and round_index > 1:
            errors.append("retry routing into intent_encoding requires explicit intent_encoding_guidance")
        if not intent.handoff_expectations:
            errors.append("handoff_expectations must not be empty")
        # 关键步骤：诊断类别只作为上下文交给 LLM，不在这里做域级硬裁决。
        return errors

    @staticmethod
    def _enrich_global_intent_contract(
        intent: GlobalControlIntent,
        *,
        context: str,
        user_input: str,
    ) -> None:
        if intent.intent_encoding_guidance and str(intent.next_agent or "").strip() != "intent_encoding":
            raise ValueError("intent_encoding_guidance must stay empty unless next_agent=intent_encoding")
