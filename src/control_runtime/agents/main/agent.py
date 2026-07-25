from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from shared.runtime import ArtifactEnvelope
from shared.runtime import ToolLoopExecutionError
from shared.agents import BaseAgent, coerce_structured_response
from shared.runtime import ArtifactWorkerMixin
from ...domain.control_plane import (
    GlobalControlIntent,
    MainControlDecision,
    MainRetryScope,
    MainRoundStrategy,
    ReuseContract,
)
from shared.logging import log_event

from ...context.prompts import MAIN_CONTROL_DYNAMIC_RULES, MainPromptBuilder, RetryPromptBuilder


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


class MainControlDecisionWire(BaseModel):
    """Permissive transport shape repaired before strict decision validation."""

    model_config = ConfigDict(extra="allow")

    supi: str = ""
    next_agent: str = ""
    requested_domains: List[Any] = []
    serving_slice_change_requested: bool = False
    domain_evidence: Dict[str, Any] = {}
    control_semantics: Dict[str, Any] = {}
    objective_profile: Dict[str, Any] = {}
    investigation_targets: List[Any] = []
    uncertainty_flags: List[Any] = []
    retry_scope: str = ""
    required_evidence: List[Any] = []
    forbidden_assumptions: List[Any] = []
    intent_encoding_guidance: str = ""


class MainControlAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "main_control"
    _CONTROL_SEMANTICS_LEAK_FIELDS = (
        "objective_profile",
        "investigation_targets",
        "uncertainty_flags",
        "retry_scope",
        "required_evidence",
        "forbidden_assumptions",
        "intent_encoding_guidance",
        "routing_decision",
        "routing_rationale",
        "reuse_contract",
    )
    _DECISION_FIELDS = frozenset(
        {
            "supi", "next_agent", "requested_domains", "serving_slice_change_requested",
            "domain_evidence", "control_semantics", "objective_profile",
            "investigation_targets", "uncertainty_flags", "retry_scope",
            "required_evidence", "forbidden_assumptions", "intent_encoding_guidance",
        }
    )
    _COMPILED_FIELDS = frozenset(
        {
            "session_id", "snapshot_id", "raw_input", "round_strategy",
            "routing_decision", "routing_rationale", "reuse_contract",
        }
    )

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507", use_local_model: bool = False) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "main_control"
        self.initialize_agent_runtime(logger_color="\033[93m")
        self.tools = []
        self.agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=MainPromptBuilder().system_prompt(),
            response_model=MainControlDecisionWire,
            max_iterations=6,
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
                "Resolve the staged semantic plan, round-level routing, retry scope, and explicit SUPI already present in the request. "
                "On a retry routed to intent_encoding, provide explicit intent_encoding_guidance; otherwise leave it empty unless needed."
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
                projected_fields: List[str] = []
                try:
                    invocation = self._invoke_global_intent_result(
                        current_prompt,
                        runtime_context=runtime_context,
                        trace_metadata=base_trace_metadata,
                    )
                    parsed_intent = self._validate_global_intent_result(
                        invocation.raw_result,
                        user_input=user_input,
                        session_id=session_id,
                        snapshot_id=snapshot_id,
                        context=context,
                        projected_fields=projected_fields,
                    )
                except RuntimeError as exc:
                    self._record_schema_projection(projected_fields)
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
                self._record_schema_projection(projected_fields)
                intent = parsed_intent
                self._normalize_semantic_target_scope(
                    intent,
                    user_input=user_input,
                )
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
                        "Main Agent could not produce a valid MainControlDecision: "
                        + "; ".join(validation_errors)
                    )
                current_prompt = self._build_validation_retry_prompt(
                    base_prompt=payload["content"],
                    validation_errors=validation_errors,
                    invocation_error="",
                )
            if intent is None:
                raise RuntimeError("Main Agent returned no intent payload")
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
        return RetryPromptBuilder().build_main(
            base_prompt=base_prompt,
            validation_errors=validation_errors,
            invocation_error=invocation_error,
        )

    @staticmethod
    def _validate_global_intent_result(
        result: Dict[str, Any],
        *,
        user_input: str = "",
        session_id: str = "",
        snapshot_id: str = "",
        context: str = "",
        projected_fields: Optional[List[str]] = None,
    ) -> GlobalControlIntent:
        try:
            raw_payload = MainControlAgent._raw_structured_payload(result)
            decision = coerce_structured_response(
                MainControlAgent._repair_global_intent_payload(
                    result,
                    projected_fields=projected_fields,
                ),
                MainControlDecision,
                error_message="Main Agent returned no structured MainControlDecision",
            )
            return MainControlAgent._compile_global_intent(
                decision,
                user_input=str(user_input or raw_payload.get("raw_input") or ""),
                session_id=session_id or str(raw_payload.get("session_id") or ""),
                snapshot_id=snapshot_id or str(raw_payload.get("snapshot_id") or ""),
                context=context,
            )
        except Exception as exc:
            raise RuntimeError(f"Main Agent returned invalid MainControlDecision payload: {exc}") from exc

    @staticmethod
    def _raw_structured_payload(result: Dict[str, Any]) -> Dict[str, Any]:
        structured = result.get("structured_response")
        if isinstance(structured, BaseModel):
            return structured.model_dump(mode="json", exclude_unset=True)
        if isinstance(structured, str):
            try:
                structured = json.loads(structured)
            except json.JSONDecodeError:
                return {}
        return dict(structured) if isinstance(structured, dict) else {}

    @staticmethod
    def _compile_global_intent(
        decision: MainControlDecision,
        *,
        user_input: str,
        session_id: str,
        snapshot_id: str,
        context: str,
    ) -> GlobalControlIntent:
        parsed_context = MainControlAgent._parse_context_payload(context)
        try:
            round_index = int(parsed_context.get("round_index") or 1)
        except (TypeError, ValueError):
            round_index = 1
        next_agent = str(decision.next_agent or "").strip()
        if round_index <= 1:
            round_strategy = MainRoundStrategy.INITIAL_GROUNDING
            retry_scope = MainRetryScope.FULL_REGROUND
        elif next_agent == "optimization_strategy":
            round_strategy = MainRoundStrategy.POLICY_REVISION
            retry_scope = decision.retry_scope or MainRetryScope.TARGET_STABLE
        else:
            round_strategy = MainRoundStrategy.REGROUNDING
            retry_scope = decision.retry_scope or MainRetryScope.FULL_REGROUND

        reuse_allowed = next_agent == "optimization_strategy"
        guidance = str(decision.intent_encoding_guidance or "").strip()
        if round_index > 1 and next_agent == "intent_encoding" and not guidance:
            guidance = "Re-ground the active stage using required_evidence; do not assume forbidden_assumptions."
        domains = ",".join(item.value for item in decision.requested_domains) or "none"
        return GlobalControlIntent(
            session_id=session_id,
            snapshot_id=snapshot_id,
            raw_input=user_input,
            supi=decision.supi,
            round_strategy=round_strategy,
            next_agent=decision.next_agent,
            requested_domains=decision.requested_domains,
            serving_slice_change_requested=decision.serving_slice_change_requested,
            domain_evidence=decision.domain_evidence,
            control_semantics=decision.control_semantics,
            objective_profile=decision.objective_profile,
            investigation_targets=decision.investigation_targets,
            uncertainty_flags=decision.uncertainty_flags,
            retry_scope=retry_scope,
            required_evidence=decision.required_evidence,
            forbidden_assumptions=decision.forbidden_assumptions,
            intent_encoding_guidance=guidance,
            routing_decision=f"{round_strategy.value}_{next_agent}_{domains}",
            routing_rationale=(
                f"Round {round_index}: route {domains} control through {next_agent}; "
                "preserve the Main-owned staged semantic plan."
            ),
            reuse_contract=ReuseContract(
                allowed=reuse_allowed,
                preserve_bindings=reuse_allowed,
                preserve_domains=reuse_allowed,
                preserve_stage_scope=reuse_allowed,
                invalidate_on=[] if reuse_allowed else ["reground_required"],
            ),
        )

    @staticmethod
    def _repair_global_intent_payload(
        result: Dict[str, Any],
        *,
        projected_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recover known Main/IEA ownership leaks before strict validation.

        The Main agent sometimes places IEA-owned flow fields on a semantic
        target, confuses uncertainty flags with investigation targets, or
        nests Main decision fields under control_semantics.  Compiler-owned
        artifact fields are discarded; decision fields are promoted before
        strict validation.
        """
        structured = result.get("structured_response")
        if isinstance(structured, BaseModel):
            structured = structured.model_dump(mode="json", exclude_unset=True)
        if not isinstance(structured, (dict, str)):
            return result
        if isinstance(structured, str):
            try:
                structured = json.loads(structured)
            except json.JSONDecodeError:
                return result
        if not isinstance(structured, dict):
            return result

        repaired = deepcopy(structured)
        for field_name in MainControlAgent._COMPILED_FIELDS:
            if field_name in repaired:
                repaired.pop(field_name)
                if projected_fields is not None:
                    projected_fields.append(field_name)
        semantics = repaired.get("control_semantics")
        if isinstance(semantics, dict):
            for field_name in MainControlAgent._CONTROL_SEMANTICS_LEAK_FIELDS:
                if field_name not in semantics:
                    continue
                leaked_value = semantics.pop(field_name)
                if field_name in MainControlAgent._DECISION_FIELDS and field_name not in repaired:
                    repaired[field_name] = leaked_value
                if projected_fields is not None:
                    projected_fields.append(field_name)
        for stage in (repaired.get("control_semantics") or {}).get("stages") or []:
            if not isinstance(stage, dict):
                continue
            for target in stage.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                for field_name in ("app_id", "flow_id", "matched_flow_ids", "resolution_status"):
                    target.pop(field_name, None)

        investigation_aliases = {
            "domain_ambiguous": "domain_boundary",
            "identifier_risk": "ue_binding",
            "runtime_evidence_missing": "assurance_gap",
            "execution_feedback_incomplete": "assurance_gap",
            "conflict_signal_present": "cross_domain_consistency",
        }
        allowed_investigations = {
            "domain_boundary",
            "ue_binding",
            "qos_flow_binding",
            "mobility_target_binding",
            "policy_feasibility",
            "cross_domain_consistency",
            "assurance_gap",
        }
        investigations = repaired.get("investigation_targets")
        if isinstance(investigations, list):
            repaired["investigation_targets"] = list(
                dict.fromkeys(
                    normalized
                    for item in investigations
                    if (
                        normalized := investigation_aliases.get(str(item or "").strip(), str(item or "").strip())
                    )
                    in allowed_investigations
                )
            )

        normalized_result = dict(result)
        normalized_result["structured_response"] = repaired
        return normalized_result

    @staticmethod
    def _validate_global_intent(
        intent: GlobalControlIntent,
        *,
        user_input: str,
        context: str = "",
    ) -> List[str]:
        errors: List[str] = []
        parsed_context = MainControlAgent._parse_context_payload(context)
        try:
            round_index = int(parsed_context.get("round_index") or 1)
        except (TypeError, ValueError):
            round_index = 1
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
        domain_values = {item.value for item in intent.requested_domains}
        if intent.serving_slice_change_requested and domain_values != {"qos", "mobility"}:
            errors.append("serving_slice_change_requested requires requested_domains=[qos,mobility]")
        explicit_supis = re.findall(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        unique_explicit_supis = list(dict.fromkeys(explicit_supis))
        if len(unique_explicit_supis) == 1:
            explicit_supi = unique_explicit_supis[0]
            if str(intent.supi or "").strip() != explicit_supi:
                errors.append(f"supi must equal explicit user-provided identifier {explicit_supi}")
        elif len(unique_explicit_supis) > 1 and str(intent.supi or "").strip():
            errors.append("top-level supi must not comma-join multiple identifiers; leave it empty for multi-SUPI requests")
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
        next_agent = str(intent.next_agent or "").strip()
        if next_agent == "intent_encoding" and not str(intent.intent_encoding_guidance or "").strip() and round_index > 1:
            errors.append("retry routing into intent_encoding requires explicit intent_encoding_guidance")
        if next_agent != "intent_encoding" and str(intent.intent_encoding_guidance or "").strip():
            errors.append("intent_encoding_guidance must stay empty unless next_agent=intent_encoding")
        if len(unique_explicit_supis) == 1:
            explicit_supi = unique_explicit_supis[0]
            for stage_index, stage in enumerate(intent.control_semantics.stages or [], start=1):
                for target_index, target in enumerate(stage.targets or [], start=1):
                    if not str(target.supi or "").strip():
                        errors.append(
                            "single-SUPI semantic target must carry target.supi "
                            f"stage {stage_index} target {target_index}: expected {explicit_supi}"
                        )
        # 关键步骤：诊断类别只作为上下文交给 LLM，不在这里做域级硬裁决。
        return errors

    @staticmethod
    def _parse_context_payload(context: str) -> Dict[str, Any]:
        text = str(context or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

        payload: Dict[str, Any] = {}
        round_match = re.search(r"(?im)^\s*-\s*round_index:\s*(\d+)\s*$", text)
        if round_match:
            payload["round_index"] = int(round_match.group(1))
        category_match = re.search(r"(?im)^\s*-\s*root_cause_category:\s*(.+?)\s*$", text)
        if category_match:
            payload["previous_diagnosis"] = {"root_cause_category": category_match.group(1).strip()}
        bindings_match = re.search(r"(?im)^\s*-\s*flow_bindings:\s*(\[.+\])\s*$", text)
        if bindings_match:
            try:
                bindings = json.loads(bindings_match.group(1))
            except Exception:
                bindings = []
            if isinstance(bindings, list):
                payload["previous_grounding_decision"] = {"flow_bindings": bindings}
        return payload

    @staticmethod
    def _normalize_semantic_target_scope(
        intent: GlobalControlIntent,
        *,
        user_input: str,
    ) -> None:
        explicit_supis = re.findall(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        unique_explicit_supis = list(dict.fromkeys(explicit_supis))
        default_supi = ""
        if len(unique_explicit_supis) == 1:
            default_supi = unique_explicit_supis[0]
        elif len(unique_explicit_supis) == 0:
            default_supi = str(intent.supi or "").strip()
        if not default_supi:
            return
        for stage in intent.control_semantics.stages or []:
            for target in stage.targets or []:
                if not str(target.supi or "").strip():
                    target.supi = default_supi
