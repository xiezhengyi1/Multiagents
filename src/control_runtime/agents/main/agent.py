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
    ControlSemanticMode,
    ControlSemantics,
    ControlStage,
    GlobalControlIntent,
    MainRetryScope,
    MainRoundStrategy,
    SemanticGoal,
    SemanticTarget,
    SemanticTargetType,
    StageTrigger,
)
from shared.logging import log_event

from .prompts import MAIN_CONTROL_SYSTEM_PROMPT


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
    LEGACY_GROUNDING_TOOLS = set()
    GROUNDING_TOOLS = set()

    def __init__(self, model_name: str = "qwen-plus", use_local_model: bool = False) -> None:
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
    ) -> GlobalControlIntent:
        self.ensure_worker_runtime_initialized()
        log_event(self.logger, "main_control_start")
        payload = {
            "role": "user",
            "content": (
                f"User input:\n{user_input}\n\n"
                f"Coordinator context:\n{context or 'N/A'}\n\n"
                "Resolve only the round-level domain routing, retry scope, explicit SUPI already present in the request, and keep intent_encoding_guidance empty unless non-empty routing guidance is strictly necessary."
            ),
        }
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            thread_id=session_id,
        )
        self._pending_invoke_messages = [payload]
        self._pending_trace_metadata = {
            **(getattr(self, "_pending_trace_metadata", {}) or {}),
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
            if hasattr(self, "_pending_trace_metadata"):
                delattr(self, "_pending_trace_metadata")

    def _invoke_global_intent_result(self, user_prompt: str, *, runtime_context: Any) -> MainControlInvocation:
        self._pending_invoke_messages = [{"role": "user", "content": user_prompt}]
        payload = {
            "messages": self._pending_invoke_messages,
            "trace_write_mode": "manual",
            "trace_metadata": getattr(self, "_pending_trace_metadata", {}) or {},
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

    @staticmethod
    def _build_validation_retry_prompt(*, base_prompt: str, validation_errors: List[str], invocation_error: str) -> str:
        issues: List[str] = []
        if invocation_error:
            issues.append(invocation_error)
        if validation_errors:
            issues.extend(validation_errors)
        return (
            f"{base_prompt}\n\n"
            "Your previous draft failed validation.\n"
            "Validation errors:\n- " + "\n- ".join(issues) + "\n\n"
            "Return a corrected GlobalControlIntent JSON only."
        )

    @staticmethod
    def _validate_global_intent_result(result: Dict[str, Any]) -> GlobalControlIntent:
        try:
            intent = coerce_structured_response(
                result,
                GlobalControlIntent,
                error_message="Main Agent returned no structured_response",
            )
            intent.intent_encoding_guidance = ""
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
            "policy_repair",
            "execution_retry_forbidden",
            "",
        }
        round_index = 0
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
        explicit_supi = ""
        match = re.search(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        if match:
            explicit_supi = match.group(1)
        if explicit_supi and str(intent.supi or "").strip() != explicit_supi:
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
            errors.append("round-1 main routing must not request retry scopes other than full_reground")
        next_agent = str(intent.next_agent or "").strip()
        if next_agent == "optimization_strategy" and retry_scope != "policy_repair":
            errors.append("next_agent=optimization_strategy requires retry_scope=policy_repair")
        if retry_scope == "policy_repair" and next_agent != "optimization_strategy":
            errors.append("retry_scope=policy_repair requires next_agent=optimization_strategy")
        # 关键步骤：诊断类别只作为上下文交给 LLM，不在这里做域级硬裁决。
        return errors

    @staticmethod
    def _enrich_global_intent_contract(
        intent: GlobalControlIntent,
        *,
        context: str,
        user_input: str,
    ) -> None:
        try:
            payload = json.loads(str(context or "").strip()) if str(context or "").strip() else {}
        except Exception:
            payload = {}

        if intent.retry_scope is None:
            if intent.round_strategy == MainRoundStrategy.INITIAL_GROUNDING:
                intent.retry_scope = None
            elif intent.round_strategy == MainRoundStrategy.REGROUNDING:
                intent.retry_scope = MainRetryScope.FULL_REGROUND
            elif intent.round_strategy == MainRoundStrategy.POLICY_REVISION:
                intent.retry_scope = MainRetryScope.POLICY_REPAIR
            elif intent.round_strategy == MainRoundStrategy.JOINT_REPLAN:
                intent.retry_scope = MainRetryScope.PARTIAL_REGROUND

        if not intent.diagnosis_summary:
            if isinstance(payload, dict):
                diagnosis = payload.get("previous_diagnosis")
                if isinstance(diagnosis, dict):
                    intent.diagnosis_summary = str(
                        diagnosis.get("reason_summary")
                        or diagnosis.get("root_cause")
                        or ""
                    ).strip()
        intent.control_semantics = MainControlAgent._derive_control_semantics(user_input)
        intent.intent_encoding_guidance = MainControlAgent._derive_intent_encoding_guidance(intent)

    @staticmethod
    def _derive_control_semantics(user_input: str) -> ControlSemantics:
        text = str(user_input or "").strip()
        if not text:
            return ControlSemantics()

        primary_text = text
        fallback_text = ""
        secondary_text = ""
        final_text = ""
        mode = ControlSemanticMode.SINGLE_STEP

        fallback_match = re.search(r"(必要时|失败时|若失败则|如果失败就|不行就)", text)
        if fallback_match:
            mode = ControlSemanticMode.CONDITIONAL_FALLBACK
            primary_text = text[: fallback_match.start()].strip(" ，,。；;")
            fallback_text = text[fallback_match.end() :].strip(" ，,。；;")
        elif "再看" in text or "然后看" in text:
            mode = ControlSemanticMode.STAGED_PRIORITY
            splitter = "再看" if "再看" in text else "然后看"
            first, second = text.split(splitter, 1)
            primary_text = first.strip(" ，,。；;")
            secondary_text = second.strip(" ，,。；;")
        final_source = secondary_text or fallback_text or primary_text or text
        if "最后处理" in final_source:
            before, after = final_source.rsplit("最后处理", 1)
            if before.strip():
                if secondary_text:
                    secondary_text = before.strip(" ，,。；;")
                elif fallback_text:
                    fallback_text = before.strip(" ，,。；;")
                elif mode == ControlSemanticMode.SINGLE_STEP:
                    mode = ControlSemanticMode.STAGED_PRIORITY
                    primary_text = before.strip(" ，,。；;")
            final_text = after.strip(" ，,。；;")
        elif "其余业务延后" in text or "其余业务最后处理" in text:
            if mode == ControlSemanticMode.SINGLE_STEP:
                mode = ControlSemanticMode.STAGED_PRIORITY
            final_text = "其余业务"

        stages: List[ControlStage] = []
        primary_stage = MainControlAgent._build_semantic_stage(
            stage_index=1,
            name="primary",
            trigger=StageTrigger.INITIAL,
            clause=primary_text,
            default_goal=SemanticGoal.PROTECT,
        )
        if primary_stage.targets:
            stages.append(primary_stage)

        if fallback_text:
            fallback_stage = MainControlAgent._build_semantic_stage(
                stage_index=len(stages) + 1,
                name="fallback",
                trigger=StageTrigger.ON_PREVIOUS_FAILURE,
                clause=fallback_text,
                default_goal=SemanticGoal.DEPRIORITIZE,
            )
            if primary_stage.targets:
                fallback_stage.targets = [target.model_copy(deep=True) for target in primary_stage.targets] + fallback_stage.targets
            if fallback_stage.targets:
                stages.append(fallback_stage)

        if secondary_text:
            secondary_stage = MainControlAgent._build_semantic_stage(
                stage_index=len(stages) + 1,
                name="secondary",
                trigger=StageTrigger.AFTER_PREVIOUS_STAGE,
                clause=secondary_text,
                default_goal=SemanticGoal.PROTECT,
            )
            if secondary_stage.targets:
                stages.append(secondary_stage)

        if final_text:
            final_stage = MainControlAgent._build_semantic_stage(
                stage_index=len(stages) + 1,
                name="deprioritized",
                trigger=StageTrigger.AFTER_PREVIOUS_STAGE,
                clause=final_text,
                default_goal=SemanticGoal.DEFER if "其余业务" in final_text else SemanticGoal.DEPRIORITIZE,
            )
            if final_stage.targets:
                stages.append(final_stage)

        if not stages:
            single_stage = MainControlAgent._build_semantic_stage(
                stage_index=1,
                name="primary",
                trigger=StageTrigger.INITIAL,
                clause=text,
                default_goal=SemanticGoal.PROTECT,
            )
            if single_stage.targets:
                stages.append(single_stage)

        if len(stages) <= 1:
            mode = ControlSemanticMode.SINGLE_STEP
        return ControlSemantics(mode=mode, current_stage=1, stages=stages)

    @staticmethod
    def _build_semantic_stage(
        *,
        stage_index: int,
        name: str,
        trigger: StageTrigger,
        clause: str,
        default_goal: SemanticGoal,
    ) -> ControlStage:
        targets = MainControlAgent._extract_semantic_targets(clause, default_goal=default_goal)
        summary = clause.strip()
        return ControlStage(
            stage_index=stage_index,
            name=name,
            trigger=trigger,
            summary=summary,
            targets=targets,
        )

    @staticmethod
    def _extract_semantic_targets(clause: str, *, default_goal: SemanticGoal) -> List[SemanticTarget]:
        text = str(clause or "").strip()
        if not text:
            return []
        metric_focus = MainControlAgent._detect_metric_focus(text)
        goal = MainControlAgent._detect_goal(text, default_goal=default_goal)
        entities = MainControlAgent._extract_named_entities(text)
        if not entities and "其余业务" in text:
            entities = [("其余业务", SemanticTargetType.SCOPE)]
        targets: List[SemanticTarget] = []
        for entity_name, target_type in entities:
            targets.append(
                SemanticTarget(
                    semantic_name=entity_name,
                    target_type=target_type,
                    goal=goal,
                    metric_focus=metric_focus,
                    note=text,
                )
            )
        return targets

    @staticmethod
    def _extract_named_entities(text: str) -> List[tuple[str, SemanticTargetType]]:
        token_pattern = r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b"
        raw_tokens = [token.strip() for token in re.findall(token_pattern, text) if token.strip()]
        results: List[tuple[str, SemanticTargetType]] = []
        seen: set[str] = set()
        for token in raw_tokens:
            normalized = token.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            token_type = SemanticTargetType.FLOW if re.search(r"_\d+$", token) else SemanticTargetType.APP
            results.append((token, token_type))
        return results

    @staticmethod
    def _detect_metric_focus(text: str) -> Optional[str]:
        lowered = str(text or "").lower()
        if any(token in lowered for token in ("时延", "延迟", "latency", "delay")):
            return "latency"
        if any(token in lowered for token in ("吞吐", "带宽", "throughput", "bandwidth", "资源")):
            return "throughput"
        if any(token in lowered for token in ("抖动", "jitter", "稳定")):
            return "jitter"
        if any(token in lowered for token in ("丢包", "可靠", "loss", "reliability")):
            return "reliability"
        return None

    @staticmethod
    def _detect_goal(text: str, *, default_goal: SemanticGoal) -> SemanticGoal:
        lowered = str(text or "").lower()
        if any(token in lowered for token in ("压低", "降", "降低", "牺牲", "最后处理", "deprioritize")):
            return SemanticGoal.DEPRIORITIZE
        if any(token in lowered for token in ("延后", "推迟", "defer")):
            return SemanticGoal.DEFER
        if any(token in lowered for token in ("观察", "监控", "observe")):
            return SemanticGoal.OBSERVE
        return default_goal

    @staticmethod
    def _derive_intent_encoding_guidance(intent: GlobalControlIntent) -> str:
        next_agent = str(intent.next_agent or "").strip()
        if next_agent != "intent_encoding":
            return ""

        requested_domains = [item.value for item in (intent.requested_domains or [])]
        requested_set = set(requested_domains)
        retry_scope = (
            intent.retry_scope.value
            if getattr(intent, "retry_scope", None) is not None and hasattr(intent.retry_scope, "value")
            else str(getattr(intent, "retry_scope", "") or "").strip()
        )

        if requested_set == {"qos"}:
            if retry_scope == "partial_reground":
                return "preserve qos-only domain boundary and reground the qos target binding"
            return "preserve qos-only domain boundary and ground the requested qos target"
        if requested_set == {"mobility"}:
            if retry_scope == "partial_reground":
                return "preserve mobility-only domain boundary and reground the mobility target binding"
            return "preserve mobility-only domain boundary and ground the requested mobility target"
        if requested_set == {"qos", "mobility"}:
            if retry_scope == "partial_reground":
                return "preserve joint domain boundary and reground cross-domain target bindings"
            return "preserve joint domain boundary and ground qos and mobility targets consistently"
        return ""
