from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime import ArtifactEnvelope
from agents.BaseAgent import BaseAgent, coerce_structured_response, extract_grounding_tool_names
from agents.tools.user_interaction_tool import ask_user_clarification
from agents.worker import ArtifactWorkerMixin
from agents.tools import (
    get_am_policy_context,
    get_knowledge_by_key,
    get_sm_ue_context,
    get_sm_ue_flow_catalog,
    search_am_policy_targets,
    search_semantic_knowledge,
    search_sm_flow_targets,
    think,
)
from domain.control_plane import ControlDomain
from domain.control_plane import GlobalControlIntent
from utils.logger import log_event, log_timing

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
    SM_GROUNDING_TOOLS = {"search_sm_flow_targets", "get_sm_ue_context", "get_sm_ue_flow_catalog"}
    AM_GROUNDING_TOOLS = {"get_am_policy_context", "search_am_policy_targets"}
    LEGACY_GROUNDING_TOOLS = {"search_flow_targets_by_name", "get_ue_context", "get_ue_flow_catalog"}
    GROUNDING_TOOLS = {
        *SM_GROUNDING_TOOLS,
        *AM_GROUNDING_TOOLS,
        *LEGACY_GROUNDING_TOOLS,
        "get_knowledge_by_key",
        "search_semantic_knowledge",
    }

    def __init__(self, model_name: str = "qwen-plus", use_local_model: bool = False) -> None:
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        self.agent_name = "main_control"
        self.initialize_agent_runtime(logger_color="\033[93m")
        self.tools = [
            ask_user_clarification,
            search_sm_flow_targets,
            get_sm_ue_context,
            get_sm_ue_flow_catalog,
            get_am_policy_context,
            search_am_policy_targets,
            get_knowledge_by_key,
            search_semantic_knowledge,
        ]
        self.agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=MAIN_CONTROL_SYSTEM_PROMPT,
            response_model=GlobalControlIntent,
            max_iterations=14,
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
                "Resolve only the round-level domain routing, retry scope, explicit identifiers already present in the request, and routing-level prompt injections."
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
            for attempt_index in range(3):
                invocation = self._invoke_global_intent_result(
                    current_prompt,
                    runtime_context=runtime_context,
                )
                grounding_tools = extract_grounding_tool_names(invocation.raw_result, self.GROUNDING_TOOLS)
                parsed_intent = self._validate_global_intent_result(invocation.raw_result)
                if self._grounding_gate_failed(
                    user_input=user_input,
                    context=context,
                    grounding_tools=grounding_tools,
                ):
                    if attempt_index == 2:
                        raise RuntimeError("Main Agent failed mandatory grounding-tool gate after repeated retries.")
                    current_prompt = self._build_grounding_retry_prompt(
                        base_prompt=payload["content"],
                        validation_errors=["grounding tool use is required for this request but no non-think grounding tool was called"],
                        user_input=user_input,
                    )
                    continue
                intent = parsed_intent
                validation_errors = self._validate_global_intent(
                    intent,
                    user_input=user_input,
                    context=context,
                    grounding_tools=grounding_tools,
                )
                if not validation_errors:
                    break
                if attempt_index == 2:
                    raise RuntimeError(
                        "Main Agent could not produce a valid GlobalControlIntent: "
                        + "; ".join(validation_errors)
                    )
                current_prompt = self._build_validation_retry_prompt(
                    base_prompt=payload["content"],
                    validation_errors=validation_errors,
                )
            if intent is None:
                raise RuntimeError("Main Agent returned no intent payload")
            if not intent.session_id:
                intent.session_id = session_id
            if not intent.snapshot_id:
                intent.snapshot_id = snapshot_id
            if not intent.raw_input:
                intent.raw_input = user_input
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
            repair_prompt = (
                f"{user_prompt}\n\n"
                "Your previous response was not valid JSON for GlobalControlIntent.\n"
                f"Parser error: {exc}\n\n"
                "Return only one JSON object that matches GlobalControlIntent. Do not include explanations, bullets, or markdown."
            )
            self._pending_invoke_messages = [{"role": "user", "content": repair_prompt}]
            payload = {
                "messages": self._pending_invoke_messages,
                "trace_write_mode": "manual",
                "trace_metadata": getattr(self, "_pending_trace_metadata", {}) or {},
            }
            result = self.agent.invoke(payload, context=runtime_context)
        return MainControlInvocation(
            raw_result=result,
            trace_agent=self.agent,
            trace_payload=payload,
            runtime_context=runtime_context,
        )

    @staticmethod
    def _recommended_grounding_tools(user_input: str) -> List[str]:
        recommended: List[str] = []
        normalized = str(user_input or "")
        lowered = normalized.lower()
        has_explicit_supi = bool(re.search(r"(?i)imsi-\d{5,}", normalized))
        has_explicit_object_id = bool(re.search(r"(?i)\b(app-\d+|flow-\d+)\b", normalized))
        has_named_flow_signal = bool(re.search(r"\b[A-Za-z]+(?:[_-][A-Za-z0-9]+)+\b", normalized))
        has_policy_object_signal = any(
            token in lowered
            for token in (
                "allowed nssai",
                "target nssai",
                "service area",
                "rfsp",
                "amf",
                "am policy",
                "ue policy",
                "npcf_",
            )
        )

        # 关键步骤：这里只给出“证据抓取”建议，不在代码里根据关键词替 LLM 做域划分。
        if has_explicit_supi:
            recommended.extend(["get_sm_ue_flow_catalog", "get_sm_ue_context"])
        if has_named_flow_signal or has_explicit_object_id:
            recommended.extend(["get_sm_ue_flow_catalog", "search_sm_flow_targets"])
        if has_explicit_supi and has_policy_object_signal:
            recommended.append("get_am_policy_context")
        if has_policy_object_signal:
            recommended.extend(["search_am_policy_targets", "get_knowledge_by_key", "search_semantic_knowledge"])
        deduped: List[str] = []
        for item in recommended:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @classmethod
    def _build_grounding_retry_prompt(
        cls,
        *,
        base_prompt: str,
        validation_errors: List[str],
        user_input: str,
    ) -> str:
        recommended_tools = cls._recommended_grounding_tools(user_input)
        tool_line = ", ".join(recommended_tools) if recommended_tools else "get_sm_ue_context or get_sm_ue_flow_catalog"
        return (
            f"{base_prompt}\n\n"
            "Your previous draft failed validation because it did not use the required grounding tools.\n"
            "Validation errors:\n- " + "\n- ".join(validation_errors) + "\n\n"
            f"Before returning JSON, call at least one applicable non-think grounding tool first. Recommended tools: {tool_line}.\n"
            "After the tool result arrives, return one corrected GlobalControlIntent JSON object only."
        )

    @staticmethod
    def _build_validation_retry_prompt(*, base_prompt: str, validation_errors: List[str]) -> str:
        return (
            f"{base_prompt}\n\n"
            "Your previous draft failed validation.\n"
            "Validation errors:\n- " + "\n- ".join(validation_errors) + "\n\n"
            "Return a corrected GlobalControlIntent JSON only."
        )

    @staticmethod
    def _grounding_gate_failed(
        *,
        user_input: str,
        context: str,
        grounding_tools: List[str],
    ) -> bool:
        if not MainControlAgent._requires_grounding_tools(user_input=user_input, context=context):
            return False
        return not grounding_tools

    @staticmethod
    def _validate_global_intent_result(result: Dict[str, Any]) -> GlobalControlIntent:
        return coerce_structured_response(
            result,
            GlobalControlIntent,
            error_message="Main Agent returned no structured_response",
        )

    @staticmethod
    def _requires_grounding_tools(*, user_input: str, context: str) -> bool:
        normalized_input = str(user_input or "")
        lowered = normalized_input.lower()
        if re.search(r"(?i)imsi-\d{5,}", normalized_input):
            return True
        if re.search(r"(?i)\b(app-\d+|flow-\d+)\b", normalized_input):
            return True
        if re.search(r"\b[A-Za-z]+(?:[_-][A-Za-z0-9]+)+\b", normalized_input):
            return True
        if any(token in lowered for token in ("allowed nssai", "target nssai", "service area", "rfsp", "amf", "am policy", "ue policy", "npcf_")):
            return True
        try:
            payload = json.loads(str(context or ""))
        except Exception:
            payload = {}
        if isinstance(payload, dict) and int(payload.get("round_index") or 1) > 1:
            return True
        return False

    @staticmethod
    def _validate_global_intent(
        intent: GlobalControlIntent,
        *,
        user_input: str,
        context: str = "",
        grounding_tools: Optional[List[str]] = None,
    ) -> List[str]:
        errors: List[str] = []
        if not intent.requested_domains:
            errors.append("requested_domains is empty")
        else:
            values = [item.value for item in intent.requested_domains]
            if any(item not in {"qos", "mobility"} for item in values):
                errors.append(f"requested_domains contains unsupported values: {values}")
        explicit_supi = ""
        match = re.search(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        if match:
            explicit_supi = match.group(1)
        if explicit_supi and str(intent.supi or "").strip() != explicit_supi:
            errors.append(f"supi must equal explicit user-provided identifier {explicit_supi}")
        explicit_app_ids = re.findall(r"(?i)\b(app-\d+)\b", str(user_input or ""))
        normalized_app_id = str(intent.app_id or "").strip()
        if explicit_app_ids:
            if normalized_app_id != explicit_app_ids[0]:
                errors.append(f"app_id must equal explicit user-provided identifier {explicit_app_ids[0]}")
        elif normalized_app_id:
            errors.append("Main Agent must not resolve app_id without an explicit app identifier in user input")
        explicit_flow_ids = re.findall(r"(?i)\b(flow-\d+)\b", str(user_input or ""))
        normalized_flow_ids = [str(item or "").strip() for item in (intent.target_flow_ids or []) if str(item or "").strip()]
        if explicit_flow_ids:
            if sorted(set(normalized_flow_ids)) != sorted(set(explicit_flow_ids)):
                errors.append(f"target_flow_ids must match explicit user-provided identifiers {sorted(set(explicit_flow_ids))}")
        elif normalized_flow_ids:
            errors.append("Main Agent must not resolve target_flow_ids without explicit flow identifiers in user input")
        if str(intent.app_name or "").strip():
            errors.append("Main Agent must not populate app_name; IEA owns app-name resolution")
        if any(str(item or "").strip() for item in (intent.target_flow_names or [])):
            errors.append("Main Agent must not populate target_flow_names; IEA owns flow-name resolution")
        if list(intent.mobility_triggers or []):
            errors.append("Main Agent must not populate mobility_triggers; IEA owns AM/mobility semantic resolution")
        if not isinstance(intent.domain_evidence, dict) or not any(intent.domain_evidence.values()):
            errors.append("domain_evidence must not be empty")
        else:
            unknown_domains = [key for key in intent.domain_evidence.keys() if key not in {"qos", "mobility"}]
            if unknown_domains:
                errors.append(f"domain_evidence contains unsupported keys: {unknown_domains}")
            requested = {item.value for item in intent.requested_domains}
            evidence_keys = {str(key).strip().lower() for key, values in intent.domain_evidence.items() if values}
            if not requested.issubset(evidence_keys):
                errors.append(
                    f"domain_evidence must cover every requested domain: requested={sorted(requested)} evidence={sorted(evidence_keys)}"
                )
        semantic_injection_tokens = (
            "allowedsnssais",
            "targetsnssais",
            "rfsp",
            "service area",
            "guami",
            "5qi",
            "arp",
            "pcc",
            "smpolicydecision",
            "ursp",
            "npcf_",
            "flow-",
            "app-",
        )
        for key, value in (intent.prompt_injections or {}).items():
            lowered_text = str(value or "").strip().lower()
            if any(token in lowered_text for token in semantic_injection_tokens):
                errors.append(
                    f"prompt_injections[{key}] must stay at routing level and must not contain semantic policy/object details"
                )
        if MainControlAgent._requires_grounding_tools(user_input=user_input, context=context):
            if not grounding_tools:
                errors.append("grounding tool use is required for this request but no non-think grounding tool was called")
        # 关键步骤：诊断类别只作为上下文交给 LLM，不在这里做域级硬裁决。
        return errors
