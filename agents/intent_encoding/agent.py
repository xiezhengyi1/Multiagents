from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set

from agent_runtime import ArtifactEnvelope
from agents.BaseAgent import BaseAgent
from agents.worker import ArtifactWorkerMixin
from domain.policy_plan import FlowSelector, OperationIntent
from agents.tools.db_tool import get_ue_flow_catalog_by_supi
from agents.tools.knowledge_tool import get_knowledge_by_key, search_semantic_knowledge
from agents.tools.pcf_tools import get_ue_context, get_ue_flow_catalog
from agents.tools import ask_user_clarification, search_flow_targets_by_name, think
from utils.logger import log_event, log_timing

from .prompts import IEA_SYSTEM_PROMPT


class IntentEncodingAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "intent_encoding"

    FLOW_KEYWORDS: Dict[str, Set[str]] = {
        "video": {"video", "stream", "视频"},
        "control": {"control", "控制"},
        "telemetry": {"telemetry", "遥测"},
    }
    UL_KEYWORDS: Set[str] = {"ul", "uplink", "upstream", "上行", "上传"}
    DL_KEYWORDS: Set[str] = {"dl", "downlink", "downstream", "下行", "下载"}
    DELETE_KEYWORDS: Set[str] = {"delete", "remove", "删除", "移除"}
    ADD_KEYWORDS: Set[str] = {"add", "create", "新增", "添加"}
    MODIFY_KEYWORDS: Set[str] = {"modify", "update", "change", "调整", "修改", "变更", "降低", "提升"}

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.agent_name = "intent_encoding"
        self.initialize_agent_runtime(logger_color="\033[95m")
        self.tools = [
            think,
            ask_user_clarification,
            search_flow_targets_by_name,
            get_ue_context,
            get_ue_flow_catalog,
            search_semantic_knowledge,
            get_knowledge_by_key,
        ]
        self.agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=IEA_SYSTEM_PROMPT,
            response_model=OperationIntent,
        )

    def expected_request_type(self) -> str:
        return "OperationIntentRequest"

    def response_artifact_type(self) -> str:
        return "OperationIntent"

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

    @classmethod
    def _infer_flow_keyword(cls, *texts: Any) -> str:
        normalized_text = " ".join(str(text or "").lower() for text in texts)
        for canonical, aliases in cls.FLOW_KEYWORDS.items():
            if any(alias in normalized_text for alias in aliases):
                return canonical
        return ""

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
        if conversation_messages is not None:
            analyze_kwargs["conversation_messages"] = conversation_messages

        operation_intent = self.analyze_intent(
            user_input,
            **analyze_kwargs,
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
        conversation_messages: List[Dict[str, Any]] | None = None,
        *,
        session_id: str = "",
        snapshot_id: str = "",
        allow_user_interaction: bool = False,
    ) -> OperationIntent:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "iea_analyze_start")

        try:
            runtime_context = self.build_runtime_context(
                agent_name=self.agent_name,
                session_id=session_id,
                snapshot_id=snapshot_id,
                supi=None,
                thread_id=session_id,
                allow_user_interaction=allow_user_interaction,
            )
            messages = self._build_conversation_messages(
                user_input=user_input,
                context=context,
                conversation_messages=conversation_messages,
                allow_user_interaction=allow_user_interaction,
            )
            self._pending_invoke_messages = messages
            structured = self.invoke_json_response(
                system_prompt=IEA_SYSTEM_PROMPT,
                user_prompt=messages[-1]["content"],
                response_model=OperationIntent,
                runtime_context=runtime_context,
            )
            normalized = self._postprocess_operation_intent(
                OperationIntent.model_validate(structured),
                user_input=user_input,
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
            return normalized
        except Exception as exc:
            self.logger.error(f"Failed to analyze operation intent: {exc}")
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")

    @staticmethod
    def _build_conversation_messages(
        *,
        user_input: str,
        context: str,
        conversation_messages: List[Dict[str, Any]] | None,
        allow_user_interaction: bool,
    ) -> List[Dict[str, str]]:
        if conversation_messages:
            normalized: List[Dict[str, str]] = []
            for message in conversation_messages:
                if not isinstance(message, dict):
                    raise TypeError("conversation_messages items must be dicts")
                role = str(message.get("role") or "").strip().lower() or "user"
                content = str(message.get("content") or "")
                if not content.strip():
                    continue
                normalized.append({"role": role, "content": content})

            if not normalized:
                raise ValueError("conversation_messages must contain at least one non-empty message")

            last_role = normalized[-1]["role"]
            if last_role != "user":
                raise ValueError("conversation_messages must end with a user message")

            normalized[-1] = {
                "role": "user",
                "content": (
                    f"{normalized[-1]['content']}\n\n"
                    f"Conversation context:\n{context or 'N/A'}\n\n"
                    f"Interactive clarification available: {'yes' if allow_user_interaction else 'no'}\n\n"
                    "Use tools when entity resolution depends on live UE context or semantic knowledge."
                ),
            }
            return normalized

        return [
            {
                "role": "user",
                "content": (
                    f"User input:\n{user_input}\n\n"
                    f"Conversation context:\n{context or 'N/A'}\n\n"
                    f"Interactive clarification available: {'yes' if allow_user_interaction else 'no'}\n\n"
                    "Use tools when entity resolution depends on live UE context or semantic knowledge."
                ),
            }
        ]

    def _resolve_single_flow(
        self,
        flow: FlowSelector,
        result: OperationIntent,
        user_input: str,
        app_catalog: List[Dict[str, Any]],
        flow_catalog: List[Dict[str, Any]],
    ) -> Optional[str]:
        def normalize_label(value: Any) -> str:
            return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").strip().lower())

        def extract_id(text: Any, prefix: str) -> Optional[str]:
            match = re.search(rf"(?i)\b{prefix}-\d{{4,8}}\b", str(text or ""))
            return match.group(0) if match else None

        def format_candidate(item: Dict[str, Any]) -> str:
            return (
                f"{item.get('app_name', '')}/{item.get('flow_name', '')} "
                f"({item.get('app_id', '')}/{item.get('flow_id', '')})"
            ).strip()

        def has_flow_keyword(flow_name: Any, keyword: str, app_name: Any) -> bool:
            aliases = self.FLOW_KEYWORDS.get(keyword, {keyword})
            normalized_name = str(flow_name or "").lower()
            normalized_app = str(app_name or "").lower()
            for separator in ("_", "-", " "):
                prefix = f"{normalized_app}{separator}"
                if normalized_name.startswith(prefix):
                    normalized_name = normalized_name[len(prefix) :]
                    break
            return any(alias in normalized_name for alias in aliases)

        explicit_flow_id = str(flow.flow_id or "").strip() or extract_id(flow.name, "flow") or extract_id(user_input, "flow")
        explicit_app_id = (
            str(flow.app_id or "").strip()
            or str(result.app_id or "").strip()
            or extract_id(result.app_name, "app")
            or extract_id(user_input, "app")
        )

        if explicit_app_id:
            exact_app_matches = [app for app in app_catalog if app.get("app_id") == explicit_app_id]
        elif result.app_name:
            exact_app_matches = [
                app
                for app in app_catalog
                if normalize_label(app.get("app_name")) == normalize_label(result.app_name)
            ]
        else:
            exact_app_matches = []

        no_keyword_match = False
        scope_candidates: List[Dict[str, Any]]

        if explicit_flow_id:
            matches = [item for item in flow_catalog if item.get("flow_id") == explicit_flow_id]
            scope_candidates = matches
        else:
            exact_flow_matches = []
            if flow.name:
                exact_flow_matches = [
                    item
                    for item in flow_catalog
                    if normalize_label(item.get("flow_name")) == normalize_label(flow.name)
                ]

            if exact_flow_matches:
                matches = exact_flow_matches
                scope_candidates = exact_flow_matches
            else:
                normalized_text = f"{flow.name} {user_input}".lower()
                keywords = {
                    canonical
                    for canonical, aliases in self.FLOW_KEYWORDS.items()
                    if any(alias in normalized_text for alias in aliases)
                }
                if len(exact_app_matches) == 1 and keywords:
                    matched_app = exact_app_matches[0]
                    app_scope = [item for item in flow_catalog if item.get("app_id") == matched_app.get("app_id")]
                    matches = [
                        item
                        for item in app_scope
                        if any(
                            has_flow_keyword(
                                item.get("flow_name"),
                                keyword,
                                matched_app.get("app_name"),
                            )
                            for keyword in keywords
                        )
                    ]
                    scope_candidates = matches if matches else app_scope
                    no_keyword_match = not matches
                else:
                    if len(exact_app_matches) == 1:
                        app_id = exact_app_matches[0].get("app_id")
                        scope_candidates = [item for item in flow_catalog if item.get("app_id") == app_id]
                    else:
                        scope_candidates = list(flow_catalog)
                    matches = list(scope_candidates) if len(scope_candidates) == 1 else []

        if len(matches) == 1:
            catalog_flow = matches[0]

            def to_float(value: Any) -> Optional[float]:
                if value in (None, ""):
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            current_ul = to_float(catalog_flow.get("current_bw_ul"))
            current_dl = to_float(catalog_flow.get("current_bw_dl"))
            requested_ul = to_float(flow.bw_ul)
            requested_dl = to_float(flow.bw_dl)
            normalized_input = user_input.lower()
            has_ul = any(keyword in normalized_input for keyword in self.UL_KEYWORDS)
            has_dl = any(keyword in normalized_input for keyword in self.DL_KEYWORDS)

            if has_ul and not has_dl:
                target_ul = requested_ul if requested_ul is not None else current_ul
                target_dl = current_dl
            elif has_dl and not has_ul:
                target_ul = current_ul
                target_dl = requested_dl if requested_dl is not None else current_dl
            elif (current_dl or 0.0) >= (current_ul or 0.0):
                target_ul = current_ul
                target_dl = requested_dl if requested_dl is not None else requested_ul if requested_ul is not None else current_dl
            else:
                target_dl = current_dl
                target_ul = requested_ul if requested_ul is not None else requested_dl if requested_dl is not None else current_ul

            flow.name = str(catalog_flow.get("flow_name") or flow.name or "")
            flow.app_id = str(catalog_flow.get("app_id") or "")
            flow.flow_id = str(catalog_flow.get("flow_id") or "")
            flow.service_type = catalog_flow.get("service_type")
            flow.service_type_id = catalog_flow.get("service_type_id")
            flow.bw_ul = target_ul
            flow.bw_dl = target_dl
            flow.gbr_ul = to_float(catalog_flow.get("gbr_ul"))
            flow.gbr_dl = to_float(catalog_flow.get("gbr_dl"))
            flow.lat = to_float(catalog_flow.get("lat"))
            flow.loss_req = to_float(catalog_flow.get("loss_req"))
            flow.jitter_req = to_float(catalog_flow.get("jitter_req"))
            flow.priority = int(catalog_flow["priority"]) if catalog_flow.get("priority") is not None else None
            flow.description = str(catalog_flow.get("flow_name") or flow.description or "")
            flow.current_bw_ul = current_ul
            flow.current_bw_dl = current_dl
            flow.five_tuple = (
                list(catalog_flow.get("five_tuple"))
                if isinstance(catalog_flow.get("five_tuple"), (list, tuple))
                else None
            )
            flow.resolution_status = "resolved"
            flow.resolution_candidates = [format_candidate(catalog_flow)]
            return str(catalog_flow.get("app_id") or "")

        flow.app_id = ""
        flow.flow_id = None
        flow.current_bw_ul = None
        flow.current_bw_dl = None
        flow.five_tuple = None
        flow.resolution_candidates = [format_candidate(item) for item in scope_candidates[:5]]
        if no_keyword_match:
            flow.resolution_status = "unmatched"
        elif len(matches) > 1 or len(scope_candidates) > 1:
            flow.resolution_status = "ambiguous"
        else:
            flow.resolution_status = "unmatched"
        return None

    def _resolve_operation_intent_against_catalog(
        self,
        result: OperationIntent,
        user_input: str,
        catalog: Optional[Dict[str, Any]] = None,
    ) -> OperationIntent:
        if not result.flows:
            result.flows = [FlowSelector(name="")]

        if not result.supi:
            result.resolution_status = "unmatched"
            result.app_id = ""
            for flow in result.flows:
                flow.resolution_status = "unmatched"
            return result

        catalog_payload = catalog if catalog is not None else get_ue_flow_catalog_by_supi(result.supi)
        app_catalog = catalog_payload.get("app_catalog") or []
        flow_catalog = catalog_payload.get("flow_catalog") or []
        if not flow_catalog:
            result.resolution_status = "unmatched"
            result.app_id = ""
            for flow in result.flows:
                flow.resolution_status = "unmatched"
            return result

        resolved_app_ids: Set[str] = set()
        resolved_app_names: Set[str] = set()
        flow_statuses: List[str] = []

        for flow in result.flows:
            flow.supi = result.supi
            app_id = self._resolve_single_flow(flow, result, user_input, app_catalog, flow_catalog)
            if app_id:
                resolved_app_ids.add(app_id)
                app_match = next((app for app in app_catalog if app.get("app_id") == app_id), None)
                if app_match and app_match.get("app_name"):
                    resolved_app_names.add(str(app_match["app_name"]))
            flow_statuses.append(flow.resolution_status)

        if flow_statuses and all(status == "resolved" for status in flow_statuses) and len(resolved_app_ids) == 1:
            result.resolution_status = "resolved"
            result.app_id = next(iter(resolved_app_ids))
            if resolved_app_names:
                result.app_name = next(iter(resolved_app_names))
        elif "ambiguous" in flow_statuses:
            result.resolution_status = "ambiguous"
            result.app_id = ""
        else:
            result.resolution_status = "unmatched"
            result.app_id = ""

        return result

    def _postprocess_operation_intent(
        self,
        result: OperationIntent,
        *,
        user_input: str,
        session_id: str = "",
        snapshot_id: str = "",
    ) -> OperationIntent:
        if result.supi:
            inferred_supi = result.supi
        else:
            inferred_supi = next((flow.supi for flow in result.flows if flow.supi), "")
            if not inferred_supi:
                normalized_input = str(user_input or "")
                imsi_match = re.search(r"(?i)imsi-\d{5,}", normalized_input)
                supi_match = re.search(r"(?i)\bsupi\s*[:=]?\s*([\w-]+)", normalized_input)
                if imsi_match:
                    inferred_supi = imsi_match.group(0)
                elif supi_match:
                    inferred_supi = supi_match.group(1)

        if inferred_supi:
            result.supi = inferred_supi
            for flow in result.flows:
                if not flow.supi:
                    flow.supi = inferred_supi

        if session_id:
            result.session_id = str(session_id)
        if snapshot_id:
            result.snapshot_id = str(snapshot_id)

        normalized_input = str(user_input or "").strip()
        lowered_input = normalized_input.lower()
        if any(keyword in lowered_input for keyword in self.DELETE_KEYWORDS):
            result.operation_type = "delete"
        elif any(keyword in lowered_input for keyword in self.ADD_KEYWORDS):
            result.operation_type = "add"
        elif any(keyword in lowered_input for keyword in self.MODIFY_KEYWORDS):
            result.operation_type = "modify"
        else:
            result.operation_type = str(result.operation_type or "modify").strip() or "modify"

        result.raw_input = normalized_input
        result.app_id = str(result.app_id or "").strip()
        result.app_name = str(result.app_name or "").strip() or None
        result.urgency = str(result.urgency or "Normal").strip() or "Normal"
        result.raw_intent_summary = str(result.raw_intent_summary or "").strip()
        result.resolution_status = str(result.resolution_status or "").strip() or "unmatched"

        for flow in result.flows:
            flow.supi = str(flow.supi or result.supi or "").strip()
            flow.app_id = str(flow.app_id or result.app_id or "").strip()
            flow.target_type = str(flow.target_type or "flow").strip() or "flow"
            flow.name = str(flow.name or "").strip()
            if not flow.name:
                flow.name = self._infer_flow_keyword(result.raw_intent_summary, normalized_input)
            flow.service_type = str(flow.service_type or "").strip() or None
            flow.description = str(flow.description or "").strip() or None
            flow.resolution_status = str(flow.resolution_status or "").strip() or "unmatched"
            flow.resolution_candidates = [str(item) for item in flow.resolution_candidates]

        return self._resolve_operation_intent_against_catalog(result, normalized_input)
