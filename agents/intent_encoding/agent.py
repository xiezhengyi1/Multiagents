from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set

from agent_runtime import AgentWorkspace, ArtifactCache, ArtifactEnvelope, ArtifactStore
from agents.basemodel import BaseAgent
from domain.policy_plan import FlowSelector, OperationIntent
from tools.db_tool import get_ue_flow_catalog_by_supi
from tools.knowledge_tool import get_knowledge_by_key, search_semantic_knowledge
from tools.pcf_tools import get_ue_context, get_ue_flow_catalog
from utils.logger import log_event, log_timing, setup_logger

from .contracts import FlowIntent, UserIntent
from .prompts import IEA_SYSTEM_PROMPT


class IntentEncodingAgent(BaseAgent):
    FLOW_KEYWORDS: Dict[str, Set[str]] = {
        "video": {"video", "视频"},
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
        self.workspace = AgentWorkspace.for_agent(self.agent_name)
        self.cache = ArtifactCache(self.workspace)
        self.artifact_store = ArtifactStore()
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[95m")
        self.tools = [get_ue_context, get_ue_flow_catalog, search_semantic_knowledge, get_knowledge_by_key]
        self.agent = self.create_structured_agent(
            tools=self.tools,
            system_prompt=IEA_SYSTEM_PROMPT,
            response_format=UserIntent,
        )

    @staticmethod
    def _build_request_message(user_input: str, context: str = "") -> str:
        return (
            f"User input:\n{user_input}\n\n"
            f"Conversation context:\n{context or 'N/A'}\n\n"
            "Use tools when entity resolution depends on live UE context or semantic knowledge."
        )

    def _cache_received_request(
        self,
        *,
        user_input: str,
        context: str,
        session_id: str,
        snapshot_id: str,
    ) -> ArtifactEnvelope:
        envelope = ArtifactEnvelope(
            artifact_type="OperationIntentRequest",
            source_agent="coordinator",
            target_agent=self.agent_name,
            session_id=str(session_id or "").strip(),
            snapshot_id=str(snapshot_id or "").strip(),
            payload={
                "user_input": str(user_input),
                "context": str(context or ""),
            },
        )
        self.cache.cache_received(envelope)
        return envelope

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        operation_intent: OperationIntent,
    ) -> None:
        self.cache.cache_produced(
            ArtifactEnvelope(
                artifact_type="OperationIntent",
                source_agent=self.agent_name,
                target_agent=request_envelope.source_agent,
                session_id=request_envelope.session_id,
                snapshot_id=request_envelope.snapshot_id,
                correlation_id=request_envelope.correlation_id,
                upstream_artifact_ids=[request_envelope.artifact_id],
                payload=operation_intent.model_dump(mode="json"),
            )
        )

    def analyze_intent(
        self,
        user_input: str,
        context: str = "",
        *,
        session_id: str = "",
        snapshot_id: str = "",
    ) -> UserIntent:
        if not str(user_input).strip():
            raise ValueError("user_input must not be empty")

        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            thread_id=session_id,
        )

        total_start = time.perf_counter()
        log_event(self.logger, "iea_analyze_start")
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": self._build_request_message(user_input, context=context)}]},
                context=runtime_context,
            )
            structured = result.get("structured_response")
            if structured is None:
                raise RuntimeError("IEA agent returned no structured_response.")

            parsed = UserIntent.model_validate(structured)
            normalized = self._postprocess_user_intent(parsed, user_input)
            log_event(self.logger, "iea_analyze_success", app_name=normalized.app_name or "")
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
            return normalized
        except Exception:
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="error")
            raise

    def analyze_operation_intent(
        self,
        user_input: str,
        context: str = "",
        *,
        session_id: str = "",
        snapshot_id: str = "",
    ) -> OperationIntent:
        request_envelope = self._cache_received_request(
            user_input=user_input,
            context=context,
            session_id=session_id,
            snapshot_id=snapshot_id,
        )
        user_intent = self.analyze_intent(
            user_input,
            context=context,
            session_id=session_id,
            snapshot_id=snapshot_id,
        )
        operation_intent = self._user_intent_to_operation_intent(
            user_intent,
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
        )
        self._cache_produced_result(
            request_envelope=request_envelope,
            operation_intent=operation_intent,
        )
        return operation_intent

    @staticmethod
    def _normalize_label(text: Any) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").strip().lower())

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_supi(text: str) -> Optional[str]:
        normalized = str(text or "")
        imsi_match = re.search(r"(?i)imsi-\d{5,}", normalized)
        if imsi_match:
            return imsi_match.group(0)

        supi_match = re.search(r"(?i)\bsupi\s*[:=]?\s*([\w-]+)", normalized)
        if supi_match:
            return supi_match.group(1)
        return None

    @staticmethod
    def _extract_explicit_id(text: str, prefix: str) -> Optional[str]:
        match = re.search(rf"(?i)\b{prefix}-\d{{4,8}}\b", str(text or ""))
        return match.group(0) if match else None

    @classmethod
    def _extract_flow_keywords(cls, text: str) -> Set[str]:
        normalized = str(text or "").lower()
        matched: Set[str] = set()
        for canonical, aliases in cls.FLOW_KEYWORDS.items():
            if any(alias in normalized for alias in aliases):
                matched.add(canonical)
        return matched

    @classmethod
    def _flow_has_keyword(cls, flow_name: str, keyword: str, app_name: Optional[str] = None) -> bool:
        aliases = cls.FLOW_KEYWORDS.get(keyword, {keyword})
        normalized_name = str(flow_name or "").lower()
        if app_name:
            normalized_app = str(app_name).lower()
            for separator in ("_", "-", " "):
                prefix = f"{normalized_app}{separator}"
                if normalized_name.startswith(prefix):
                    normalized_name = normalized_name[len(prefix) :]
                    break
        return any(alias in normalized_name for alias in aliases)

    @classmethod
    def _detect_direction(cls, text: str) -> Optional[str]:
        normalized = str(text or "").lower()
        has_ul = any(keyword in normalized for keyword in cls.UL_KEYWORDS)
        has_dl = any(keyword in normalized for keyword in cls.DL_KEYWORDS)
        if has_ul and not has_dl:
            return "ul"
        if has_dl and not has_ul:
            return "dl"
        return None

    @classmethod
    def _contains_any(cls, text: str, keywords: Iterable[str]) -> bool:
        normalized = str(text or "").lower()
        return any(keyword in normalized for keyword in keywords)

    @classmethod
    def _infer_operation_type(cls, result: UserIntent, user_input: str) -> str:
        if cls._contains_any(user_input, cls.DELETE_KEYWORDS):
            return "delete"
        if cls._contains_any(user_input, cls.ADD_KEYWORDS):
            return "add"
        if cls._contains_any(user_input, cls.MODIFY_KEYWORDS):
            return "modify"
        normalized = str(result.operation_type or "").strip()
        return normalized or "modify"

    @staticmethod
    def _format_candidate_label(flow_entry: Dict[str, Any]) -> str:
        return (
            f"{flow_entry.get('app_name', '')}/{flow_entry.get('flow_name', '')} "
            f"({flow_entry.get('app_id', '')}/{flow_entry.get('flow_id', '')})"
        ).strip()

    def _resolve_target_bandwidths(
        self,
        flow: FlowIntent,
        catalog_flow: Dict[str, Any],
        user_input: str,
    ) -> Dict[str, Optional[float]]:
        current_ul = self._coerce_float(catalog_flow.get("current_bw_ul"))
        current_dl = self._coerce_float(catalog_flow.get("current_bw_dl"))
        requested_ul = self._coerce_float(flow.bw_ul)
        requested_dl = self._coerce_float(flow.bw_dl)
        direction = self._detect_direction(user_input)

        if direction == "ul":
            target_ul = requested_ul if requested_ul is not None else current_ul
            target_dl = current_dl
        elif direction == "dl":
            target_ul = current_ul
            target_dl = requested_dl if requested_dl is not None else current_dl
        else:
            dominant_direction = "dl" if (current_dl or 0.0) >= (current_ul or 0.0) else "ul"
            if dominant_direction == "dl":
                target_ul = current_ul
                target_dl = requested_dl if requested_dl is not None else requested_ul if requested_ul is not None else current_dl
            else:
                target_dl = current_dl
                target_ul = requested_ul if requested_ul is not None else requested_dl if requested_dl is not None else current_ul

        return {
            "bw_ul": target_ul,
            "bw_dl": target_dl,
            "current_bw_ul": current_ul,
            "current_bw_dl": current_dl,
        }

    def _apply_resolved_catalog_flow(self, flow: FlowIntent, catalog_flow: Dict[str, Any], user_input: str) -> None:
        bandwidths = self._resolve_target_bandwidths(flow, catalog_flow, user_input)
        flow.name = str(catalog_flow.get("flow_name") or flow.name or "")
        flow.flow_id = str(catalog_flow.get("flow_id") or "")
        flow.service_type = catalog_flow.get("service_type")
        flow.service_type_id = catalog_flow.get("service_type_id")
        flow.bw_ul = bandwidths["bw_ul"]
        flow.bw_dl = bandwidths["bw_dl"]
        flow.gbr_ul = self._coerce_float(catalog_flow.get("gbr_ul"))
        flow.gbr_dl = self._coerce_float(catalog_flow.get("gbr_dl"))
        flow.lat = self._coerce_float(catalog_flow.get("lat"))
        flow.loss_req = self._coerce_float(catalog_flow.get("loss_req"))
        flow.jitter_req = self._coerce_float(catalog_flow.get("jitter_req"))
        flow.priority = int(catalog_flow.get("priority")) if catalog_flow.get("priority") is not None else None
        flow.description = str(catalog_flow.get("flow_name") or flow.description or "")
        flow.current_bw_ul = bandwidths["current_bw_ul"]
        flow.current_bw_dl = bandwidths["current_bw_dl"]
        flow.five_tuple = list(catalog_flow.get("five_tuple")) if isinstance(catalog_flow.get("five_tuple"), (list, tuple)) else None
        flow.resolution_status = "resolved"
        flow.resolution_candidates = [self._format_candidate_label(catalog_flow)]

    def _resolve_single_flow(
        self,
        flow: FlowIntent,
        result: UserIntent,
        user_input: str,
        app_catalog: List[Dict[str, Any]],
        flow_catalog: List[Dict[str, Any]],
    ) -> Optional[str]:
        explicit_flow_id = self._extract_explicit_id(flow.name, "flow") or self._extract_explicit_id(user_input, "flow")
        explicit_app_id = self._extract_explicit_id(result.app_name or "", "app") or self._extract_explicit_id(user_input, "app")

        if explicit_app_id:
            exact_app_matches = [app for app in app_catalog if app.get("app_id") == explicit_app_id]
        elif result.app_name:
            exact_app_matches = [
                app for app in app_catalog
                if self._normalize_label(app.get("app_name")) == self._normalize_label(result.app_name)
            ]
        else:
            exact_app_matches = []

        if explicit_flow_id:
            matches = [item for item in flow_catalog if item.get("flow_id") == explicit_flow_id]
            scope_candidates = matches
            no_keyword_match = False
        else:
            exact_flow_matches = []
            if flow.name:
                exact_flow_matches = [
                    item for item in flow_catalog
                    if self._normalize_label(item.get("flow_name")) == self._normalize_label(flow.name)
                ]

            if exact_flow_matches:
                matches = exact_flow_matches
                scope_candidates = exact_flow_matches
                no_keyword_match = False
            else:
                keywords = self._extract_flow_keywords(f"{flow.name} {user_input}")
                if len(exact_app_matches) == 1 and keywords:
                    matched_app = exact_app_matches[0]
                    app_scope = [item for item in flow_catalog if item.get("app_id") == matched_app.get("app_id")]
                    keyword_scope = [
                        item
                        for item in app_scope
                        if any(
                            self._flow_has_keyword(
                                item.get("flow_name", ""),
                                keyword,
                                app_name=str(matched_app.get("app_name") or ""),
                            )
                            for keyword in keywords
                        )
                    ]
                    matches = keyword_scope
                    scope_candidates = keyword_scope if keyword_scope else app_scope
                    no_keyword_match = not keyword_scope
                else:
                    if len(exact_app_matches) == 1:
                        matched_app = exact_app_matches[0]
                        scope_candidates = [item for item in flow_catalog if item.get("app_id") == matched_app.get("app_id")]
                    else:
                        scope_candidates = list(flow_catalog)

                    matches = list(scope_candidates) if len(scope_candidates) == 1 else []
                    no_keyword_match = False

        if len(matches) == 1:
            resolved_flow = matches[0]
            self._apply_resolved_catalog_flow(flow, resolved_flow, user_input)
            return str(resolved_flow.get("app_id") or "")

        flow.flow_id = None
        flow.current_bw_ul = None
        flow.current_bw_dl = None
        flow.five_tuple = None
        flow.resolution_candidates = [self._format_candidate_label(item) for item in scope_candidates[:5]]
        if no_keyword_match:
            flow.resolution_status = "unmatched"
        elif len(matches) > 1 or len(scope_candidates) > 1:
            flow.resolution_status = "ambiguous"
        else:
            flow.resolution_status = "unmatched"
        return None

    def _resolve_user_intent_against_catalog(
        self,
        result: UserIntent,
        user_input: str,
        catalog: Optional[Dict[str, Any]] = None,
    ) -> UserIntent:
        if not result.flows:
            result.flows = [FlowIntent(name="")]

        if not result.supi:
            result.resolution_status = "unmatched"
            result.app_id = None
            for flow in result.flows:
                flow.resolution_status = "unmatched"
            return result

        catalog_payload = catalog if catalog is not None else get_ue_flow_catalog_by_supi(result.supi)
        app_catalog = catalog_payload.get("app_catalog") or []
        flow_catalog = catalog_payload.get("flow_catalog") or []
        if not flow_catalog:
            result.resolution_status = "unmatched"
            result.app_id = None
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
            result.app_id = None
        else:
            result.resolution_status = "unmatched"
            result.app_id = None

        return result

    def _postprocess_user_intent(self, result: UserIntent, user_input: str) -> UserIntent:
        inferred_supi = (
            result.supi
            or next((flow.supi for flow in result.flows if flow.supi), None)
            or self._extract_supi(user_input)
        )
        if inferred_supi:
            result.supi = inferred_supi
            for flow in result.flows:
                if not flow.supi:
                    flow.supi = inferred_supi

        result.operation_type = self._infer_operation_type(result, user_input)
        return self._resolve_user_intent_against_catalog(result, user_input)

    @staticmethod
    def _user_intent_to_operation_intent(
        user_intent: UserIntent,
        *,
        user_input: str,
        session_id: str = "",
        snapshot_id: str = "",
    ) -> OperationIntent:
        flow_selectors = [
            FlowSelector(
                supi=str(flow.supi or user_intent.supi or "").strip(),
                app_id=str(user_intent.app_id or "").strip(),
                flow_id=str(flow.flow_id or "").strip() or None,
                target_type="flow",
                name=str(flow.name or "").strip(),
                service_type=str(flow.service_type or "").strip() or None,
                service_type_id=flow.service_type_id,
                bw_ul=flow.bw_ul,
                bw_dl=flow.bw_dl,
                gbr_ul=flow.gbr_ul,
                gbr_dl=flow.gbr_dl,
                lat=flow.lat,
                loss_req=flow.loss_req,
                jitter_req=flow.jitter_req,
                priority=flow.priority,
                description=str(flow.description or "").strip() or None,
                five_tuple=list(flow.five_tuple) if isinstance(flow.five_tuple, (list, tuple)) else None,
                current_bw_ul=flow.current_bw_ul,
                current_bw_dl=flow.current_bw_dl,
                resolution_status=str(flow.resolution_status or "").strip() or "unmatched",
                resolution_candidates=[str(item) for item in flow.resolution_candidates],
            )
            for flow in user_intent.flows
        ]

        return OperationIntent(
            session_id=session_id,
            snapshot_id=snapshot_id,
            supi=str(user_intent.supi or "").strip(),
            app_id=str(user_intent.app_id or "").strip(),
            app_name=str(user_intent.app_name or "").strip() or None,
            operation_type=str(user_intent.operation_type or "modify").strip() or "modify",
            urgency=str(user_intent.urgency or "Normal").strip() or "Normal",
            raw_input=str(user_input or "").strip(),
            raw_intent_summary=str(user_intent.raw_intent_summary or "").strip(),
            resolution_status=str(user_intent.resolution_status or "").strip() or "unmatched",
            flows=flow_selectors,
        )
