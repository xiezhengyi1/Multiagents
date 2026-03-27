from typing import List, Optional, Dict, Any, Set
import re
import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import PydanticOutputParser

from .basemodel import BaseAgent
from .Prompt import IEA_SYSTEM_PROMPT
from utils.logger import setup_logger, log_event, log_timing
from tools.pcf_tools import get_ue_context, get_ue_flow_catalog
from tools.knowledge_tool import search_semantic_knowledge, get_knowledge_by_key
from tools.db_tool import get_ue_flow_catalog_by_supi


class FlowIntent(BaseModel):
    name: str = Field(default="", description="Flow name clue from the user request.")
    flow_id: Optional[str] = Field(default=None, description="Resolved existing flow ID when uniquely matched.")
    service_type: Optional[str] = Field(default=None, description="Inherited service type for the resolved flow.")
    service_type_id: Optional[int] = Field(default=None, description="Inherited service type ID for the resolved flow.")
    bw_ul: Optional[float] = Field(default=None, description="Requested target UL bandwidth in Mbps.")
    bw_dl: Optional[float] = Field(default=None, description="Requested target DL bandwidth in Mbps.")
    gbr_ul: Optional[float] = Field(default=None, description="Inherited GBR UL in Mbps.")
    gbr_dl: Optional[float] = Field(default=None, description="Inherited GBR DL in Mbps.")
    lat: Optional[float] = Field(default=None, description="Inherited latency requirement in ms.")
    loss_req: Optional[float] = Field(default=None, description="Inherited packet loss requirement.")
    jitter_req: Optional[float] = Field(default=None, description="Inherited jitter requirement in ms.")
    priority: Optional[int] = Field(default=None, description="Inherited priority for the resolved flow.")
    description: Optional[str] = Field(default=None, description="Human-readable flow description.")
    supi: Optional[str] = Field(default=None, description="User SUPI.")
    resolution_status: str = Field(default="unmatched", description="One of: resolved, ambiguous, unmatched.")
    current_bw_ul: Optional[float] = Field(default=None, description="Current UL bandwidth baseline in Mbps.")
    current_bw_dl: Optional[float] = Field(default=None, description="Current DL bandwidth baseline in Mbps.")
    five_tuple: Optional[List[Any]] = Field(default=None, description="Resolved five tuple: [src_ip, dst_ip, src_port, dst_port, protocol].")
    resolution_candidates: List[str] = Field(default_factory=list, description="Candidate app/flow labels when not uniquely resolved.")


class UserIntent(BaseModel):
    supi: Optional[str] = Field(default=None, description="UE SUPI such as imsi-...")
    app_name: Optional[str] = Field(default=None, description="App name clue from the user request.")
    app_id: Optional[str] = Field(default=None, description="Resolved existing app ID when uniquely matched.")
    operation_type: str = Field(default="modify", description="Operation type: add, modify, delete.")
    flows: List[FlowIntent] = Field(default_factory=list, description="Requested flows to be modified.")
    urgency: str = Field(default="Normal", description="Overall urgency.")
    raw_intent_summary: str = Field(default="", description="Summary of the raw user request.")
    resolution_status: str = Field(default="unmatched", description="One of: resolved, ambiguous, unmatched.")


class IntentEncodingAgent(BaseAgent):
    _FLOW_KEYWORDS = {
        "video": {"video", "视频"},
        "control": {"control", "控制"},
        "telemetry": {"telemetry", "遥测"},
    }
    _UL_KEYWORDS = {"ul", "uplink", "upstream", "上行", "上传"}
    _DL_KEYWORDS = {"dl", "downlink", "downstream", "下行", "下载"}

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=UserIntent)
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[95m")
        self.tools = [get_ue_context, get_ue_flow_catalog, search_semantic_knowledge, get_knowledge_by_key]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}

    def analyze_intent(self, user_input: str, context: str = "") -> Optional[UserIntent]:
        ue_context = ""
        prompt = ChatPromptTemplate.from_messages([("system", IEA_SYSTEM_PROMPT)])
        formatted_messages = prompt.format_messages(
            user_input=user_input,
            context=context,
            ue_context=ue_context,
            format_instructions=self.parser.get_format_instructions(),
        )
        messages = formatted_messages

        total_start = time.perf_counter()
        log_event(self.logger, "iea_analyze_start")
        try:
            for iteration in range(5):
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    output_str = response.content.strip()
                    log_event(self.logger, "iea_parse_output_start", iteration=iteration + 1)
                    try:
                        if "```json" in output_str:
                            output_str = output_str.split("```json", 1)[1].split("```", 1)[0]
                        elif "```" in output_str:
                            output_str = output_str.split("```", 1)[1].split("```", 1)[0]

                        result = self.parser.parse(output_str)
                        result = self._postprocess_user_intent(result, user_input)
                        log_event(self.logger, "iea_analyze_success", app_name=result.app_name or "")
                        log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="success")
                        return result
                    except Exception as parse_err:
                        self.logger.error(f"解析错误: {parse_err}. 输出字段: {output_str}")
                        log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="parse_failed")
                        return None

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    log_event(self.logger, "iea_tool_call", tool=tool_name)

                    if tool_name in self.tool_map:
                        tool_instance = self.tool_map[tool_name]
                        try:
                            tool_start = time.perf_counter()
                            tool_result = tool_instance.invoke(tool_args)
                            log_timing(self.logger, "iea_tool_call", time.perf_counter() - tool_start, tool=tool_name)
                        except Exception as exc:
                            tool_result = f"工具执行异常: {exc}"
                    else:
                        tool_result = f"Error: Tool {tool_name} not found."

                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

            self.logger.warning("达到最大迭代次数，未生成有效结果。")
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="max_iterations")
            return None
        except Exception as exc:
            self.logger.error(f"意图解析过程中出错: {exc}")
            log_timing(self.logger, "iea_total", time.perf_counter() - total_start, status="error")
            return None

    @staticmethod
    def _normalize_label(text: Any) -> str:
        cleaned = str(text or "").strip().lower()
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", cleaned)

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
        if not text:
            return None
        match = re.search(r"(?i)imsi-\d{5,}", text)
        if match:
            return match.group(0)

        match = re.search(r"(?i)\bsupi\s*[:=]?\s*([\w-]+)", text)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _extract_explicit_id(text: str, prefix: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(rf"(?i)\b{prefix}-\d{{4,8}}\b", text)
        if match:
            return match.group(0)
        return None

    @classmethod
    def _extract_flow_keywords(cls, text: str) -> Set[str]:
        normalized = str(text or "").lower()
        keywords: Set[str] = set()
        for canonical, aliases in cls._FLOW_KEYWORDS.items():
            if any(alias in normalized for alias in aliases):
                keywords.add(canonical)
        return keywords

    @classmethod
    def _flow_has_keyword(cls, flow_name: str, keyword: str, app_name: Optional[str] = None) -> bool:
        aliases = cls._FLOW_KEYWORDS.get(keyword, {keyword})
        haystack = str(flow_name or "").lower()
        if app_name:
            app_prefix = str(app_name).lower()
            for separator in ("_", "-", " "):
                candidate_prefix = f"{app_prefix}{separator}"
                if haystack.startswith(candidate_prefix):
                    haystack = haystack[len(candidate_prefix):]
                    break
        return any(alias in haystack for alias in aliases)

    @classmethod
    def _detect_direction(cls, text: str) -> Optional[str]:
        normalized = str(text or "").lower()
        has_ul = any(keyword in normalized for keyword in cls._UL_KEYWORDS)
        has_dl = any(keyword in normalized for keyword in cls._DL_KEYWORDS)
        if has_ul and not has_dl:
            return "ul"
        if has_dl and not has_ul:
            return "dl"
        return None

    @staticmethod
    def _infer_operation_type(result: UserIntent, user_input: str) -> str:
        normalized = str(user_input or "").lower()
        if any(token in normalized for token in ("delete", "remove", "删除", "移除")):
            return "delete"
        if any(token in normalized for token in ("add", "create", "新增", "创建")):
            return "add"
        if any(token in normalized for token in ("modify", "update", "change", "调整", "修改", "降低", "提高", "增加", "减少")):
            return "modify"
        return str(result.operation_type or "modify").strip() or "modify"

    @staticmethod
    def _format_candidate_label(flow_entry: Dict[str, Any]) -> str:
        app_name = str(flow_entry.get("app_name") or "")
        flow_name = str(flow_entry.get("flow_name") or "")
        app_id = str(flow_entry.get("app_id") or "")
        flow_id = str(flow_entry.get("flow_id") or "")
        return f"{app_name}/{flow_name} ({app_id}/{flow_id})".strip()

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
            dominant = "dl" if (current_dl or 0.0) >= (current_ul or 0.0) else "ul"
            if dominant == "dl":
                target_ul = current_ul
                if requested_dl is not None:
                    target_dl = requested_dl
                elif requested_ul is not None:
                    target_dl = requested_ul
                else:
                    target_dl = current_dl
            else:
                target_dl = current_dl
                if requested_ul is not None:
                    target_ul = requested_ul
                elif requested_dl is not None:
                    target_ul = requested_dl
                else:
                    target_ul = current_ul

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
        no_keyword_match = False

        exact_app_matches: List[Dict[str, Any]] = []
        if explicit_app_id:
            exact_app_matches = [app for app in app_catalog if app.get("app_id") == explicit_app_id]
        elif result.app_name:
            exact_app_matches = [
                app for app in app_catalog
                if self._normalize_label(app.get("app_name")) == self._normalize_label(result.app_name)
            ]

        if explicit_flow_id:
            matches = [item for item in flow_catalog if item.get("flow_id") == explicit_flow_id]
            scope_candidates = matches
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
            else:
                keywords = self._extract_flow_keywords(f"{flow.name} {user_input}")
                if len(exact_app_matches) == 1 and keywords:
                    app_id = exact_app_matches[0].get("app_id")
                    app_scope = [item for item in flow_catalog if item.get("app_id") == app_id]
                    keyword_scope = [
                        item for item in app_scope
                        if any(
                            self._flow_has_keyword(
                                item.get("flow_name", ""),
                                keyword,
                                app_name=str(exact_app_matches[0].get("app_name") or ""),
                            )
                            for keyword in keywords
                        )
                    ]
                    matches = keyword_scope
                    scope_candidates = keyword_scope if keyword_scope else app_scope
                    no_keyword_match = not keyword_scope
                else:
                    if len(exact_app_matches) == 1:
                        app_id = exact_app_matches[0].get("app_id")
                        scope_candidates = [item for item in flow_catalog if item.get("app_id") == app_id]
                    else:
                        scope_candidates = list(flow_catalog)

                    if len(scope_candidates) == 1:
                        matches = list(scope_candidates)
                    else:
                        matches = []

        if len(matches) == 1:
            resolved = matches[0]
            self._apply_resolved_catalog_flow(flow, resolved, user_input)
            return str(resolved.get("app_id") or "")

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

        catalog = catalog or get_ue_flow_catalog_by_supi(result.supi)
        app_catalog = catalog.get("app_catalog") or []
        flow_catalog = catalog.get("flow_catalog") or []
        if not flow_catalog:
            result.resolution_status = "unmatched"
            result.app_id = None
            for flow in result.flows:
                flow.resolution_status = "unmatched"
            return result

        resolved_app_ids: Set[str] = set()
        resolved_app_names: Set[str] = set()
        statuses: List[str] = []
        for flow in result.flows:
            flow.supi = result.supi
            app_id = self._resolve_single_flow(flow, result, user_input, app_catalog, flow_catalog)
            if app_id:
                resolved_app_ids.add(app_id)
                app_match = next((app for app in app_catalog if app.get("app_id") == app_id), None)
                if app_match and app_match.get("app_name"):
                    resolved_app_names.add(str(app_match["app_name"]))
            statuses.append(flow.resolution_status)

        if statuses and all(status == "resolved" for status in statuses) and len(resolved_app_ids) == 1:
            result.resolution_status = "resolved"
            result.app_id = next(iter(resolved_app_ids))
            if resolved_app_names:
                result.app_name = next(iter(resolved_app_names))
        elif "ambiguous" in statuses:
            result.resolution_status = "ambiguous"
            result.app_id = None
        else:
            result.resolution_status = "unmatched"
            result.app_id = None

        return result

    def _postprocess_user_intent(self, result: UserIntent, user_input: str) -> UserIntent:
        inferred_supi = result.supi
        if not inferred_supi:
            inferred_supi = next((flow.supi for flow in result.flows if flow.supi), None)
        if not inferred_supi:
            inferred_supi = self._extract_supi(user_input)

        if inferred_supi:
            result.supi = inferred_supi
            for flow in result.flows:
                if not flow.supi:
                    flow.supi = inferred_supi

        result.operation_type = self._infer_operation_type(result, user_input)
        return self._resolve_user_intent_against_catalog(result, user_input)
