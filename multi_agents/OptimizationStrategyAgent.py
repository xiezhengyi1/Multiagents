from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_serializer

from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest
from tools.db_tool import build_flow_description_from_five_tuple, build_flow_info_from_five_tuple
from tools.network_status import get_network_status_summary
from tools.optimizer import optimize_network_slices
from utils.logger import log_event, log_timing, setup_logger
from .Prompt import OSA_SYSTEM_PROMPT
from .basemodel import BaseAgent

logger = setup_logger(__name__)


@tool
def fetch_network_status(flow_type_id: int) -> str:
    """Fetch network status for a service type."""
    try:
        status = get_network_status_summary(flow_type_id=flow_type_id)
        logger.info("Fetched network status.")
        # logger.info(f"Network Status:\n{status}")
        return status
    except Exception as exc:
        return f"获取网络状态失败: {str(exc)}"


@tool
def run_optimization_solver(w1: float, w2: float, w3: float, mode: str, app_details: str) -> str:
    """
    Invoke the optimizer.
    Args:
    - w1: Load balancing weight
    - w2: Reconfiguration cost weight
    - w3: Service experience weight
    - mode: Optimization mode (full/incremental/hybrid)
    - app_details: JSON string containing app and flow details (including QoS requirements)
    
    Returns:
    - A JSON string containing the optimized strategy, including recommended actions and policy details.
    """
    try:
        app_data = json.loads(app_details) if isinstance(app_details, str) else app_details
        result = optimize_network_slices(app_data, w1, w2, w3, mode=mode)
        meta = result.get("meta", {})
        logger.info(f"Optimization Solver Result:\n{meta.get('status')}")
        return result
    except Exception as exc:
        return f"工具执行错误: {str(exc)}"


def _json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _extract_json_payload(text: str) -> str:
    cleaned = (text or "").strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
    return cleaned.strip()


def _strip_rule_prefix(candidate: Any) -> Optional[str]:
    if candidate is None:
        return None
    text = str(candidate).strip()
    if not text:
        return None
    for prefix in ("pcc-", "qos-", "sess-", "smp-", "ursp-"):
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix) :]
    return text


def _build_policy_id(policy_type: str, app_id: str, flow_id: Optional[str], target_type: str) -> str:
    prefix = "ursp" if policy_type == "UrspRuleRequest" else "smp"
    if target_type == "flow" and flow_id:
        return f"{prefix}-{app_id}-{flow_id}"
    return f"{prefix}-{app_id}"


def _normalize_app_id(app_id: Any) -> str:
    text = str(app_id or "").strip()
    if not text:
        return ""
    if text.startswith("app_"):
        return f"app-{text[4:].replace('_', '-')}"
    if text.startswith("app-"):
        return f"app-{text[4:].replace('_', '-')}"
    if re.fullmatch(r"app\d+", text, flags=re.IGNORECASE):
        return f"app-{text[3:]}"
    return text.replace("_", "-")


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _extract_ursp_flow_desc_from_five_tuple(five_tuple: Any) -> Optional[str]:
    if not isinstance(five_tuple, (list, tuple)) or len(five_tuple) != 5:
        return None
    _, dst_ip, _, dst_port, protocol = five_tuple
    dst_ip = str(dst_ip or "").strip()
    protocol = str(protocol or "").strip().upper()
    try:
        dst_port = int(dst_port)
    except (TypeError, ValueError):
        return None
    if not dst_ip or not protocol:
        return None
    return f"{protocol} {dst_ip} {dst_port}"


class Strategy(BaseModel):
    """OSA strategy envelope."""

    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    supi: str = Field(default="", description="User SUPI")
    app_id: str = Field(default="", description="Application ID")
    flow_id: Optional[str] = Field(default=None, description="Flow ID")
    target_type: str = Field(default="flow", description="Target scope")
    policy_id: str = Field(default="", description="Unique policy ID")
    policy_type: str = Field(..., description="Policy type")
    policy_details: Dict[str, Any] = Field(default_factory=dict, description="Raw policy details")

    @field_serializer("policy_details", when_used="always")
    def _serialize_policy_details(self, value: Dict[str, Any]) -> Any:
        return _json_friendly(value)


class OutputStrategy(BaseModel):
    """OSA output envelope."""

    supi: str = Field(default="", description="User SUPI")
    all_policies: List[Strategy] = Field(default_factory=list, description="All generated policies")


class OptimizationStrategyAgent(BaseAgent):
    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[94m")
        self.tools = [run_optimization_solver, fetch_network_status]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool_obj.name: tool_obj for tool_obj in self.tools}
        self.output_parser = PydanticOutputParser(pydantic_object=OutputStrategy)

    def _normalize_tool_args_for_log(self, tool_args: Any) -> Any:
        if not isinstance(tool_args, dict):
            return tool_args

        normalized = dict(tool_args)
        for key in ["incremental_flows", "strategy_json", "app_details"]:
            value = normalized.get(key)
            if isinstance(value, str):
                try:
                    normalized[key] = json.loads(value)
                except Exception:
                    pass
        return normalized

    @staticmethod
    def _extract_flow_id_from_policy_data(data: Dict[str, Any]) -> Optional[str]:
        flow_id = _strip_rule_prefix(data.get("flow_id") or data.get("flowId"))
        if flow_id:
            return flow_id

        for mapping_name, id_field in (("qosDecs", "qosId"), ("pccRules", "pccRuleId"), ("sessRules", "sessRuleId")):
            mapping = data.get(mapping_name) or data.get(mapping_name[0].lower() + mapping_name[1:])
            if not isinstance(mapping, dict) or not mapping:
                continue

            for key, item in mapping.items():
                key_flow_id = _strip_rule_prefix(key)
                if key_flow_id:
                    return key_flow_id
                if isinstance(item, dict):
                    item_flow_id = _strip_rule_prefix(item.get(id_field))
                    if item_flow_id:
                        return item_flow_id
        return None

    @staticmethod
    def _pick_mapping_entry(mapping: Dict[str, Any], flow_id: Optional[str], id_field: str) -> Tuple[str, Dict[str, Any]]:
        if not isinstance(mapping, dict) or not mapping:
            return "", {}

        if flow_id:
            for key, item in mapping.items():
                if _strip_rule_prefix(key) == flow_id:
                    return str(key), _json_friendly(item)
                if isinstance(item, dict) and _strip_rule_prefix(item.get(id_field)) == flow_id:
                    return str(key), _json_friendly(item)

        first_key = next(iter(mapping.keys()))
        return str(first_key), _json_friendly(mapping[first_key])

    @staticmethod
    def _normalize_ursp_app_descs(data: Dict[str, Any]) -> Dict[str, Any]:
        traffic_desc = data.get("trafficDesc")
        if not isinstance(traffic_desc, dict):
            return data

        app_descs = traffic_desc.get("appDescs")
        if isinstance(app_descs, dict) and "osId" in app_descs and "appIds" in app_descs:
            os_id = str(app_descs["osId"])
            traffic_desc["appDescs"] = {
                os_id: {
                    "osId": os_id,
                    "appIds": app_descs["appIds"],
                }
            }
        elif isinstance(app_descs, list):
            traffic_desc.pop("appDescs", None)
        return data

    @staticmethod
    def _normalize_snssai(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        text = str(value or "").strip()
        if len(text) >= 3 and text.isdigit():
            if len(text) >= 8:
                return {"sst": int(text[:2]), "sd": text[2:8]}
            return {"sst": int(text[:1]), "sd": text[1:] or None}
        return value

    @staticmethod
    def _pick_first_dict(*values: Any) -> Dict[str, Any]:
        for value in values:
            if isinstance(value, dict):
                return _json_friendly(value)
        return {}

    @staticmethod
    def _build_flow_infos(
        flow_ctx: Optional[Dict[str, Any]],
        selected_pcc: Dict[str, Any],
        canonical_flow_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        existing = selected_pcc.get("flowInfos")
        if isinstance(existing, list) and existing:
            return _json_friendly(existing)

        if flow_ctx:
            flow_info = build_flow_info_from_five_tuple(flow_ctx.get("five_tuple"))
            if flow_info:
                return [flow_info]
            flow_description = build_flow_description_from_five_tuple(flow_ctx.get("five_tuple"))
            if flow_description:
                return [{"flowDescription": flow_description, "flowDirection": "BIDIRECTIONAL"}]

            description = str(flow_ctx.get("description") or flow_ctx.get("name") or canonical_flow_id or "flow").strip()
            if description:
                return [{"flowDescription": description, "flowDirection": "BIDIRECTIONAL"}]

        if canonical_flow_id:
            return [{"flowDescription": canonical_flow_id, "flowDirection": "BIDIRECTIONAL"}]
        return []

    @staticmethod
    def _normalize_traffic_desc_flow_descs(flow_descs: Any) -> List[str]:
        normalized: List[str] = []
        if not isinstance(flow_descs, list):
            return normalized

        for item in flow_descs:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            protocol = str(item.get("protocol") or "").strip().upper()
            server_ip = str(item.get("serverIp") or item.get("server_ip") or "").strip()
            server_port = item.get("serverPort") or item.get("server_port")
            if protocol and server_ip and server_port not in (None, ""):
                normalized.append(f"{protocol} {server_ip} {server_port}")
        return normalized

    def _build_traffic_desc(self, details: Dict[str, Any], flow_ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        top_level = details.get("trafficDesc")
        nested = None
        route_sets = details.get("routeSelParamSets")
        if isinstance(route_sets, list):
            for route_set in route_sets:
                if isinstance(route_set, dict) and isinstance(route_set.get("trafficDesc"), dict):
                    nested = route_set.get("trafficDesc")
                    break

        traffic_desc = self._pick_first_dict(top_level, nested)
        if not traffic_desc and flow_ctx:
            flow_desc = _extract_ursp_flow_desc_from_five_tuple(flow_ctx.get("five_tuple"))
            if flow_desc:
                traffic_desc = {"flowDescs": [flow_desc]}

        if not traffic_desc:
            return None

        traffic_desc = self._normalize_ursp_app_descs({"trafficDesc": traffic_desc}).get("trafficDesc", {})
        flow_descs = self._normalize_traffic_desc_flow_descs(traffic_desc.get("flowDescs"))
        if not flow_descs and flow_ctx:
            flow_desc = _extract_ursp_flow_desc_from_five_tuple(flow_ctx.get("five_tuple"))
            if flow_desc:
                flow_descs = [flow_desc]
        if flow_descs:
            traffic_desc["flowDescs"] = flow_descs
        else:
            traffic_desc.pop("flowDescs", None)

        cleaned: Dict[str, Any] = {}
        for key in ("appDescs", "flowDescs", "domainDescs", "ethFlowDescs", "dnns", "connCaps"):
            value = traffic_desc.get(key)
            if value not in (None, "", [], {}):
                cleaned[key] = value
        return cleaned or None

    def _build_route_selection_parameter_sets(
        self,
        details: Dict[str, Any],
        flow_ctx: Optional[Dict[str, Any]],
        traffic_desc: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        route_sets = details.get("routeSelParamSets")
        precedence_default = _coerce_int(details.get("relatPrecedence")) or _coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1
        default_dnn = None
        if isinstance(traffic_desc, dict):
            dnns = traffic_desc.get("dnns")
            if isinstance(dnns, list) and dnns:
                default_dnn = dnns[0]

        normalized_route_sets: List[Dict[str, Any]] = []
        if isinstance(route_sets, list):
            for route_set in route_sets:
                if not isinstance(route_set, dict):
                    continue
                precedence = _coerce_int(route_set.get("precedence")) or _coerce_int(route_set.get("priority")) or precedence_default
                dnn = _coerce_str(route_set.get("dnn")) or default_dnn or "default"
                normalized_route_set: Dict[str, Any] = {
                    "dnn": dnn,
                    "precedence": precedence,
                }
                snssai = self._normalize_snssai(route_set.get("snssai"))
                if snssai is not None:
                    normalized_route_set["snssai"] = snssai
                normalized_route_sets.append(normalized_route_set)

        if normalized_route_sets:
            return normalized_route_sets

        fallback_route_set: Dict[str, Any] = {
            "dnn": default_dnn or "default",
            "precedence": precedence_default,
        }
        snssai = self._normalize_snssai(details.get("snssai"))
        if snssai is not None:
            fallback_route_set["snssai"] = snssai
        return [fallback_route_set]

    def _normalize_sm_policy_details(
        self,
        details: Dict[str, Any],
        flow_id: Optional[str],
        flow_ctx: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        data = _json_friendly(details)

        pcc_rules = data.get("pccRules") or data.get("pcc_rules")
        qos_decs = data.get("qosDecs") or data.get("qos_decs")
        if not isinstance(pcc_rules, dict) or not pcc_rules:
            raise ValueError("SmPolicyDecision 缺少 pccRules")
        if not isinstance(qos_decs, dict) or not qos_decs:
            raise ValueError("SmPolicyDecision 缺少 qosDecs")

        _, selected_pcc = self._pick_mapping_entry(pcc_rules, flow_id, "pccRuleId")
        _, selected_qos = self._pick_mapping_entry(qos_decs, flow_id, "qosId")

        canonical_flow_id = flow_id or self._extract_flow_id_from_policy_data(data)
        canonical_pcc_id = f"pcc-{canonical_flow_id}" if canonical_flow_id else str(selected_pcc.get("pccRuleId") or "pcc-default")
        canonical_qos_id = f"qos-{canonical_flow_id}" if canonical_flow_id else str(selected_qos.get("qosId") or "qos-default")

        for key in (
            "priorityLevel",
            "packetDelayBudget",
            "packetErrorRate",
            "maxbrUl",
            "maxbrDl",
            "gbrUl",
            "gbrDl",
            "arp",
            "5qi",
            "var5qi",
            "jitterReq",
        ):
            if key in selected_pcc and key not in selected_qos:
                selected_qos[key] = selected_pcc.get(key)
        selected_qos.pop("refQosData", None)

        pcc_payload: Dict[str, Any] = {
            "pccRuleId": canonical_pcc_id,
            "precedence": _coerce_int(selected_pcc.get("precedence")) or _coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1,
            "refQosData": [canonical_qos_id],
        }
        flow_infos = self._build_flow_infos(flow_ctx, selected_pcc, canonical_flow_id)
        if flow_infos:
            pcc_payload["flowInfos"] = flow_infos
        if flow_ctx and flow_ctx.get("app_id"):
            pcc_payload["appId"] = _normalize_app_id(flow_ctx.get("app_id"))
        elif selected_pcc.get("appId"):
            pcc_payload["appId"] = _normalize_app_id(selected_pcc.get("appId"))
        for key in (
            "appDescriptor",
            "contVer",
            "afSigProtocol",
            "appReloc",
            "easRedisInd",
            "refAltQosParams",
            "refTcData",
            "refChgData",
            "refChgN3gData",
            "refUmData",
            "refUmN3gData",
            "refCondData",
            "refQosMon",
            "addrPreserInd",
            "tscaiInputDl",
            "tscaiInputUl",
            "tscaiTimeDom",
            "ddNotifCtrl",
            "ddNotifCtrl2",
            "disUeNotif",
            "packFiltAllPrec",
        ):
            value = selected_pcc.get(key)
            if value not in (None, "", [], {}):
                pcc_payload[key] = value

        qos_payload: Dict[str, Any] = {
            "qosId": canonical_qos_id,
            "priorityLevel": _coerce_int(selected_qos.get("priorityLevel")) or _coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1,
            "packetDelayBudget": _coerce_int(selected_qos.get("packetDelayBudget")) or _coerce_int(flow_ctx.get("lat") if flow_ctx else None) or 0,
            "packetErrorRate": _coerce_str(selected_qos.get("packetErrorRate")) or _coerce_str(flow_ctx.get("loss_req") if flow_ctx else None) or "0.0",
        }
        for qos_key, flow_key in (
            ("maxbrUl", "bw_ul"),
            ("maxbrDl", "bw_dl"),
            ("gbrUl", "gbr_ul"),
            ("gbrDl", "gbr_dl"),
        ):
            qos_payload[qos_key] = _coerce_str(selected_qos.get(qos_key)) or _coerce_str(flow_ctx.get(flow_key) if flow_ctx else None)
        var5qi = selected_qos.get("var5qi", selected_qos.get("5qi"))
        if var5qi not in (None, ""):
            qos_payload["var5qi"] = _coerce_int(var5qi)
        arp = selected_qos.get("arp")
        if isinstance(arp, dict):
            arp = _json_friendly(arp)
            arp["priorityLevel"] = _coerce_int(arp.get("priorityLevel")) or qos_payload["priorityLevel"]
            qos_payload["arp"] = arp
        for key in (
            "qnc",
            "averWindow",
            "maxDataBurstVol",
            "reflectiveQos",
            "sharingKeyDl",
            "sharingKeyUl",
            "maxPacketLossRateDl",
            "maxPacketLossRateUl",
            "defQosFlowIndication",
            "extMaxDataBurstVol",
        ):
            value = selected_qos.get(key)
            if value not in (None, "", [], {}):
                qos_payload[key] = value

        normalized: Dict[str, Any] = {
            "pccRules": {canonical_pcc_id: pcc_payload},
            "qosDecs": {canonical_qos_id: qos_payload},
        }

        sess_rules = data.get("sessRules") or data.get("sess_rules")
        if canonical_flow_id and isinstance(sess_rules, dict) and sess_rules:
            _, selected_sess = self._pick_mapping_entry(sess_rules, canonical_flow_id, "sessRuleId")
            canonical_sess_id = f"sess-{canonical_flow_id}"
            selected_sess["sessRuleId"] = canonical_sess_id
            normalized["sessRules"] = {canonical_sess_id: selected_sess}

        validated = SmPolicyDecision.model_validate(_json_friendly(normalized))
        return _json_friendly(validated)

    def _normalize_ursp_policy_details(self, details: Dict[str, Any], target_type: str) -> Tuple[Dict[str, Any], str]:
        data = _json_friendly(details)
        flow_ctx = data.pop("_flow_ctx", None) if isinstance(data, dict) else None
        if not isinstance(flow_ctx, dict):
            flow_ctx = None

        traffic_desc = self._build_traffic_desc(data, flow_ctx)
        route_sets = self._build_route_selection_parameter_sets(data, flow_ctx, traffic_desc)
        relat_precedence = _coerce_int(data.get("relatPrecedence"))
        if relat_precedence is None and route_sets:
            relat_precedence = _coerce_int(route_sets[0].get("precedence")) or 1

        if target_type == "flow" and not traffic_desc:
            target_type = "app"

        ursp_payload: Dict[str, Any] = {
            "relatPrecedence": relat_precedence or 1,
            "routeSelParamSets": route_sets,
        }
        if traffic_desc:
            ursp_payload["trafficDesc"] = traffic_desc

        validated = UrspRuleRequest.model_validate(ursp_payload)
        return _json_friendly(validated), target_type

    def _normalize_output_strategy(self, output: OutputStrategy, user_intent: Dict[str, Any]) -> OutputStrategy:
        normalized_user_intent = _json_friendly(user_intent)
        normalized_user_intent["app_id"] = _normalize_app_id(normalized_user_intent.get("app_id"))
        user_flows = normalized_user_intent.get("flows") or []
        flow_map = {}
        for flow in user_flows:
            if isinstance(flow, dict) and flow.get("flow_id"):
                normalized_flow = dict(flow)
                normalized_flow["app_id"] = normalized_user_intent.get("app_id")
                flow_map[str(flow.get("flow_id"))] = normalized_flow
        single_flow_id = next(iter(flow_map.keys())) if len(flow_map) == 1 else None
        base_supi = str(
            output.supi
            or normalized_user_intent.get("supi")
            or next((flow.get("supi") for flow in user_flows if isinstance(flow, dict) and flow.get("supi")), "")
            or ""
        ).strip()
        base_app_id = _normalize_app_id(normalized_user_intent.get("app_id"))

        normalized_policies: List[Strategy] = []
        for index, strategy in enumerate(output.all_policies):
            details_payload = _json_friendly(strategy.policy_details)
            inferred_flow_id = strategy.flow_id or self._extract_flow_id_from_policy_data(details_payload) or single_flow_id
            flow_ctx = flow_map.get(str(inferred_flow_id)) if inferred_flow_id else None
            supi = str(strategy.supi or base_supi or (flow_ctx.get("supi") if flow_ctx else "") or "").strip()
            app_id = _normalize_app_id(strategy.app_id or base_app_id or "")
            target_type = strategy.target_type or ("flow" if inferred_flow_id else "app")

            if strategy.policy_type == "SmPolicyDecision":
                policy_details = self._normalize_sm_policy_details(details_payload, inferred_flow_id, flow_ctx)
            elif strategy.policy_type == "UrspRuleRequest":
                details_payload["_flow_ctx"] = flow_ctx
                policy_details, target_type = self._normalize_ursp_policy_details(details_payload, target_type)
            else:
                raise ValueError(f"Unsupported policy_type: {strategy.policy_type}")

            policy_id = _build_policy_id(strategy.policy_type, app_id or "app", inferred_flow_id, target_type)
            normalized_policies.append(
                Strategy(
                    recommended_actions=list(strategy.recommended_actions or []),
                    supi=supi,
                    app_id=app_id,
                    flow_id=inferred_flow_id,
                    target_type=target_type,
                    policy_id=policy_id,
                    policy_type=strategy.policy_type,
                    policy_details=policy_details,
                )
            )

            if target_type == "flow" and strategy.policy_type == "SmPolicyDecision" and not inferred_flow_id:
                raise ValueError(f"策略 #{index + 1} 缺少 flow_id")

        return OutputStrategy(supi=base_supi, all_policies=normalized_policies)

    def _fallback_output(self, user_intent: Dict[str, Any], message: str) -> OutputStrategy:
        supi = str(
            user_intent.get("supi")
            or next((flow.get("supi") for flow in user_intent.get("flows", []) if isinstance(flow, dict) and flow.get("supi")), "")
            or ""
        ).strip()
        app_id = _normalize_app_id(user_intent.get("app_id"))
        flow_id = None
        flows = user_intent.get("flows") or []
        if len(flows) == 1 and isinstance(flows[0], dict):
            flow_id = flows[0].get("flow_id")

        fallback_details = _json_friendly(SmPolicyDecision(pccRules={}, qosDecs={}))
        return OutputStrategy(
            supi=supi,
            all_policies=[
                Strategy(
                    recommended_actions=[message],
                    supi=supi,
                    app_id=app_id,
                    flow_id=flow_id,
                    target_type="flow" if flow_id else "app",
                    policy_id=_build_policy_id("SmPolicyDecision", app_id or "app", flow_id, "flow" if flow_id else "app"),
                    policy_type="SmPolicyDecision",
                    policy_details=fallback_details,
                )
            ],
        )

    def generate_strategy(self, user_intent: Dict[str, Any]) -> OutputStrategy:
        normalized_user_intent = _json_friendly(user_intent)
        normalized_user_intent["app_id"] = _normalize_app_id(normalized_user_intent.get("app_id"))
        intent_json = json.dumps(normalized_user_intent, ensure_ascii=False)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", OSA_SYSTEM_PROMPT),
                ("user", "用户意图:\n{intent_json}\n\n请先获取网络状态，再执行优化并返回结构化策略。"),
            ]
        )
        messages = prompt.format_messages(
            intent_json=intent_json,
            format_instructions=self.output_parser.get_format_instructions(),
        )

        total_start = time.perf_counter()
        log_event(self.logger, "osa_generate_start")
        try:
            for iteration in range(5):
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)

                if not getattr(response, "tool_calls", None):
                    log_event(self.logger, "osa_parse_output_start", iteration=iteration + 1)
                    try:
                        payload = _extract_json_payload(response.content)
                        final_output = self.output_parser.parse(payload)
                        normalized_output = self._normalize_output_strategy(final_output, normalized_user_intent)
                        log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="success")
                        return normalized_output
                    except Exception as parse_err:
                        self.logger.error(f"解析最终输出失败: {parse_err}. Content: {response.content}")
                        log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="parse_failed")
                        return self._fallback_output(normalized_user_intent, "Error parsing or normalizing agent output")

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    log_args = self._normalize_tool_args_for_log(tool_args)
                    self.logger.info(
                        f"LLM 决定调用工具: {tool_name}\n参数: {json.dumps(log_args, ensure_ascii=False, indent=2)}"
                    )

                    if tool_name in self.tool_map:
                        tool_instance = self.tool_map[tool_name]
                        try:
                            tool_start = time.perf_counter()
                            tool_result = tool_instance.invoke(tool_args)
                            log_timing(self.logger, "osa_tool_call", time.perf_counter() - tool_start, tool=tool_name)
                        except Exception as exc:
                            tool_result = f"工具执行异常: {exc}"
                    else:
                        tool_result = f"Error: Tool {tool_name} not found."

                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

            self.logger.warning("达到最大迭代次数。")
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="max_iterations")
            return self._fallback_output(normalized_user_intent, "Max iterations reached")

        except Exception as exc:
            self.logger.error(f"策略生成出错: {exc}")
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="error")
            return self._fallback_output(normalized_user_intent, f"Agent execution error: {exc}")
