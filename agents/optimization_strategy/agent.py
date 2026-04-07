from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel

from agent_runtime import AgentRuntimeContext, ArtifactEnvelope
from agents.BaseAgent import BaseAgent
from agents.worker import ArtifactWorkerMixin
from domain.collaboration import PlanningRequest
from domain.policy_plan import OperationIntent, PolicyDraft, PolicyPlanDraft
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest
from agents.tools import think
from agents.tools.db_tool import (
    build_flow_description_from_five_tuple,
    build_flow_info_from_five_tuple,
)
from agents.tools.network_status import get_network_status_summary
from agents.tools.optimizer import optimize_network_slices
from utils.logger import log_event, log_timing, setup_logger

from .prompts import OSA_SYSTEM_PROMPT

logger = setup_logger(__name__)


@tool
def fetch_network_status(
    flow_type_id: int,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """Fetch network status for a service type."""
    try:
        status = get_network_status_summary(flow_type_id=flow_type_id)
        if runtime is not None:
            ctx = runtime.context
            logger.info(
                "Fetched network status for agent=%s session=%s snapshot=%s",
                ctx.agent_name,
                ctx.session_id,
                ctx.snapshot_id,
            )
        else:
            logger.info("Fetched network status.")
        return status
    except Exception as exc:
        return f"Failed to fetch network status: {exc}"


@tool
def run_optimization_solver(
    w1: float,
    w2: float,
    w3: float,
    mode: str,
    app_details: str,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Invoke the network optimizer and return a JSON string payload.
    """
    try:
        app_data = json.loads(app_details) if isinstance(app_details, str) else app_details
        result = optimize_network_slices(app_data, w1, w2, w3, mode=mode)
        meta = result.get("meta", {}) if isinstance(result, dict) else {}
        if runtime is not None:
            ctx = runtime.context
            logger.info(
                "Optimization solver completed for agent=%s session=%s snapshot=%s status=%s",
                ctx.agent_name,
                ctx.session_id,
                ctx.snapshot_id,
                meta.get("status"),
            )
        else:
            logger.info("Optimization solver completed with status=%s", meta.get("status"))
        return json.dumps(_json_friendly(result), ensure_ascii=False)
    except Exception as exc:
        return f"Optimization solver failed: {exc}"


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


def _strip_rule_prefix(candidate: Any) -> Optional[str]:
    """移除规则ID中的常见前缀。"""
    text = str(candidate or "").strip()
    if not text:
        return None
    for prefix in ("pcc-", "qos-", "sess-", "smp-", "ursp-"):
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix) :]
    return text


def _build_policy_id(policy_type: str, app_id: str, flow_id: Optional[str], target_type: str) -> str:
    """构建标准化的策略ID。"""
    prefix = "ursp" if policy_type == "UrspRuleRequest" else "smp"
    return f"{prefix}-{app_id}-{flow_id}" if target_type == "flow" and flow_id else f"{prefix}-{app_id}"


def _normalize_app_id(app_id: Any) -> str:
    """将应用ID格式化为标准 app-xxx 格式。"""
    text = str(app_id or "").strip()
    if not text:
        return ""
    if text.startswith(("app_", "app-")):
        return f"app-{text[4:].replace('_', '-')}"
    if re.fullmatch(r"app\d+", text, flags=re.IGNORECASE):
        return f"app-{text[3:]}"
    return text.replace("_", "-")


def _coerce_int(value: Any) -> Optional[int]:
    """尝试将值转换为整数。"""
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    """尝试将值转换为字符串。"""
    return None if value in (None, "") else str(value)


def _coerce_numeric_str(value: Any) -> Optional[str]:
    """尝试将数值转换为字符串，保留原有格式。"""
    if value in (None, ""):
        return None
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def _extract_ursp_flow_desc_from_five_tuple(five_tuple: Any) -> Optional[str]:
    """从五元组中提取 URSP 流描述符。"""
    if not isinstance(five_tuple, (list, tuple)) or len(five_tuple) != 5:
        return None
    _, dst_ip, _, dst_port, protocol = five_tuple
    dst_ip, protocol = str(dst_ip or "").strip(), str(protocol or "").strip().upper()
    try:
        dst_port = int(dst_port)
    except (TypeError, ValueError):
        return None
    return f"{protocol} {dst_ip} {dst_port}" if dst_ip and protocol else None


class OptimizationStrategyAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "optimization_strategy"

    def __init__(self, model_name: str = "qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.agent_name = "optimization_strategy"
        self.initialize_agent_runtime(logger_color="\033[94m")
        self.tools = [think, fetch_network_status, run_optimization_solver]
        self.agent = self.create_json_agent(
            tools=self.tools,
            system_prompt=OSA_SYSTEM_PROMPT,
            response_model=PolicyPlanDraft,
        )

    def expected_request_type(self) -> str:
        return "PlanningRequest"

    def response_artifact_type(self) -> str:
        return "PolicyPlanDraft"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> PolicyPlanDraft:
        planning_request = PlanningRequest.model_validate(envelope.payload)
        return self.generate_strategy_from_request(
            planning_request=planning_request,
            request_envelope=envelope,
        )

    def _cache_received_request(self, planning_request: PlanningRequest) -> ArtifactEnvelope:
        return self.cache_received_artifact(
            artifact_type="PlanningRequest",
            payload=planning_request,
            session_id=planning_request.context.session_id,
            snapshot_id=planning_request.context.snapshot_id,
        )

    def _cache_produced_result(
        self,
        *,
        request_envelope: ArtifactEnvelope,
        policy_plan: PolicyPlanDraft,
    ) -> None:
        self.cache_produced_artifact(
            artifact_type="PolicyPlanDraft",
            request_envelope=request_envelope,
            payload=policy_plan,
        )

    @staticmethod
    def _extract_flow_id_from_policy_data(data: Dict[str, Any]) -> Optional[str]:
        """尝试从顶层或内嵌的规则字典中提取flow_id。"""
        if flow_id := _strip_rule_prefix(data.get("flow_id") or data.get("flowId")):
            return flow_id

        for mapping_name, id_field in (("qosDecs", "qosId"), ("pccRules", "pccRuleId"), ("sessRules", "sessRuleId")):
            mapping = data.get(mapping_name) or data.get(mapping_name[0].lower() + mapping_name[1:])
            if isinstance(mapping, dict):
                for key, item in mapping.items():
                    if key_flow_id := _strip_rule_prefix(key):
                        return key_flow_id
                    if isinstance(item, dict) and (item_flow_id := _strip_rule_prefix(item.get(id_field))):
                        return item_flow_id
        return None

    @staticmethod
    def _pick_mapping_entry(mapping: Dict[str, Any], flow_id: Optional[str], id_field: str) -> Tuple[str, Dict[str, Any]]:
        """从映射中选取匹配指定flow_id的项，或默认第一项。"""
        if not isinstance(mapping, dict) or not mapping:
            return "", {}

        if flow_id:
            for key, item in mapping.items():
                if _strip_rule_prefix(key) == flow_id or (isinstance(item, dict) and _strip_rule_prefix(item.get(id_field)) == flow_id):
                    return str(key), _json_friendly(item)

        first_key = next(iter(mapping.keys()))
        return str(first_key), _json_friendly(mapping[first_key])

    @staticmethod
    def _build_flow_infos(
        flow_ctx: Optional[Dict[str, Any]],
        selected_pcc: Dict[str, Any],
        canonical_flow_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """基于上下文/解析的规则生成 flowInfos。"""
        if flow_ctx:
            five_tuple = flow_ctx.get("five_tuple")
            if info := build_flow_info_from_five_tuple(five_tuple):
                return [info]
            if desc := build_flow_description_from_five_tuple(five_tuple):
                return [{"flowDescription": desc, "flowDirection": "BIDIRECTIONAL"}]
            if fallback_desc := str(flow_ctx.get("description") or flow_ctx.get("name") or canonical_flow_id or "flow").strip():
                return [{"flowDescription": fallback_desc, "flowDirection": "BIDIRECTIONAL"}]

        if existing := selected_pcc.get("flowInfos"):
            if isinstance(existing, list):
                return _json_friendly(existing)

        return [{"flowDescription": canonical_flow_id, "flowDirection": "BIDIRECTIONAL"}] if canonical_flow_id else []

    def _build_traffic_desc(self, details: Dict[str, Any], flow_ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        top_level = details.get("trafficDesc")
        nested = None
        route_sets = details.get("routeSelParamSets")
        if isinstance(route_sets, list):
            for route_set in route_sets:
                if isinstance(route_set, dict) and isinstance(route_set.get("trafficDesc"), dict):
                    nested = route_set.get("trafficDesc")
                    break

        traffic_desc = _json_friendly(top_level) if isinstance(top_level, dict) else {}
        if not traffic_desc and isinstance(nested, dict):
            traffic_desc = _json_friendly(nested)
        if not traffic_desc and flow_ctx:
            flow_desc = _extract_ursp_flow_desc_from_five_tuple(flow_ctx.get("five_tuple"))
            if flow_desc:
                traffic_desc = {"flowDescs": [flow_desc]}

        if not traffic_desc:
            return None

        app_descs = traffic_desc.get("appDescs")
        if isinstance(app_descs, dict) and "osId" in app_descs and "appIds" in app_descs:
            os_id = str(app_descs["osId"])
            traffic_desc["appDescs"] = {os_id: {"osId": os_id, "appIds": app_descs["appIds"]}}
        elif isinstance(app_descs, list):
            traffic_desc.pop("appDescs", None)

        flow_descs: List[str] = []
        raw_flow_descs = traffic_desc.get("flowDescs")
        if isinstance(raw_flow_descs, list):
            for item in raw_flow_descs:
                if isinstance(item, str) and item.strip():
                    flow_descs.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                protocol = str(item.get("protocol") or "").strip().upper()
                server_ip = str(item.get("serverIp") or item.get("server_ip") or "").strip()
                server_port = item.get("serverPort") or item.get("server_port")
                if protocol and server_ip and server_port not in (None, ""):
                    flow_descs.append(f"{protocol} {server_ip} {server_port}")
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
        """构建并标准化 routeSelParamSets，提取 DNN 和 S-NSSAI 信息。"""
        def normalize_snssai(value: Any) -> Any:
            if isinstance(value, dict):
                return value
            text = str(value or "").strip()
            if len(text) < 3 or not text.isdigit():
                return value
            return {"sst": int(text[:2]), "sd": text[2:8]} if len(text) >= 8 else {"sst": int(text[:1]), "sd": text[1:] or None}

        precedence = _coerce_int(details.get("relatPrecedence")) or _coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1
        default_dnn = dnns[0] if isinstance(traffic_desc, dict) and isinstance(dnns := traffic_desc.get("dnns"), list) and dnns else None

        normalized_route_sets = [
            {
                "dnn": _coerce_str(rs.get("dnn")) or default_dnn or "default",
                "precedence": _coerce_int(rs.get("precedence")) or _coerce_int(rs.get("priority")) or precedence,
                **({"snssai": snssai} if (snssai := normalize_snssai(rs.get("snssai"))) is not None else {})
            }
            for rs in (details.get("routeSelParamSets") or []) if isinstance(rs, dict)
        ]

        if normalized_route_sets:
            return normalized_route_sets

        default_rs: Dict[str, Any] = {"dnn": default_dnn or "default", "precedence": precedence}
        if (snssai := normalize_snssai(details.get("snssai"))) is not None:
            default_rs["snssai"] = snssai
        return [default_rs]

    def _normalize_sm_policy_details(
        self, details: Dict[str, Any], flow_id: Optional[str], flow_ctx: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """标准化 SM Policy details，合并上下文信息。"""
        data = _json_friendly(details)
        if not isinstance(pcc := data.get("pccRules") or data.get("pcc_rules"), dict) or not pcc:
            raise ValueError("SmPolicyDecision is missing pccRules.")
        if not isinstance(qos := data.get("qosDecs") or data.get("qos_decs"), dict) or not qos:
            raise ValueError("SmPolicyDecision is missing qosDecs.")

        _, sel_pcc = self._pick_mapping_entry(pcc, flow_id, "pccRuleId")
        _, sel_qos = self._pick_mapping_entry(qos, flow_id, "qosId")

        c_flow_id = flow_id or self._extract_flow_id_from_policy_data(data)
        c_pcc_id = f"pcc-{c_flow_id}" if c_flow_id else str(sel_pcc.get("pccRuleId") or "pcc-default")
        c_qos_id = f"qos-{c_flow_id}" if c_flow_id else str(sel_qos.get("qosId") or "qos-default")

        for k in ("priorityLevel", "packetDelayBudget", "packetErrorRate", "maxbrUl", "maxbrDl", "gbrUl", "gbrDl", "arp", "5qi", "var5qi", "jitterReq"):
            if k in sel_pcc and k not in sel_qos:
                sel_qos[k] = sel_pcc[k]
        sel_qos.pop("refQosData", None)

        pcc_payload = {
            "pccRuleId": c_pcc_id,
            "precedence": _coerce_int(sel_pcc.get("precedence")) or _coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1,
            "refQosData": [c_qos_id],
        }
        if (infos := self._build_flow_infos(flow_ctx, sel_pcc, c_flow_id)):
            pcc_payload["flowInfos"] = infos
        
        if (app_id := (flow_ctx.get("app_id") if flow_ctx else None) or sel_pcc.get("appId")):
            pcc_payload["appId"] = _normalize_app_id(app_id)

        for k in ("appDescriptor", "contVer", "afSigProtocol", "appReloc", "easRedisInd", "refAltQosParams", "refTcData", "refChgData", "refChgN3gData", "refUmData", "refUmN3gData", "refCondData", "refQosMon", "addrPreserInd", "tscaiInputDl", "tscaiInputUl", "tscaiTimeDom", "ddNotifCtrl", "ddNotifCtrl2", "disUeNotif", "packFiltAllPrec"):
            if (val := sel_pcc.get(k)) not in (None, "", [], {}):
                pcc_payload[k] = val

        qos_payload = {
            "qosId": c_qos_id,
            "priorityLevel": _coerce_int(flow_ctx.get("priority") if flow_ctx else None) or _coerce_int(sel_qos.get("priorityLevel")) or 1,
            "packetDelayBudget": _coerce_int(flow_ctx.get("lat") if flow_ctx else None) or _coerce_int(sel_qos.get("packetDelayBudget")) or 0,
            "packetErrorRate": _coerce_str(flow_ctx.get("loss_req") if flow_ctx else None) or _coerce_str(sel_qos.get("packetErrorRate")) or "0.0",
        }
        for qk, fk in (("maxbrUl", "bw_ul"), ("maxbrDl", "bw_dl"), ("gbrUl", "gbr_ul"), ("gbrDl", "gbr_dl")):
            qos_payload[qk] = _coerce_numeric_str(flow_ctx.get(fk) if flow_ctx else None) or _coerce_numeric_str(sel_qos.get(qk))

        if (var5qi := sel_qos.get("var5qi", sel_qos.get("5qi"))) not in (None, ""):
            qos_payload["var5qi"] = _coerce_int(var5qi)

        if isinstance(arp := sel_qos.get("arp"), dict):
            arp_payload = _json_friendly(arp)
            arp_payload["priorityLevel"] = _coerce_int(arp_payload.get("priorityLevel")) or qos_payload["priorityLevel"]
            qos_payload["arp"] = arp_payload

        for k in ("qnc", "averWindow", "maxDataBurstVol", "reflectiveQos", "sharingKeyDl", "sharingKeyUl", "maxPacketLossRateDl", "maxPacketLossRateUl", "defQosFlowIndication", "extMaxDataBurstVol"):
            if (val := sel_qos.get(k)) not in (None, "", [], {}):
                qos_payload[k] = val

        normalized: Dict[str, Any] = {"pccRules": {c_pcc_id: pcc_payload}, "qosDecs": {c_qos_id: qos_payload}}

        if c_flow_id and isinstance(sess := data.get("sessRules") or data.get("sess_rules"), dict) and sess:
            _, sel_sess = self._pick_mapping_entry(sess, c_flow_id, "sessRuleId")
            sel_sess["sessRuleId"] = c_sess_id = f"sess-{c_flow_id}"
            normalized["sessRules"] = {c_sess_id: sel_sess}

        return _json_friendly(SmPolicyDecision.model_validate(_json_friendly(normalized)))

    def _normalize_ursp_policy_details(self, details: Dict[str, Any], target_type: str) -> Tuple[Dict[str, Any], str]:
        """标准化 URSP policy details。"""
        data = _json_friendly(details)
        flow_ctx = data.pop("_flow_ctx", None) if isinstance(data, dict) else None
        if not isinstance(flow_ctx, dict):
            flow_ctx = None

        traffic_desc = self._build_traffic_desc(data, flow_ctx)
        route_sets = self._build_route_selection_parameter_sets(data, flow_ctx, traffic_desc)
        relat_precedence = _coerce_int(data.get("relatPrecedence")) or (_coerce_int(route_sets[0].get("precedence")) if route_sets else None) or 1

        if target_type == "flow" and not traffic_desc:
            target_type = "app"

        ursp_payload: Dict[str, Any] = {"relatPrecedence": relat_precedence, "routeSelParamSets": route_sets}
        if traffic_desc:
            ursp_payload["trafficDesc"] = traffic_desc

        return _json_friendly(UrspRuleRequest.model_validate(ursp_payload)), target_type

    def _normalize_policy_plan_draft(
        self, draft: PolicyPlanDraft, operation_intent: OperationIntent
    ) -> PolicyPlanDraft:
        """标准化 PolicyPlanDraft，补全 flow_id/supi/app_id 等关键字段。"""
        ui = _json_friendly(operation_intent.model_dump(mode="json"))
        base_app_id = ui["app_id"] = _normalize_app_id(ui.get("app_id"))
        
        flow_map = {
            str(f["flow_id"]): {**f, "app_id": base_app_id}
            for f in ui.get("flows", []) if isinstance(f, dict) and f.get("flow_id")
        }
        single_flow_id = next(iter(flow_map.keys())) if len(flow_map) == 1 else None

        base_supi = str(draft.supi or ui.get("supi") or next((f.get("supi") for f in ui.get("flows", []) if isinstance(f, dict) and f.get("supi")), "") or "").strip()

        normalized_policies = []
        for i, pd in enumerate(draft.all_policies):
            details = _json_friendly(pd.policy_details)
            f_id = pd.flow_id or self._extract_flow_id_from_policy_data(details) or single_flow_id
            flow_ctx = flow_map.get(str(f_id)) if f_id else None
            
            supi = str(pd.supi or base_supi or (flow_ctx.get("supi") if flow_ctx else "") or "").strip()
            app_id = _normalize_app_id(pd.app_id or base_app_id or "")
            target_type = pd.target_type or ("flow" if f_id else "app")

            if pd.policy_type == "SmPolicyDecision":
                norm_details = self._normalize_sm_policy_details(details, f_id, flow_ctx)
            elif pd.policy_type == "UrspRuleRequest":
                details["_flow_ctx"] = flow_ctx
                norm_details, target_type = self._normalize_ursp_policy_details(details, target_type)
            else:
                raise ValueError(f"Unsupported policy_type: {pd.policy_type}")

            if target_type == "flow" and pd.policy_type == "SmPolicyDecision" and not f_id:
                raise ValueError(f"Policy #{i + 1} is missing flow_id for a flow-scoped SmPolicyDecision.")

            normalized_policies.append(
                PolicyDraft(
                    recommended_actions=list(pd.recommended_actions or []), supi=supi, app_id=app_id,
                    flow_id=f_id, target_type=target_type, policy_id=_build_policy_id(pd.policy_type, app_id or "app", f_id, target_type),
                    policy_type=pd.policy_type, policy_details=norm_details,
                )
            )

        return PolicyPlanDraft(
            supi=base_supi, session_id=str(draft.session_id or ui.get("session_id") or "").strip(),
            snapshot_id=str(draft.snapshot_id or ui.get("snapshot_id") or "").strip(), all_policies=normalized_policies
        )

    def generate_strategy(self, planning_request: PlanningRequest) -> PolicyPlanDraft:
        self.ensure_worker_runtime_initialized()
        if not isinstance(planning_request, PlanningRequest):
            raise TypeError("generate_strategy expects a PlanningRequest instance")
        request_envelope = self._cache_received_request(planning_request)
        return self.generate_strategy_from_request(
            planning_request=planning_request,
            request_envelope=request_envelope,
        )

    def generate_strategy_from_request(
        self,
        *,
        planning_request: PlanningRequest,
        request_envelope: ArtifactEnvelope,
    ) -> PolicyPlanDraft:
        operation_intent = planning_request.operation_intent
        normalized_user_intent = _json_friendly(operation_intent.model_dump(mode="json"))
        normalized_user_intent["app_id"] = _normalize_app_id(normalized_user_intent.get("app_id"))
        coordination_context = _json_friendly(planning_request.context.model_dump(mode="json"))

        total_start = time.perf_counter()
        log_event(self.logger, "osa_generate_start")
        try:
            runtime_context = self.build_runtime_context(
                agent_name=self.agent_name,
                session_id=planning_request.context.session_id,
                snapshot_id=planning_request.context.snapshot_id,
                supi=operation_intent.supi,
                thread_id=planning_request.context.session_id,
            )
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Operation intent:\n"
                        f"{json.dumps(normalized_user_intent, ensure_ascii=False)}\n\n"
                        "Collaboration context:\n"
                        f"{json.dumps(coordination_context, ensure_ascii=False)}\n\n"
                        "Fetch network status first, then run optimization and return a structured policy draft."
                    ),
                }
            ]
            self._pending_invoke_messages = messages
            structured = self.invoke_json_response(
                system_prompt=OSA_SYSTEM_PROMPT,
                user_prompt=messages[-1]["content"],
                response_model=PolicyPlanDraft,
                runtime_context=runtime_context,
            )
            final_output = PolicyPlanDraft.model_validate(structured)
            normalized_output = self._normalize_policy_plan_draft(final_output, operation_intent)
            self._cache_produced_result(
                request_envelope=request_envelope,
                policy_plan=normalized_output,
            )
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="success")
            return normalized_output
        except Exception as exc:
            self.logger.error(f"Failed to generate optimization strategy: {exc}")
            log_timing(self.logger, "osa_total", time.perf_counter() - total_start, status="error")
            raise
        finally:
            if hasattr(self, "_pending_invoke_messages"):
                delattr(self, "_pending_invoke_messages")
