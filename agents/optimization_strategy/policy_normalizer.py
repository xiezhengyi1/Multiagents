from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from agents.tools.db_tool import build_flow_description_from_five_tuple, build_flow_info_from_five_tuple
from domain.policy_plan import OperationIntent, PolicyDraft, PolicyPlanDraft
from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest


def json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def strip_rule_prefix(candidate: Any) -> Optional[str]:
    text = str(candidate or "").strip()
    if not text:
        return None
    for prefix in ("pcc-", "qos-", "sess-", "smp-", "ursp-"):
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix) :]
    return text


def build_policy_id(policy_type: str, app_id: str, flow_id: Optional[str], target_type: str) -> str:
    if policy_type == "PcfAmPolicyControlPolicyAssociation":
        if flow_id:
            return f"amp-{flow_id}"
        if app_id:
            return f"amp-{app_id}"
        return "amp-ue"
    prefix = "ursp" if policy_type == "UrspRuleRequest" else "smp"
    return f"{prefix}-{app_id}-{flow_id}" if target_type == "flow" and flow_id else f"{prefix}-{app_id}"


def normalize_app_id(app_id: Any) -> str:
    text = str(app_id or "").strip()
    if not text:
        return ""
    if text.startswith(("app_", "app-")):
        return f"app-{text[4:].replace('_', '-')}"
    if re.fullmatch(r"app\d+", text, flags=re.IGNORECASE):
        return f"app-{text[3:]}"
    return text.replace("_", "-")


def coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_str(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


def coerce_numeric_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def extract_ursp_flow_desc_from_five_tuple(five_tuple: Any) -> Optional[str]:
    if not isinstance(five_tuple, (list, tuple)) or len(five_tuple) != 5:
        return None
    _, dst_ip, _, dst_port, protocol = five_tuple
    dst_ip, protocol = str(dst_ip or "").strip(), str(protocol or "").strip().upper()
    try:
        dst_port = int(dst_port)
    except (TypeError, ValueError):
        return None
    return f"{protocol} {dst_ip} {dst_port}" if dst_ip and protocol else None


def extract_flow_id_from_policy_data(data: Dict[str, Any]) -> Optional[str]:
    if flow_id := strip_rule_prefix(data.get("flow_id") or data.get("flowId")):
        return flow_id

    for mapping_name, id_field in (("qosDecs", "qosId"), ("pccRules", "pccRuleId"), ("sessRules", "sessRuleId")):
        mapping = data.get(mapping_name) or data.get(mapping_name[0].lower() + mapping_name[1:])
        if isinstance(mapping, dict):
            for key, item in mapping.items():
                if key_flow_id := strip_rule_prefix(key):
                    return key_flow_id
                if isinstance(item, dict) and (item_flow_id := strip_rule_prefix(item.get(id_field))):
                    return item_flow_id
    return None


def pick_mapping_entry(mapping: Dict[str, Any], flow_id: Optional[str], id_field: str) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(mapping, dict) or not mapping:
        return "", {}

    if flow_id:
        for key, item in mapping.items():
            if strip_rule_prefix(key) == flow_id or (isinstance(item, dict) and strip_rule_prefix(item.get(id_field)) == flow_id):
                return str(key), json_friendly(item)

    first_key = next(iter(mapping.keys()))
    return str(first_key), json_friendly(mapping[first_key])


def build_flow_infos(
    flow_ctx: Optional[Dict[str, Any]],
    selected_pcc: Dict[str, Any],
    canonical_flow_id: Optional[str],
) -> List[Dict[str, Any]]:
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
            return json_friendly(existing)

    return [{"flowDescription": canonical_flow_id, "flowDirection": "BIDIRECTIONAL"}] if canonical_flow_id else []


def build_traffic_desc(details: Dict[str, Any], flow_ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    top_level = details.get("trafficDesc")
    nested = None
    route_sets = details.get("routeSelParamSets")
    if isinstance(route_sets, list):
        for route_set in route_sets:
            if isinstance(route_set, dict) and isinstance(route_set.get("trafficDesc"), dict):
                nested = route_set.get("trafficDesc")
                break

    traffic_desc = json_friendly(top_level) if isinstance(top_level, dict) else {}
    if not traffic_desc and isinstance(nested, dict):
        traffic_desc = json_friendly(nested)
    if not traffic_desc and flow_ctx:
        flow_desc = extract_ursp_flow_desc_from_five_tuple(flow_ctx.get("five_tuple"))
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
        flow_desc = extract_ursp_flow_desc_from_five_tuple(flow_ctx.get("five_tuple"))
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


def build_route_selection_parameter_sets(
    details: Dict[str, Any],
    flow_ctx: Optional[Dict[str, Any]],
    traffic_desc: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def normalize_snssai(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        text = str(value or "").strip()
        if len(text) < 3 or not text.isdigit():
            return value
        return {"sst": int(text[:2]), "sd": text[2:8]} if len(text) >= 8 else {"sst": int(text[:1]), "sd": text[1:] or None}

    precedence = coerce_int(details.get("relatPrecedence")) or coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1
    default_dnn = dnns[0] if isinstance(traffic_desc, dict) and isinstance(dnns := traffic_desc.get("dnns"), list) and dnns else None

    normalized_route_sets = [
        {
            "dnn": coerce_str(rs.get("dnn")) or default_dnn or "default",
            "precedence": coerce_int(rs.get("precedence")) or coerce_int(rs.get("priority")) or precedence,
            **({"snssai": snssai} if (snssai := normalize_snssai(rs.get("snssai"))) is not None else {}),
        }
        for rs in (details.get("routeSelParamSets") or [])
        if isinstance(rs, dict)
    ]

    if normalized_route_sets:
        return normalized_route_sets

    default_rs: Dict[str, Any] = {"dnn": default_dnn or "default", "precedence": precedence}
    if (snssai := normalize_snssai(details.get("snssai"))) is not None:
        default_rs["snssai"] = snssai
    return [default_rs]


def normalize_sm_policy_details(details: Dict[str, Any], flow_id: Optional[str], flow_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = json_friendly(details)
    preserve_explicit = bool(data.pop("_preserve_explicit_qos_values", False))
    if not isinstance(pcc := data.get("pccRules") or data.get("pcc_rules"), dict) or not pcc:
        raise ValueError("SmPolicyDecision is missing pccRules.")
    if not isinstance(qos := data.get("qosDecs") or data.get("qos_decs"), dict) or not qos:
        raise ValueError("SmPolicyDecision is missing qosDecs.")

    _, sel_pcc = pick_mapping_entry(pcc, flow_id, "pccRuleId")
    _, sel_qos = pick_mapping_entry(qos, flow_id, "qosId")

    c_flow_id = flow_id or extract_flow_id_from_policy_data(data)
    c_pcc_id = f"pcc-{c_flow_id}" if c_flow_id else str(sel_pcc.get("pccRuleId") or "pcc-default")
    c_qos_id = f"qos-{c_flow_id}" if c_flow_id else str(sel_qos.get("qosId") or "qos-default")

    for k in ("priorityLevel", "packetDelayBudget", "packetErrorRate", "maxbrUl", "maxbrDl", "gbrUl", "gbrDl", "arp", "5qi", "var5qi", "jitterReq"):
        if k in sel_pcc and k not in sel_qos:
            sel_qos[k] = sel_pcc[k]
    sel_qos.pop("refQosData", None)

    pcc_payload = {
        "pccRuleId": c_pcc_id,
        "precedence": coerce_int(sel_pcc.get("precedence")) or coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1,
        "refQosData": [c_qos_id],
    }
    if infos := build_flow_infos(flow_ctx, sel_pcc, c_flow_id):
        pcc_payload["flowInfos"] = infos

    if (app_id := (flow_ctx.get("app_id") if flow_ctx else None) or sel_pcc.get("appId")):
        pcc_payload["appId"] = normalize_app_id(app_id)

    for k in (
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
        if (val := sel_pcc.get(k)) not in (None, "", [], {}):
            pcc_payload[k] = val

    if preserve_explicit:
        priority_level = coerce_int(sel_qos.get("priorityLevel")) or coerce_int(flow_ctx.get("priority") if flow_ctx else None) or 1
        packet_delay_budget = coerce_int(sel_qos.get("packetDelayBudget")) or coerce_int(flow_ctx.get("lat") if flow_ctx else None) or 0
        packet_error_rate = coerce_str(sel_qos.get("packetErrorRate")) or coerce_str(flow_ctx.get("loss_req") if flow_ctx else None) or "0.0"
    else:
        priority_level = coerce_int(flow_ctx.get("priority") if flow_ctx else None) or coerce_int(sel_qos.get("priorityLevel")) or 1
        packet_delay_budget = coerce_int(flow_ctx.get("lat") if flow_ctx else None) or coerce_int(sel_qos.get("packetDelayBudget")) or 0
        packet_error_rate = coerce_str(flow_ctx.get("loss_req") if flow_ctx else None) or coerce_str(sel_qos.get("packetErrorRate")) or "0.0"

    qos_payload = {
        "qosId": c_qos_id,
        "priorityLevel": priority_level,
        "packetDelayBudget": packet_delay_budget,
        "packetErrorRate": packet_error_rate,
    }
    for qk, fk in (("maxbrUl", "bw_ul"), ("maxbrDl", "bw_dl"), ("gbrUl", "gbr_ul"), ("gbrDl", "gbr_dl")):
        if preserve_explicit:
            qos_payload[qk] = coerce_numeric_str(sel_qos.get(qk)) or coerce_numeric_str(flow_ctx.get(fk) if flow_ctx else None)
        else:
            qos_payload[qk] = coerce_numeric_str(flow_ctx.get(fk) if flow_ctx else None) or coerce_numeric_str(sel_qos.get(qk))

    if (var5qi := sel_qos.get("var5qi", sel_qos.get("5qi"))) not in (None, ""):
        qos_payload["var5qi"] = coerce_int(var5qi)

    if isinstance(arp := sel_qos.get("arp"), dict):
        arp_payload = json_friendly(arp)
        arp_payload["priorityLevel"] = coerce_int(arp_payload.get("priorityLevel")) or qos_payload["priorityLevel"]
        if "preemptCap" not in arp_payload:
            raw_cap = arp_payload.get("preemptionCapability")
            if isinstance(raw_cap, bool):
                arp_payload["preemptCap"] = "MAY_PREEMPT" if raw_cap else "NOT_PREEMPT"
            elif raw_cap not in (None, ""):
                arp_payload["preemptCap"] = raw_cap
        if "preemptVuln" not in arp_payload:
            raw_vuln = arp_payload.get("preemptionVulnerability")
            if isinstance(raw_vuln, bool):
                arp_payload["preemptVuln"] = "PREEMPTABLE" if raw_vuln else "NOT_PREEMPTABLE"
            elif raw_vuln not in (None, ""):
                arp_payload["preemptVuln"] = raw_vuln
        qos_payload["arp"] = arp_payload

    for k in ("qnc", "averWindow", "maxDataBurstVol", "reflectiveQos", "sharingKeyDl", "sharingKeyUl", "maxPacketLossRateDl", "maxPacketLossRateUl", "defQosFlowIndication", "extMaxDataBurstVol"):
        if (val := sel_qos.get(k)) not in (None, "", [], {}):
            qos_payload[k] = val

    normalized: Dict[str, Any] = {"pccRules": {c_pcc_id: pcc_payload}, "qosDecs": {c_qos_id: qos_payload}}

    if c_flow_id and isinstance(sess := data.get("sessRules") or data.get("sess_rules"), dict) and sess:
        _, sel_sess = pick_mapping_entry(sess, c_flow_id, "sessRuleId")
        sel_sess["sessRuleId"] = c_sess_id = f"sess-{c_flow_id}"
        normalized["sessRules"] = {c_sess_id: sel_sess}

    return json_friendly(SmPolicyDecision.model_validate(json_friendly(normalized)))


def normalize_ursp_policy_details(details: Dict[str, Any], target_type: str) -> Tuple[Dict[str, Any], str]:
    data = json_friendly(details)
    flow_ctx = data.pop("_flow_ctx", None) if isinstance(data, dict) else None
    if not isinstance(flow_ctx, dict):
        flow_ctx = None

    traffic_desc = build_traffic_desc(data, flow_ctx)
    route_sets = build_route_selection_parameter_sets(data, flow_ctx, traffic_desc)
    relat_precedence = coerce_int(data.get("relatPrecedence")) or (coerce_int(route_sets[0].get("precedence")) if route_sets else None) or 1

    if target_type == "flow" and not traffic_desc:
        target_type = "app"

    ursp_payload: Dict[str, Any] = {"relatPrecedence": relat_precedence, "routeSelParamSets": route_sets}
    if traffic_desc:
        ursp_payload["trafficDesc"] = traffic_desc

    return json_friendly(UrspRuleRequest.model_validate(ursp_payload)), target_type


def normalize_am_policy_details(details: Dict[str, Any], supi: str) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("PcfAmPolicyControlPolicyAssociation is missing policy_details.")

    request = data.get("request")
    if not isinstance(request, dict):
        raise ValueError("PcfAmPolicyControlPolicyAssociation requires a request object.")
    request.setdefault("supi", supi)
    request.setdefault("suppFeat", "1")
    data.setdefault("request", request)
    data.setdefault("suppFeat", "1")
    return json_friendly(PcfAmPolicyControlPolicyAssociation.model_validate(data))


def normalize_policy_plan_draft(draft: PolicyPlanDraft, operation_intent: OperationIntent) -> PolicyPlanDraft:
    ui = json_friendly(operation_intent.model_dump(mode="json"))
    base_app_id = ui["app_id"] = normalize_app_id(ui.get("app_id"))

    flow_map = {
        str(f["flow_id"]): {**f, "app_id": base_app_id}
        for f in ui.get("flows", [])
        if isinstance(f, dict) and f.get("flow_id")
    }
    single_flow_id = next(iter(flow_map.keys())) if len(flow_map) == 1 else None

    base_supi = str(
        draft.supi
        or ui.get("supi")
        or next((f.get("supi") for f in ui.get("flows", []) if isinstance(f, dict) and f.get("supi")), "")
        or ""
    ).strip()

    normalized_policies = []
    for i, pd in enumerate(draft.all_policies):
        details = json_friendly(pd.policy_details)
        f_id = pd.flow_id or extract_flow_id_from_policy_data(details) or single_flow_id
        flow_ctx = flow_map.get(str(f_id)) if f_id else None

        supi = str(pd.supi or base_supi or (flow_ctx.get("supi") if flow_ctx else "") or "").strip()
        app_id = normalize_app_id(pd.app_id or base_app_id or "")
        target_type = pd.target_type or ("flow" if f_id else "app")

        if pd.policy_type == "SmPolicyDecision":
            norm_details = normalize_sm_policy_details(details, f_id, flow_ctx)
        elif pd.policy_type == "UrspRuleRequest":
            details["_flow_ctx"] = flow_ctx
            norm_details, target_type = normalize_ursp_policy_details(details, target_type)
        elif pd.policy_type == "PcfAmPolicyControlPolicyAssociation":
            target_type = "ue"
            norm_details = normalize_am_policy_details(details, supi)
            f_id = None
            app_id = ""
        else:
            raise ValueError(f"Unsupported policy_type: {pd.policy_type}")

        if target_type == "flow" and pd.policy_type == "SmPolicyDecision" and not f_id:
            raise ValueError(f"Policy #{i + 1} is missing flow_id for a flow-scoped SmPolicyDecision.")

        normalized_policies.append(
            PolicyDraft(
                recommended_actions=list(pd.recommended_actions or []),
                supi=supi,
                app_id=app_id,
                flow_id=f_id,
                target_type=target_type,
                policy_id=str(
                    (norm_details.get("request", {}) if isinstance(norm_details.get("request"), dict) else {}).get("supi")
                    or build_policy_id(pd.policy_type, app_id or supi or "app", f_id, target_type)
                ).strip()
                if pd.policy_type == "PcfAmPolicyControlPolicyAssociation"
                else build_policy_id(pd.policy_type, app_id or supi or "app", f_id, target_type),
                policy_type=pd.policy_type,
                policy_details=norm_details,
            )
        )

    return PolicyPlanDraft(
        supi=base_supi,
        session_id=str(draft.session_id or ui.get("session_id") or "").strip(),
        snapshot_id=str(draft.snapshot_id or ui.get("snapshot_id") or "").strip(),
        planning_metadata=json_friendly(draft.planning_metadata),
        all_policies=normalized_policies,
    )


__all__ = [
    "json_friendly",
    "normalize_app_id",
    "normalize_policy_plan_draft",
]
