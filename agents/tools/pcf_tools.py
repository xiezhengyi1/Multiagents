import json
import os
import uuid
from typing import Any, Dict, Optional

import requests
from langchain.tools import ToolRuntime, tool

from agents.tools.wrapper_think import tool_with_reason

from agent_runtime.core.context import AgentRuntimeContext
from agents.tools.db_tool import (
    build_flow_info_from_five_tuple,
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    list_am_policy_associations_by_supi,
    search_am_policy_targets_by_context,
    search_flow_targets_by_semantic,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Unified policy execution gateway address used by the local integration environment.
PCF_BASE_URL = str(os.getenv("PCF_BASE_URL", "http://localhost:18080")).rstrip("/")
PCF_FEEDBACK_REQUEST_TIMEOUT_SEC = 5
AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"
POLICY_EXECUTION_PATH = "/policy-executions"


def _trim_ue_context_for_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    trimmed = dict(payload)
    trimmed.pop("app_catalog", None)
    trimmed.pop("flow_catalog", None)
    return trimmed


def _trim_sm_ue_context_for_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    return {
        "supi": str(payload.get("supi") or "").strip(),
        "smPolicyData": payload.get("smPolicyData"),
        "pccRules": payload.get("pccRules"),
        "qosDecs": payload.get("qosDecs"),
        "sessRules": payload.get("sessRules"),
        "traffContDecs": payload.get("traffContDecs"),
        "chgDecs": payload.get("chgDecs"),
        "urspRules": payload.get("urspRules"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _trim_am_policy_context_for_agent(
    payload: Dict[str, Any],
    *,
    association_id: str = "",
    include_associations: bool = True,
    include_access_context: bool = True,
    include_mobility_summary: bool = True,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    normalized_association_id = str(association_id or "").strip()
    am_policy_context = dict(payload.get("amPolicyContext") or {})
    association_map = dict(am_policy_context.get("associations") or {})
    if normalized_association_id:
        association_map = {
            key: value
            for key, value in association_map.items()
            if str(key or "").strip() == normalized_association_id
        }
    if include_associations:
        am_policy_context["associations"] = association_map
    else:
        am_policy_context.pop("associations", None)

    result: Dict[str, Any] = {
        "supi": str(payload.get("supi") or "").strip(),
        "amPolicyContext": am_policy_context,
    }
    if include_access_context:
        result["accessMobilityContext"] = payload.get("accessMobilityContext") or {}
    if include_mobility_summary:
        result["mobilitySummary"] = payload.get("mobilitySummary") or {}
    return result


def _parse_policy_details(policy_json: Any) -> Dict[str, Any]:
    if isinstance(policy_json, dict):
        payload = policy_json
    elif isinstance(policy_json, str):
        try:
            payload = json.loads(policy_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"policy_json is not valid JSON: {exc}") from exc
    else:
        raise ValueError(f"policy_json must be dict or JSON string, got {type(policy_json).__name__}")

    if not isinstance(payload, dict):
        raise ValueError("policy payload must be a JSON object")
    return payload


def _coerce_identifier(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _snssai_from_code(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        sst = value.get("sst")
        sd = value.get("sd")
        if sst in (None, "") or sd in (None, ""):
            return None
        return {"sst": int(sst), "sd": str(sd)}
    text = str(value or "").strip().lower()
    if len(text) < 2 or not text[:2].isdigit():
        return None
    payload: Dict[str, Any] = {"sst": int(text[:2])}
    if len(text) > 2:
        payload["sd"] = text[2:]
    return payload


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_positive_float(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _format_mbps(value: Any) -> str:
    parsed = _to_float(value)
    if parsed is None or parsed <= 0:
        return ""
    return f"{parsed:g} Mbps"


def _serving_network_from_supi(supi: str) -> Optional[Dict[str, str]]:
    digits = "".join(char for char in str(supi or "") if char.isdigit())
    if len(digits) < 5:
        return None
    return {"mcc": digits[:3], "mnc": digits[3:5]}


def _build_subs_sess_ambr(
    existing: Any,
    *,
    target_flow: Dict[str, Any],
    qos_payload: Dict[str, Any],
    live_session: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    ambr = dict(existing) if isinstance(existing, dict) else {}
    session_ambr = dict(live_session.get("authSessAmbr")) if isinstance(live_session, dict) and isinstance(live_session.get("authSessAmbr"), dict) else {}
    flow_sla = target_flow.get("sla") if isinstance(target_flow.get("sla"), dict) else {}
    flow_allocation = target_flow.get("allocation") if isinstance(target_flow.get("allocation"), dict) else {}

    downlink = _coerce_text(ambr.get("downlink") or session_ambr.get("downlink"))
    uplink = _coerce_text(ambr.get("uplink") or session_ambr.get("uplink"))

    if not downlink:
        downlink = _format_mbps(
            _first_positive_float(
                qos_payload.get("maxbrDl"),
                flow_allocation.get("allocated_bandwidth_dl"),
                flow_sla.get("bandwidth_dl"),
                qos_payload.get("gbrDl"),
                flow_sla.get("guaranteed_bandwidth_dl"),
            )
        )
    if not uplink:
        uplink = _format_mbps(
            _first_positive_float(
                qos_payload.get("maxbrUl"),
                flow_allocation.get("allocated_bandwidth_ul"),
                flow_sla.get("bandwidth_ul"),
                qos_payload.get("gbrUl"),
                flow_sla.get("guaranteed_bandwidth_ul"),
            )
        )

    if downlink:
        ambr["downlink"] = downlink
    if uplink:
        ambr["uplink"] = uplink
    return ambr or None


def _build_subs_def_qos(
    existing: Any,
    *,
    target_flow: Dict[str, Any],
    qos_payload: Dict[str, Any],
    live_qos: Optional[Dict[str, Any]],
    live_session: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    subs_def_qos = dict(existing) if isinstance(existing, dict) else {}
    session_qos = dict(live_session.get("authDefQos")) if isinstance(live_session, dict) and isinstance(live_session.get("authDefQos"), dict) else {}
    flow_sla = target_flow.get("sla") if isinstance(target_flow.get("sla"), dict) else {}

    five_qi = _to_int(subs_def_qos.get("5qi"))
    if five_qi is None:
        five_qi = _to_int(subs_def_qos.get("var5qi"))
    if five_qi is None:
        five_qi = _to_int(qos_payload.get("5qi", qos_payload.get("var5qi")))
    if five_qi is None and isinstance(live_qos, dict):
        five_qi = _to_int(live_qos.get("5qi", live_qos.get("var5qi")))
    if five_qi is None:
        five_qi = _to_int(session_qos.get("5qi", session_qos.get("var5qi")))
    if five_qi is not None:
        subs_def_qos["5qi"] = five_qi

    priority = _to_int(subs_def_qos.get("priorityLevel"))
    if priority is None:
        priority = _to_int(qos_payload.get("priorityLevel"))
    if priority is None and isinstance(live_qos, dict):
        priority = _to_int(live_qos.get("priorityLevel"))
    if priority is None:
        priority = _to_int(session_qos.get("priorityLevel"))
    if priority is None:
        priority = _to_int(flow_sla.get("priority"))
    if priority is not None:
        subs_def_qos["priorityLevel"] = priority

    arp_source: Any = subs_def_qos.get("arp")
    if not isinstance(arp_source, dict) and isinstance(live_qos, dict) and isinstance(live_qos.get("arp"), dict):
        arp_source = live_qos.get("arp")
    if not isinstance(arp_source, dict) and isinstance(session_qos.get("arp"), dict):
        arp_source = session_qos.get("arp")
    if isinstance(arp_source, dict) or priority is not None:
        arp = dict(arp_source) if isinstance(arp_source, dict) else {}
        if _to_int(arp.get("priorityLevel")) is None and priority is not None:
            arp["priorityLevel"] = priority
        subs_def_qos["arp"] = arp

    return subs_def_qos or None


def _five_tuple_to_sequence(five_tuple: Any) -> Optional[tuple[Any, Any, Any, Any, Any]]:
    if isinstance(five_tuple, (list, tuple)) and len(five_tuple) == 5:
        return tuple(five_tuple)
    if not isinstance(five_tuple, dict):
        return None
    keys = ("source_ip", "destination_ip", "source_port", "destination_port", "protocol")
    if not all(key in five_tuple for key in keys):
        return None
    return tuple(five_tuple[key] for key in keys)


def _extract_pdu_session_id(sm_policy_key: str, supi: str) -> Optional[int]:
    text = str(sm_policy_key or "").strip()
    prefix = f"{str(supi or '').strip()}-"
    if text.startswith(prefix):
        suffix = text[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    if "-" in text:
        suffix = text.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _find_sm_policy_key(ue_context: Dict[str, Any], flow_id: str) -> str:
    target = str(flow_id or "").strip()
    if not target:
        return ""

    for mapping_name, id_field in (("pccRules", "pccRuleId"), ("qosDecs", "qosId"), ("sessRules", "sessRuleId")):
        top_level = ue_context.get(mapping_name)
        if not isinstance(top_level, dict):
            continue
        for sm_policy_key, policy_map in top_level.items():
            if not isinstance(policy_map, dict):
                continue
            for rule_key, rule_payload in policy_map.items():
                candidate = str(rule_key or "").strip()
                if isinstance(rule_payload, dict) and not candidate:
                    candidate = str(rule_payload.get(id_field) or "").strip()
                if candidate.endswith(target):
                    return str(sm_policy_key)

    sm_policy_data = ue_context.get("smPolicyData")
    if isinstance(sm_policy_data, dict) and len(sm_policy_data) == 1:
        return str(next(iter(sm_policy_data)))
    return ""


def _find_target_flow(flow_catalog: Any, flow_id: str, app_id: str = "") -> Optional[Dict[str, Any]]:
    target_flow_id = str(flow_id or "").strip()
    target_app_id = str(app_id or "").strip()
    if not isinstance(flow_catalog, list):
        return None
    for flow in flow_catalog:
        if not isinstance(flow, dict):
            continue
        candidate_flow_id = str(flow.get("flow_id") or flow.get("id") or "").strip()
        candidate_app_id = str(flow.get("app_id") or "").strip()
        if candidate_flow_id != target_flow_id:
            continue
        if target_app_id and candidate_app_id and candidate_app_id != target_app_id:
            continue
        return flow
    return None


def _select_live_rule(top_level: Any, sm_policy_key: str, flow_id: str, id_field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(top_level, dict):
        return None
    candidate_map = top_level.get(sm_policy_key) if sm_policy_key else None
    if not isinstance(candidate_map, dict):
        if len(top_level) == 1:
            candidate_map = next(iter(top_level.values()))
        else:
            candidate_map = None
    if not isinstance(candidate_map, dict):
        return None

    target = str(flow_id or "").strip()
    for rule_key, rule_payload in candidate_map.items():
        if not isinstance(rule_payload, dict):
            continue
        candidate = str(rule_key or rule_payload.get(id_field) or "").strip()
        if candidate.endswith(target):
            return dict(rule_payload)

    first_payload = next((item for item in candidate_map.values() if isinstance(item, dict)), None)
    return dict(first_payload) if isinstance(first_payload, dict) else None


def _enrich_sm_policy_details(
    policy_details: Dict[str, Any],
    *,
    policy_id: str,
    supi: str,
    flow_id: str,
    app_id: str,
) -> Dict[str, Any]:
    ue_context = get_ue_context_by_supi(supi) or {}
    if not isinstance(ue_context, dict):
        return policy_details

    flow_catalog = ue_context.get("flow_catalog") or []
    target_flow = _find_target_flow(flow_catalog, flow_id, app_id)
    if not isinstance(target_flow, dict):
        return policy_details

    sm_policy_key = _find_sm_policy_key(ue_context, flow_id)
    live_qos = _select_live_rule(ue_context.get("qosDecs"), sm_policy_key, flow_id, "qosId")
    live_pcc = _select_live_rule(ue_context.get("pccRules"), sm_policy_key, flow_id, "pccRuleId")
    live_session = _select_live_rule(ue_context.get("sessRules"), sm_policy_key, flow_id, "sessRuleId")

    qos_decs = policy_details.get("qosDecs")
    if isinstance(qos_decs, dict) and qos_decs:
        qos_key = next((key for key, value in qos_decs.items() if isinstance(value, dict)), "")
        if qos_key:
            qos_payload = dict(qos_decs[qos_key])
            live_5qi = None
            if isinstance(live_qos, dict):
                live_5qi = live_qos.get("5qi", live_qos.get("var5qi"))
            if qos_payload.get("5qi") in (None, "") and qos_payload.get("var5qi") in (None, "") and live_5qi not in (None, ""):
                qos_payload["5qi"] = live_5qi
            qos_decs[qos_key] = qos_payload

    pcc_rules = policy_details.get("pccRules")
    if isinstance(pcc_rules, dict) and pcc_rules:
        pcc_key = next((key for key, value in pcc_rules.items() if isinstance(value, dict)), "")
        if pcc_key:
            pcc_payload = dict(pcc_rules[pcc_key])
            flow_info = build_flow_info_from_five_tuple(
                _five_tuple_to_sequence(((target_flow.get("traffic") or {}).get("five_tuple")))
            )
            if flow_info:
                pcc_payload["flowInfos"] = [flow_info]
            elif isinstance(live_pcc, dict) and isinstance(live_pcc.get("flowInfos"), list) and live_pcc["flowInfos"]:
                pcc_payload["flowInfos"] = live_pcc["flowInfos"]
            pcc_rules[pcc_key] = pcc_payload

    existing_upstream = (
        dict(policy_details.get("upstreamSmPolicyContextData"))
        if isinstance(policy_details.get("upstreamSmPolicyContextData"), dict)
        else {}
    )
    flow_allocation = target_flow.get("allocation") or {}
    flow_service = target_flow.get("service") or {}
    access_context = ue_context.get("accessMobilityContext") or {}
    slice_info = (
        _snssai_from_code(existing_upstream.get("sliceInfo"))
        or _snssai_from_code(flow_allocation.get("current_slice_snssai"))
        or next((item for item in access_context.get("targetSnssais", []) if isinstance(item, dict)), None)
        or next((item for item in access_context.get("allowedSnssais", []) if isinstance(item, dict)), None)
    )

    if existing_upstream or slice_info:
        upstream: Dict[str, Any] = dict(existing_upstream)
        normalized_supi = _coerce_text(upstream.get("supi") or supi)
        if normalized_supi:
            upstream["supi"] = normalized_supi

        dnn = _coerce_text(upstream.get("dnn") or target_flow.get("dnn") or flow_service.get("dnn") or "internet")
        if dnn:
            upstream["dnn"] = dnn

        pdu_session_id = _to_int(upstream.get("pduSessionId"))
        if pdu_session_id is None or pdu_session_id <= 0:
            pdu_session_id = _extract_pdu_session_id(sm_policy_key, normalized_supi) or 1
        upstream["pduSessionId"] = pdu_session_id

        if not _coerce_text(upstream.get("pduSessionType")):
            upstream["pduSessionType"] = "IPV4"
        if not _coerce_text(upstream.get("notificationUri")):
            upstream["notificationUri"] = f"http://127.0.0.1/callbacks/sm/{policy_id}"
        if not _coerce_text(upstream.get("accessType")):
            upstream["accessType"] = _coerce_text(access_context.get("accessType")) or "3GPP_ACCESS"
        if not _coerce_text(upstream.get("ratType")):
            upstream["ratType"] = _coerce_text(access_context.get("ratType")) or "NR"
        if not _coerce_text(upstream.get("ueTimeZone")):
            upstream["ueTimeZone"] = _coerce_text(access_context.get("timeZone")) or "+08:00"

        if slice_info:
            upstream["sliceInfo"] = slice_info

        serving_network = dict(upstream.get("servingNetwork")) if isinstance(upstream.get("servingNetwork"), dict) else {}
        derived_serving_network = None
        if isinstance(access_context.get("servingPlmn"), dict):
            derived_serving_network = access_context.get("servingPlmn")
        elif normalized_supi:
            derived_serving_network = _serving_network_from_supi(normalized_supi)
        if isinstance(derived_serving_network, dict):
            if derived_serving_network.get("mcc") not in (None, ""):
                serving_network.setdefault("mcc", str(derived_serving_network["mcc"]))
            if derived_serving_network.get("mnc") not in (None, ""):
                serving_network.setdefault("mnc", str(derived_serving_network["mnc"]))
        if _coerce_text(serving_network.get("mcc")) and _coerce_text(serving_network.get("mnc")):
            upstream["servingNetwork"] = serving_network

        if not isinstance(upstream.get("userLocationInfo"), dict) and isinstance(access_context.get("userLoc"), dict):
            upstream["userLocationInfo"] = access_context["userLoc"]

        subs_sess_ambr = _build_subs_sess_ambr(
            upstream.get("subsSessAmbr"),
            target_flow=target_flow,
            qos_payload=qos_payload,
            live_session=live_session,
        )
        if subs_sess_ambr is not None:
            upstream["subsSessAmbr"] = subs_sess_ambr

        subs_def_qos = _build_subs_def_qos(
            upstream.get("subsDefQos"),
            target_flow=target_flow,
            qos_payload=qos_payload,
            live_qos=live_qos,
            live_session=live_session,
        )
        if subs_def_qos is not None:
            upstream["subsDefQos"] = subs_def_qos

        policy_details["upstreamSmPolicyContextData"] = upstream

    return policy_details


def build_dispatch_envelope(
    policy_type: str,
    policy_json: Any,
    *,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    snapshot_id: Optional[str] = None,
) -> Dict[str, Any]:
    parsed_payload = _parse_policy_details(policy_json)
    if isinstance(parsed_payload.get("policy_details"), dict):
        parsed_payload = dict(parsed_payload)
        policy_details = dict(parsed_payload["policy_details"])
        parsed_payload["policy_details"] = policy_details
    else:
        policy_details = dict(parsed_payload)
        parsed_payload = policy_details
    normalized_policy_type = _coerce_identifier(policy_type, "policy_type")
    policy_id = _coerce_identifier(parsed_payload.get("policy_id") or policy_details.get("policy_id"), "policy_id")
    if normalized_policy_type == "SmPolicyDecision":
        supi = str(parsed_payload.get("supi") or policy_details.get("supi") or "").strip()
        flow_id = str(parsed_payload.get("flow_id") or policy_details.get("flow_id") or "").strip()
        app_id = str(parsed_payload.get("app_id") or policy_details.get("app_id") or "").strip()
        if supi and flow_id:
            policy_details = _enrich_sm_policy_details(
                policy_details,
                policy_id=policy_id,
                supi=supi,
                flow_id=flow_id,
                app_id=app_id,
            )
            parsed_payload["policy_details"] = policy_details

    envelope: Dict[str, Any] = {
        "request_id": str(request_id or f"req-{uuid.uuid4()}"),
        "session_id": str(session_id or "").strip(),
        "snapshot_id": str(snapshot_id or "").strip(),
        "policy_id": policy_id,
        "policy_type": normalized_policy_type,
        "policy_details": policy_details,
    }

    flow_id = str(policy_details.get("flow_id") or "").strip()
    if flow_id:
        envelope["flow_id"] = flow_id

    target_type = str(policy_details.get("target_type") or "").strip()
    if target_type:
        envelope["target_type"] = target_type

    return envelope


def dispatch_policy_to_pcf_request(
    policy_type: str,
    policy_json: Any,
    *,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    snapshot_id: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        payload = build_dispatch_envelope(
            policy_type,
            policy_json,
            request_id=request_id,
            session_id=session_id,
            snapshot_id=snapshot_id,
        )
    except ValueError as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "request_id": str(request_id or ""),
            "session_id": str(session_id or ""),
            "snapshot_id": str(snapshot_id or ""),
            "policy_id": "",
            "policy_type": str(policy_type or "").strip(),
        }

    if not str(PCF_BASE_URL or "").strip():
        return {
            "status": "failed",
            "error": "PCF address not configured",
            **payload,
        }

    try:
        response = requests.post(
            f"{PCF_BASE_URL}{POLICY_EXECUTION_PATH}",
            json=payload,
        )
    except requests.exceptions.RequestException as exc:
        return {
            "status": "failed",
            "error": f"PCF request failed: {exc}",
            **payload,
        }

    try:
        response_payload = response.json()
    except ValueError:
        response_payload = {"raw_response": response.text}

    result: Dict[str, Any] = {"status": "success" if response.ok else "failed", "response_code": response.status_code, **payload}
    if response.ok:
        result.update(response_payload if isinstance(response_payload, dict) else {"response": response_payload})
    else:
        result["error"] = (
            response_payload.get("error")
            if isinstance(response_payload, dict) and response_payload.get("error")
            else response.text
        )
    return result


def dispatch_policy_to_pcf(policy_type: str, policy_json: str) -> str:
    """
    Dispatch a policy payload to the policy execution gateway.

    Returns a JSON string so callers can parse the final execution result deterministically.
    """
    result = dispatch_policy_to_pcf_request(policy_type, policy_json)
    return json.dumps(result, ensure_ascii=False)


def get_network_feedback(policy_id: str) -> str:
    """
    Query feedback for a policy from the monitoring side.
    """
    normalized_policy_id = str(policy_id or "").strip()
    if not normalized_policy_id:
        return json.dumps({"status": "failed", "error": "policy_id is required"}, ensure_ascii=False)

    try:
        response = requests.get(
            f"{PCF_BASE_URL}{POLICY_EXECUTION_PATH}/{normalized_policy_id}",
            timeout=PCF_FEEDBACK_REQUEST_TIMEOUT_SEC,
        )
    except requests.exceptions.RequestException as exc:
        return json.dumps(
            {"status": "failed", "policy_id": normalized_policy_id, "error": f"monitor request failed: {exc}"},
            ensure_ascii=False,
        )

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_response": response.text}

    result = {
        "status": "success" if response.ok else "failed",
        "policy_id": normalized_policy_id,
        "response_code": response.status_code,
    }
    if response.ok:
        if isinstance(payload, dict):
            result.update(payload)
        else:
            result["response"] = payload
    else:
        result["error"] = payload.get("error") if isinstance(payload, dict) else response.text
    return json.dumps(result, ensure_ascii=False)


@tool_with_reason
def get_ue_context(
    supi: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Query UE context details by SUPI.
    """
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return "UE Context Query Failed: supi is required"

    try:
        db_ctx = get_ue_context_by_supi(normalized_supi)
    except Exception as exc:
        logger.error(f"Failed to read UE context for {normalized_supi}: {exc}")
        return f"UE Context Query Failed: {exc}"

    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "

    if db_ctx:
        return f"{prefix}UE Context Retrieved From DB:\n{json.dumps(db_ctx, ensure_ascii=False, indent=2)}"
        # trimmed = _trim_ue_context_for_agent(db_ctx)
        # return f"{prefix}UE Context Retrieved From DB:\n{json.dumps(trimmed, ensure_ascii=False, indent=2)}"
    return f"UE Context Not Found for SUPI: {normalized_supi}"


@tool_with_reason
def get_sm_ue_context(
    supi: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Query SM-domain UE policy context by SUPI.
    Use this when QoS / SM intent needs current PCC, QoS or session-policy evidence.
    """
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return "SM UE Context Query Failed: supi is required"

    try:
        db_ctx = get_ue_context_by_supi(normalized_supi)
    except Exception as exc:
        logger.error(f"Failed to read SM UE context for {normalized_supi}: {exc}")
        return f"SM UE Context Query Failed: {exc}"

    if not db_ctx:
        return f"SM UE Context Not Found for SUPI: {normalized_supi}"

    trimmed = _trim_sm_ue_context_for_agent(db_ctx)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}SM UE Context Retrieved:\n{json.dumps(trimmed, ensure_ascii=False, indent=2)}"


@tool_with_reason
def get_ue_flow_catalog(
    supi: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Return the app/flow catalog of a UE from the latest scenario snapshot.
    """
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return "UE Flow Catalog Query Failed: supi is required"

    catalog = get_ue_flow_catalog_by_supi(normalized_supi)
    result = json.dumps(catalog, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}UE Flow Catalog Retrieved:\n {result}"


@tool_with_reason
def get_sm_ue_flow_catalog(
    supi: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Return the SM-domain app/flow catalog of a UE from the latest scenario snapshot.
    Use this when QoS / SM intent has a SUPI and needs app/flow grounding.
    """
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return "SM UE Flow Catalog Query Failed: supi is required"

    catalog = get_ue_flow_catalog_by_supi(normalized_supi)
    result = json.dumps(catalog, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}SM UE Flow Catalog Retrieved:\n {result}"


@tool_with_reason
def search_flow_targets_by_name(
    app_name: str = "",
    flow_name: str = "",
    limit: int = 5,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Semantically search the latest snapshot for flow targets by app_name and/or flow_name.
    Use this when the user names an app or flow but does not provide a SUPI.
    """
    normalized_app_name = str(app_name or "").strip()
    normalized_flow_name = str(flow_name or "").strip()
    if not normalized_app_name and not normalized_flow_name:
        return "Semantic Flow Target Search Failed: app_name or flow_name is required"

    payload = search_flow_targets_by_semantic(
        app_name=normalized_app_name,
        flow_name=normalized_flow_name,
        limit=limit,
    )
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}Semantic Flow Target Search Retrieved:\n {result}"


@tool_with_reason
def search_sm_flow_targets(
    app_name: str = "",
    flow_name: str = "",
    limit: int = 5,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Search SM-domain flow targets by app_name and/or flow_name.
    Use this when QoS / SM intent names an app or flow but lacks a unique catalog target.
    """
    normalized_app_name = str(app_name or "").strip()
    normalized_flow_name = str(flow_name or "").strip()
    if not normalized_app_name and not normalized_flow_name:
        return "SM Flow Target Search Failed: app_name or flow_name is required"

    payload = search_flow_targets_by_semantic(
        app_name=normalized_app_name,
        flow_name=normalized_flow_name,
        limit=limit,
    )
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}SM Flow Target Search Retrieved:\n {result}"


@tool_with_reason
def get_am_policy_context(
    supi: str = "",
    association_id: str = "",
    include_associations: bool = True,
    include_access_context: bool = True,
    include_mobility_summary: bool = True,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Query AM-domain UE context by SUPI.
    Use this when mobility / AM intent needs current AM policy, access-mobility state or association evidence.
    """
    normalized_supi = str(supi or "").strip()
    normalized_association_id = str(association_id or "").strip()
    if not normalized_supi:
        return "AM Policy Context Query Failed: supi is required"

    try:
        db_ctx = get_ue_context_by_supi(normalized_supi)
    except Exception as exc:
        logger.error(f"Failed to read AM policy context for {normalized_supi}: {exc}")
        return f"AM Policy Context Query Failed: {exc}"

    if not db_ctx:
        return f"AM Policy Context Not Found for SUPI: {normalized_supi}"

    trimmed = _trim_am_policy_context_for_agent(
        db_ctx,
        association_id=normalized_association_id,
        include_associations=bool(include_associations),
        include_access_context=bool(include_access_context),
        include_mobility_summary=bool(include_mobility_summary),
    )
    if include_associations:
        association_records = list_am_policy_associations_by_supi(normalized_supi)
        if normalized_association_id:
            association_records = [
                item for item in association_records if str(item.get("polAssoId") or "").strip() == normalized_association_id
            ]
        trimmed["associationRecords"] = association_records

    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}AM Policy Context Retrieved:\n{json.dumps(trimmed, ensure_ascii=False, indent=2)}"


@tool_with_reason
def search_am_policy_targets(
    supi: str = "",
    association_id: str = "",
    allowed_snssai: str = "",
    target_snssai: str = "",
    service_area: str = "",
    rfsp: str = "",
    access_type: str = "",
    limit: int = 5,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Search AM-domain policy targets by association, NSSAI, RFSP, service-area or access-type evidence.
    Use this when mobility / AM intent must ground to existing AM policy state rather than QoS flow names.
    """
    if not any(
        str(value or "").strip()
        for value in (supi, association_id, allowed_snssai, target_snssai, service_area, rfsp, access_type)
    ):
        return (
            "AM Policy Target Search Failed: at least one of supi, association_id, allowed_snssai, "
            "target_snssai, service_area, rfsp or access_type is required"
        )

    payload = search_am_policy_targets_by_context(
        supi=str(supi or "").strip(),
        association_id=str(association_id or "").strip(),
        allowed_snssai=str(allowed_snssai or "").strip(),
        target_snssai=str(target_snssai or "").strip(),
        service_area=str(service_area or "").strip(),
        rfsp=str(rfsp or "").strip(),
        access_type=str(access_type or "").strip(),
        limit=limit,
    )
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}AM Policy Target Search Retrieved:\n {result}"
