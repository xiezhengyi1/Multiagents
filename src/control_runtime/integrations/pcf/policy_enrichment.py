from __future__ import annotations

from typing import Any, Dict, Optional

from ..storage import build_flow_info_from_five_tuple, get_ue_context_by_supi
from .helpers import (
    _coerce_text,
    _first_positive_float,
    _format_mbps,
    _serving_network_from_supi,
    _snssai_from_code,
    _to_float,
    _to_int,
)

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
    snapshot_id: str = "",
) -> Dict[str, Any]:
    ue_context = get_ue_context_by_supi(supi, snapshot_id=snapshot_id) or {}
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


