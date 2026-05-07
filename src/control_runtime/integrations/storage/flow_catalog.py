from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

def build_flow_description_from_five_tuple(five_tuple: Any) -> Optional[str]:
    """Encode a flow five-tuple into the flowDescription string supported by PCC flowInfos."""
    if not isinstance(five_tuple, (list, tuple)) or len(five_tuple) != 5:
        return None

    src_ip, dst_ip, src_port, dst_port, protocol = five_tuple
    src_ip = str(src_ip or "").strip()
    dst_ip = str(dst_ip or "").strip()
    protocol = str(protocol or "ip").strip().lower() or "ip"

    if not src_ip or not dst_ip:
        return None

    try:
        src_port = int(src_port)
        dst_port = int(dst_port)
    except (TypeError, ValueError):
        return None

    return f"permit out {protocol} from {src_ip} {src_port} to {dst_ip} {dst_port}"


def build_flow_info_from_five_tuple(five_tuple: Any, *, flow_direction: str = "BIDIRECTIONAL") -> Optional[Dict[str, Any]]:
    """Build a FlowInformation-compatible dict from a five tuple."""
    flow_description = build_flow_description_from_five_tuple(five_tuple)
    if not flow_description:
        return None
    return {
        "flowDescription": flow_description,
        "flowDirection": flow_direction,
    }


def _normalize_catalog_flow(app: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, Any]:
    service = flow.get("service") if isinstance(flow.get("service"), dict) else {}
    sla = flow.get("sla") if isinstance(flow.get("sla"), dict) else {}
    allocation = flow.get("allocation") if isinstance(flow.get("allocation"), dict) else {}
    traffic = flow.get("traffic") if isinstance(flow.get("traffic"), dict) else {}

    return {
        "supi": app.get("supi"),
        "app_name": app.get("name"),
        "app_id": app.get("id"),
        "flow_name": flow.get("name"),
        "flow_id": flow.get("id"),
        "dnn": flow.get("dnn") or service.get("dnn"),
        "service": {
            "service_type": service.get("service_type"),
            "service_type_id": service.get("service_type_id"),
            "dnn": service.get("dnn") or flow.get("dnn"),
        },
        "sla": {
            "bandwidth_ul": sla.get("bandwidth_ul"),
            "bandwidth_dl": sla.get("bandwidth_dl"),
            "guaranteed_bandwidth_ul": sla.get("guaranteed_bandwidth_ul"),
            "guaranteed_bandwidth_dl": sla.get("guaranteed_bandwidth_dl"),
            "latency": sla.get("latency"),
            "jitter": sla.get("jitter"),
            "loss_rate": sla.get("loss_rate"),
            "priority": sla.get("priority"),
        },
        "allocation": {
            "current_slice_snssai": allocation.get("current_slice_snssai"),
            "allocated_bandwidth_ul": allocation.get("allocated_bandwidth_ul"),
            "allocated_bandwidth_dl": allocation.get("allocated_bandwidth_dl"),
        },
        "traffic": {
            "packet_size": traffic.get("packet_size"),
            "arrival_rate": traffic.get("arrival_rate"),
            "five_tuple": list(traffic.get("five_tuple")) if isinstance(traffic.get("five_tuple"), (list, tuple)) else None,
        },
    }


def _build_catalogs_from_app_data(app_data: Any, supi: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    app_catalog: List[Dict[str, Any]] = []
    flow_catalog: List[Dict[str, Any]] = []
    if not isinstance(app_data, list) or not supi:
        return app_catalog, flow_catalog

    target_supi = str(supi).strip()
    for app in app_data:
        if not isinstance(app, dict):
            continue
        app_supi = str(app.get("supi") or "").strip()
        if app_supi != target_supi:
            continue

        app_entry = {
            "supi": target_supi,
            "app_name": app.get("name"),
            "app_id": app.get("id"),
            "flow_count": len(app.get("flows") or []),
        }
        app_catalog.append(app_entry)

        flows = app.get("flows") or []
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_catalog.append(_normalize_catalog_flow(app, flow))

    return app_catalog, flow_catalog



def _extract_flow_id_from_pcc_rule_id(rule_id: Any) -> Optional[str]:
    text = str(rule_id or "").strip()
    if not text:
        return None
    if text.startswith("pcc-") and len(text) > 4:
        return text[4:]
    if text.startswith("flow-"):
        return text
    return None


def _enrich_pcc_rules_with_flow_catalog(
    pcc_rules: Optional[Dict[str, Any]],
    flow_catalog: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(pcc_rules, dict) or not pcc_rules:
        return pcc_rules

    flow_map = {
        str(flow.get("flow_id")): flow
        for flow in flow_catalog
        if isinstance(flow, dict) and flow.get("flow_id")
    }

    enriched_top: Dict[str, Any] = {}
    for sm_policy_id, rule_map in pcc_rules.items():
        if not isinstance(rule_map, dict):
            enriched_top[sm_policy_id] = rule_map
            continue

        enriched_rule_map: Dict[str, Any] = {}
        for rule_key, rule_obj in rule_map.items():
            if not isinstance(rule_obj, dict):
                enriched_rule_map[rule_key] = rule_obj
                continue

            enriched_rule = dict(rule_obj)
            flow_id = _extract_flow_id_from_pcc_rule_id(enriched_rule.get("pccRuleId") or rule_key)
            flow_entry = flow_map.get(flow_id) if flow_id else None
            if flow_entry:
                traffic = flow_entry.get("traffic") if isinstance(flow_entry.get("traffic"), dict) else {}
                flow_info = build_flow_info_from_five_tuple(traffic.get("five_tuple"))
                if flow_info:
                    enriched_rule["flowInfos"] = [flow_info]
            enriched_rule_map[rule_key] = enriched_rule

        enriched_top[sm_policy_id] = enriched_rule_map

    return enriched_top


