from __future__ import annotations

import json

from langchain.tools import ToolRuntime

from shared.logging import setup_logger
from shared.runtime import AgentRuntimeContext
from shared.tools import tool_with_reason
from ..storage import (
    get_ue_context_by_supi,
    get_ue_flow_catalog_by_supi,
    list_am_policy_associations_by_supi,
    search_am_policy_targets_by_context,
    search_flow_targets_by_semantic,
)
from .helpers import _trim_am_policy_context_for_agent, _trim_sm_ue_context_for_agent, _trim_ue_context_for_agent

logger = setup_logger(__name__)

def _normalize_snssai_key(value: object) -> str:
    if isinstance(value, dict):
        try:
            sst = int(value.get("sst"))
        except (TypeError, ValueError):
            return ""
        sd = str(value.get("sd") or "").strip().lower()
        if not 0 <= sst <= 255 or len(sd) != 6 or any(char not in "0123456789abcdef" for char in sd):
            return ""
        return f"{sst:02x}{sd}"
    text = str(value or "").strip().lower()
    if len(text) == 8 and all(char in "0123456789abcdef" for char in text):
        return text
    return ""


def _collect_snssai_keys(*raw_values: object) -> list[str]:
    keys: list[str] = []
    for raw_value in raw_values:
        values = raw_value if isinstance(raw_value, (list, tuple)) else []
        for value in values:
            key = _normalize_snssai_key(value)
            if key and key not in keys:
                keys.append(key)
    return keys


def _resolve_serving_plmn_id(
    *,
    requested_plmn_id: str,
    am_policy_context: dict,
    access_mobility_context: dict,
) -> str:
    if requested_plmn_id:
        return requested_plmn_id
    for context in (am_policy_context, access_mobility_context):
        plmn = context.get("servingPlmn")
        if not isinstance(plmn, dict):
            continue
        mcc = str(plmn.get("mcc") or "").strip()
        mnc = str(plmn.get("mnc") or "").strip()
        if mcc and mnc:
            return f"{mcc}{mnc}"
    return ""


def _normalize_ue_slice_subscription_from_db(
    *,
    supi: str,
    serving_plmn_id: str,
    ue_context: dict,
) -> dict:
    am_policy_context = (
        ue_context.get("amPolicyContext")
        if isinstance(ue_context.get("amPolicyContext"), dict)
        else {}
    )
    access_mobility_context = (
        ue_context.get("accessMobilityContext")
        if isinstance(ue_context.get("accessMobilityContext"), dict)
        else {}
    )
    associations = (
        am_policy_context.get("associations")
        if isinstance(am_policy_context.get("associations"), dict)
        else {}
    )
    association_allowed_snssais: list[object] = []
    for association in associations.values():
        if not isinstance(association, dict):
            continue
        request = association.get("request")
        if isinstance(request, dict):
            association_allowed_snssais.append(request.get("allowedSnssais"))

    authorized_snssais = _collect_snssai_keys(
        am_policy_context.get("allowedSnssais"),
        access_mobility_context.get("allowedSnssais"),
        *association_allowed_snssais,
    )
    default_snssais = _collect_snssai_keys(
        am_policy_context.get("defaultSingleNssais"),
        access_mobility_context.get("defaultSingleNssais"),
    )
    current_target_snssais = _collect_snssai_keys(
        am_policy_context.get("targetSnssais"),
        access_mobility_context.get("targetSnssais"),
    )
    return {
        "supi": supi,
        "authority": "postgresql_ue_context",
        "source_table": "ue_context",
        "serving_plmn_id": _resolve_serving_plmn_id(
            requested_plmn_id=serving_plmn_id,
            am_policy_context=am_policy_context,
            access_mobility_context=access_mobility_context,
        ),
        "records_found": True,
        "authorized_snssais": authorized_snssais,
        "default_snssais": default_snssais,
        "current_target_snssais": current_target_snssais,
        # The local UE context has no DNN entitlement column. Keep absent
        # facts empty rather than inventing a UDR-derived subscription.
        "dnn_availability": {},
    }


def _project_sm_flow_catalog_for_agent(catalog: object) -> dict:
    """Keep identity, policy binding, and runtime evidence required by IEA."""
    if not isinstance(catalog, dict):
        return {"supi": "", "app_catalog": [], "flow_catalog": []}
    apps = [
        {
            key: item.get(key)
            for key in ("supi", "app_id", "app_name", "flow_count")
            if item.get(key) not in (None, "", [], {})
        }
        for item in (catalog.get("app_catalog") or [])
        if isinstance(item, dict)
    ]
    flows = []
    for item in catalog.get("flow_catalog") or []:
        if not isinstance(item, dict):
            continue
        traffic = item.get("traffic") if isinstance(item.get("traffic"), dict) else {}
        projected = {
            key: item.get(key)
            for key in ("supi", "app_id", "app_name", "flow_id", "flow_name")
            if item.get(key) not in (None, "", [], {})
        }
        nested_fields = {
            "service": ("service_type", "service_type_id"),
            "sla": None,
            "allocation": None,
            "telemetry": (
                "throughput_ul",
                "throughput_dl",
                "latency",
                "jitter",
                "loss_rate",
                "packet_sent",
                "packet_received",
                "tcp_retransmits",
                "tcp_retransmits_per_second",
                "tcp_rtt_ms_average",
                "tcp_rtt_ms_maximum",
                "availability",
                "observed_at",
            ),
            "ran": (
                "serving_cell_id",
                "gnb_id",
                "rnti",
                "qfi",
                "drb_id",
                "mcs_ul",
                "mcs_dl",
                "bler_ul",
                "bler_dl",
                "prb_ul",
                "prb_dl",
                "rsrp_dbm",
                "sinr_db",
                "handover_state",
            ),
            "session": (
                "pdu_session_id",
                "dnn",
                "snssai",
                "qfi",
                "five_qi",
                "upf_id",
                "ue_ip",
            ),
            "path": (
                "site_id",
                "service_instance_id",
                "upf_path",
                "app_endpoint",
            ),
            "data_quality": (
                "window_start",
                "window_end",
                "sample_count",
                "stale",
                "correlation_status",
            ),
        }
        for nested_key, allowed_fields in nested_fields.items():
            nested = item.get(nested_key)
            if isinstance(nested, dict):
                projected[nested_key] = {
                    key: value
                    for key, value in nested.items()
                    if allowed_fields is None or key in allowed_fields
                    if value not in (None, "", [], {})
                }
        if traffic.get("five_tuple"):
            projected["traffic"] = {"five_tuple": traffic.get("five_tuple")}
        flows.append(projected)
    return {
        "supi": str(catalog.get("supi") or "").strip(),
        "app_catalog": apps,
        "flow_catalog": flows,
    }


def _tool_snapshot_id(runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    snapshot_id = str(runtime.context.snapshot_id or "").strip() if runtime is not None and runtime.context is not None else ""
    if not snapshot_id:
        raise ValueError("PCF read tools require a bound snapshot_id")
    return snapshot_id


@tool_with_reason(side_effect="read", result_format="json")
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
        db_ctx = get_ue_context_by_supi(
            normalized_supi,
            snapshot_id=_tool_snapshot_id(runtime),
        )
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


@tool_with_reason(side_effect="read", result_format="json")
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
        db_ctx = get_ue_context_by_supi(
            normalized_supi,
            snapshot_id=_tool_snapshot_id(runtime),
        )
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


@tool_with_reason(side_effect="read", result_format="json")
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

    catalog = get_ue_flow_catalog_by_supi(
        normalized_supi,
        snapshot_id=_tool_snapshot_id(runtime),
    )
    result = json.dumps(catalog, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}UE Flow Catalog Retrieved:\n {result}"


@tool_with_reason(side_effect="read", result_format="json")
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

    catalog = get_ue_flow_catalog_by_supi(
        normalized_supi,
        snapshot_id=_tool_snapshot_id(runtime),
    )
    projected_catalog = _project_sm_flow_catalog_for_agent(catalog)
    result = json.dumps(projected_catalog, ensure_ascii=False, separators=(",", ":"))
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}SM UE Flow Catalog Retrieved:\n {result}"


@tool_with_reason(side_effect="read", result_format="json")
def get_ue_slice_subscription(
    supi: str = "",
    serving_plmn_id: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Read authoritative UE slice entitlement from the PostgreSQL ue_context table.
    Use this before treating a requested S-NSSAI change as executable.
    This tool is read-only and does not create or modify subscription data.
    """
    normalized_supi = str(supi or "").strip()
    normalized_plmn = str(serving_plmn_id or "").strip()
    if not normalized_supi:
        return "UE Slice Subscription Query Failed: supi is required"
    ue_context = get_ue_context_by_supi(normalized_supi)
    if not ue_context:
        return "UE Slice Subscription Query Failed: ue_context record not found"
    subscription = _normalize_ue_slice_subscription_from_db(
        supi=normalized_supi,
        serving_plmn_id=normalized_plmn,
        ue_context=ue_context,
    )

    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}UE Slice Subscription Retrieved:\n{json.dumps(subscription, ensure_ascii=False, indent=2)}"


@tool_with_reason(side_effect="read", result_format="json")
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
        snapshot_id=_tool_snapshot_id(runtime),
        limit=limit,
    )
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}Semantic Flow Target Search Retrieved:\n {result}"


@tool_with_reason(side_effect="read", result_format="json")
def search_sm_flow_targets(
    supi: str = "",
    app_name: str = "",
    flow_name: str = "",
    limit: int = 5,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Search SM-domain flow targets by app_name and/or flow_name.
    Use this when QoS / SM intent names an app or flow but lacks a unique catalog target.
    """
    normalized_supi = str(supi or "").strip()
    normalized_app_name = str(app_name or "").strip()
    normalized_flow_name = str(flow_name or "").strip()
    if not normalized_supi and not normalized_app_name and not normalized_flow_name:
        return "SM Flow Target Search Failed: supi, app_name or flow_name is required"

    payload = search_flow_targets_by_semantic(
        supi=normalized_supi,
        app_name=normalized_app_name,
        flow_name=normalized_flow_name,
        snapshot_id=_tool_snapshot_id(runtime),
        limit=limit,
    )
    if normalized_supi and isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        filtered_candidates = [
            item
            for item in payload.get("candidates") or []
            if isinstance(item, dict) and str(item.get("supi") or "").strip() == normalized_supi
        ]
        payload = dict(payload)
        payload["candidates"] = filtered_candidates
        payload["candidate_count"] = len(filtered_candidates)
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}SM Flow Target Search Retrieved:\n {result}"


@tool_with_reason(side_effect="read", result_format="json")
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
        db_ctx = get_ue_context_by_supi(
            normalized_supi,
            snapshot_id=_tool_snapshot_id(runtime),
        )
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


@tool_with_reason(side_effect="read", result_format="json")
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

    _tool_snapshot_id(runtime)
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
