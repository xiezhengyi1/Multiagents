from __future__ import annotations

import json
import os

from langchain.tools import ToolRuntime
import requests

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

UDR_BASE_URL = str(os.getenv("UDR_BASE_URL", "http://10.100.200.4:8000")).rstrip("/")
UDR_SERVING_PLMN_ID = str(os.getenv("UDR_SERVING_PLMN_ID", "20893")).strip()
try:
    UDR_REQUEST_TIMEOUT_SEC = max(0.1, float(os.getenv("UDR_REQUEST_TIMEOUT_SEC", "5")))
except (TypeError, ValueError):
    UDR_REQUEST_TIMEOUT_SEC = 5.0


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


def _read_udr_json(path: str) -> tuple[dict, bool]:
    response = requests.get(f"{UDR_BASE_URL}{path}", timeout=UDR_REQUEST_TIMEOUT_SEC)
    if response.status_code == 404:
        return {}, False
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"UDR returned a non-object payload for {path}")
    return dict(payload), True


def _normalize_ue_slice_subscription(
    *,
    supi: str,
    serving_plmn_id: str,
    am_data: dict,
    smf_selection_data: dict,
    am_data_found: bool,
    smf_selection_found: bool,
) -> dict:
    nssai = am_data.get("nssai") if isinstance(am_data.get("nssai"), dict) else {}
    subscribed_infos = (
        smf_selection_data.get("subscribedSnssaiInfos")
        if isinstance(smf_selection_data.get("subscribedSnssaiInfos"), dict)
        else {}
    )
    default_snssais = [
        key
        for key in (_normalize_snssai_key(item) for item in (nssai.get("defaultSingleNssais") or []))
        if key
    ]
    am_single_snssais = [
        key
        for key in (_normalize_snssai_key(item) for item in (nssai.get("singleNssais") or []))
        if key
    ]
    subscribed_snssai_infos: dict[str, dict] = {}
    for raw_key, value in subscribed_infos.items():
        key = _normalize_snssai_key(raw_key)
        if key and isinstance(value, dict):
            subscribed_snssai_infos[key] = dict(value)
    authorized_snssais = list(
        dict.fromkeys([*default_snssais, *am_single_snssais, *subscribed_snssai_infos.keys()])
    )
    dnn_availability = {
        key: [
            str(item.get("dnn") or "").strip()
            for item in (details.get("dnnInfos") or [])
            if isinstance(item, dict) and str(item.get("dnn") or "").strip()
        ]
        for key, details in subscribed_snssai_infos.items()
    }
    return {
        "supi": supi,
        "authority": "free5gc_udr_subscription_data",
        "serving_plmn_id": serving_plmn_id,
        "records_found": bool(am_data_found or smf_selection_found),
        "authorized_snssais": authorized_snssais,
        "default_snssais": list(dict.fromkeys(default_snssais)),
        "subscribed_snssai_infos": subscribed_snssai_infos,
        "dnn_availability": dnn_availability,
    }


def _project_sm_flow_catalog_for_agent(catalog: object) -> dict:
    """Keep only identity, SLA, binding, and policy-selector fields for IEA."""
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


@tool_with_reason
def get_ue_slice_subscription(
    supi: str = "",
    serving_plmn_id: str = "",
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Read authoritative UE slice subscription entitlement from free5GC UDR.
    Use this before treating a requested S-NSSAI change as executable.
    This tool is read-only and does not create or modify subscription data.
    """
    normalized_supi = str(supi or "").strip()
    normalized_plmn = str(serving_plmn_id or UDR_SERVING_PLMN_ID).strip()
    if not normalized_supi:
        return "UE Slice Subscription Query Failed: supi is required"
    if not normalized_plmn:
        return "UE Slice Subscription Query Failed: serving_plmn_id is required"

    base_path = f"/nudr-dr/v2/subscription-data/{normalized_supi}/{normalized_plmn}/provisioned-data"
    try:
        am_data, am_data_found = _read_udr_json(f"{base_path}/am-data")
        smf_selection_data, smf_selection_found = _read_udr_json(
            f"{base_path}/smf-selection-subscription-data"
        )
        subscription = _normalize_ue_slice_subscription(
            supi=normalized_supi,
            serving_plmn_id=normalized_plmn,
            am_data=am_data,
            smf_selection_data=smf_selection_data,
            am_data_found=am_data_found,
            smf_selection_found=smf_selection_found,
        )
    except (requests.RequestException, ValueError) as exc:
        logger.error(f"Failed to read UE slice subscription for {normalized_supi}: {exc}")
        return f"UE Slice Subscription Query Failed: {exc}"

    prefix = ""
    if runtime is not None:
        ctx = runtime.context
        prefix = f"[agent={ctx.agent_name}][session={ctx.session_id}][snapshot={ctx.snapshot_id}] "
    return f"{prefix}UE Slice Subscription Retrieved:\n{json.dumps(subscription, ensure_ascii=False, indent=2)}"


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
        snapshot_id=_tool_snapshot_id(runtime),
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
