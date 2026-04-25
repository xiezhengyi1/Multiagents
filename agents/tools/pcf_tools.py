import json
import os
import uuid
from typing import Any, Dict, Optional

import requests
from langchain.tools import ToolRuntime, tool

from agents.tools.wrapper_think import tool_with_reason

from agent_runtime.core.context import AgentRuntimeContext
from agents.tools.db_tool import (
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
PCF_REQUEST_TIMEOUT_SEC = 5
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
        policy_details = parsed_payload["policy_details"]
    else:
        policy_details = parsed_payload
    normalized_policy_type = _coerce_identifier(policy_type, "policy_type")
    policy_id = _coerce_identifier(parsed_payload.get("policy_id") or policy_details.get("policy_id"), "policy_id")

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
            timeout=PCF_REQUEST_TIMEOUT_SEC,
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
            timeout=PCF_REQUEST_TIMEOUT_SEC,
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
