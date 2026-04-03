import json
import uuid
from typing import Any, Dict, Optional

import requests
from langchain.tools import ToolRuntime, tool

from agent_runtime import AgentRuntimeContext
from tools.db_tool import get_ue_context_by_supi, get_ue_flow_catalog_by_supi
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Mock PCF base address used by the local integration environment.
PCF_BASE_URL = "http://localhost:8000"
PCF_REQUEST_TIMEOUT_SEC = 5


def _trim_ue_context_for_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    trimmed = dict(payload)
    trimmed.pop("app_catalog", None)
    trimmed.pop("flow_catalog", None)
    return trimmed


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
    policy_details = _parse_policy_details(policy_json)
    normalized_policy_type = _coerce_identifier(policy_type, "policy_type")
    policy_id = _coerce_identifier(policy_details.get("policy_id"), "policy_id")

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
            f"{PCF_BASE_URL}/pcf/policies",
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

    result: Dict[str, Any] = {
        "status": "success" if response.ok else "failed",
        "response_code": response.status_code,
        **payload,
    }
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
    Dispatch a policy payload to PCF through HTTP POST.

    Returns a JSON string so callers can parse the ack deterministically.
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
        response = requests.get(f"{PCF_BASE_URL}/monitor/status/{normalized_policy_id}", timeout=PCF_REQUEST_TIMEOUT_SEC)
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


@tool
def get_ue_context(
    supi: str,
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


@tool
def get_ue_flow_catalog(
    supi: str,
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
