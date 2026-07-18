from __future__ import annotations

import json
import math
import os
import time
import uuid
from typing import Any, Dict, Optional

import requests

from .helpers import _coerce_identifier, _parse_policy_details
from .policy_enrichment import _enrich_sm_policy_details

def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return max(0.01, value) if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


PCF_BASE_URL = str(os.getenv('PCF_BASE_URL', 'http://localhost:18080')).rstrip('/')
POLICY_GATEWAY_INSTANCE_ID = str(os.getenv('POLICY_GATEWAY_INSTANCE_ID', '')).strip()
PCF_FEEDBACK_REQUEST_TIMEOUT_SEC = _positive_env_float('PCF_FEEDBACK_REQUEST_TIMEOUT_SEC', 5)
PCF_POLICY_POLL_INTERVAL_SEC = _positive_env_float('PCF_POLICY_POLL_INTERVAL_SEC', 0.5)
# ns-3 can legitimately report ``pending`` while applying a policy.  Keep the
# synchronous dispatcher alive long enough for the normal asynchronous
# transition, rather than turning a healthy transition into a control retry.
PCF_POLICY_OVERALL_DEADLINE_SEC = _positive_env_float('PCF_POLICY_OVERALL_DEADLINE_SEC', 120)
AM_POLICY_TYPE = 'PcfAmPolicyControlPolicyAssociation'
POLICY_EXECUTION_PATH = '/policy-executions'

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
                snapshot_id=str(snapshot_id or "").strip(),
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
    overall_deadline_sec: Optional[float] = None,
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
            timeout=PCF_FEEDBACK_REQUEST_TIMEOUT_SEC,
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
        "gateway_url": f"{PCF_BASE_URL}{POLICY_EXECUTION_PATH}",
        **payload,
    }
    expected_instance_id = POLICY_GATEWAY_INSTANCE_ID
    if expected_instance_id:
        actual_service = str(response.headers.get("X-Policy-Acceptor") or "").strip()
        actual_instance_id = str(response.headers.get("X-Policy-Acceptor-Instance") or "").strip()
        result["gateway_expected_instance_id"] = expected_instance_id
        result["gateway_service"] = actual_service
        result["gateway_instance_id"] = actual_instance_id
        if actual_service != "ns3-free5gc-policy-acceptor" or actual_instance_id != expected_instance_id:
            result.update(
                {
                    "status": "failed",
                    "phase": "gateway_identity",
                    "error": (
                        "policy gateway identity mismatch: "
                        f"expected service=ns3-free5gc-policy-acceptor instance={expected_instance_id}, "
                        f"got service={actual_service or '<missing>'} instance={actual_instance_id or '<missing>'}"
                    ),
                    "gateway_response": response_payload,
                }
            )
            return result
    if response.ok:
        result.update(response_payload if isinstance(response_payload, dict) else {"response": response_payload})
        if str(result.get('status') or '').strip().lower() == 'pending':
            return _poll_policy_execution(payload, result, overall_deadline_sec=overall_deadline_sec)
    else:
        result["error"] = (
            response_payload.get("error")
            if isinstance(response_payload, dict) and response_payload.get("error")
            else response.text.strip()
            or f"gateway returned HTTP {response.status_code} with an empty response body"
        )
        result["gateway_response"] = response_payload
    return result


def dispatch_policy_to_pcf(policy_type: str, policy_json: str) -> str:
    """
    Dispatch a policy payload to the policy execution gateway.

    Returns a JSON string so callers can parse the final execution result deterministically.
    """
    result = dispatch_policy_to_pcf_request(policy_type, policy_json)
    return json.dumps(result, ensure_ascii=False)


def _fetch_network_feedback(execution_id: str) -> Dict[str, Any]:
    normalized_execution_id = str(execution_id or '').strip()
    if not normalized_execution_id:
        return {'status': 'failed', 'error': 'operation_id is required'}

    try:
        response = requests.get(
            f"{PCF_BASE_URL}{POLICY_EXECUTION_PATH}/{normalized_execution_id}",
            timeout=PCF_FEEDBACK_REQUEST_TIMEOUT_SEC,
        )
    except requests.exceptions.RequestException as exc:
        return {
            'status': 'failed',
            'operation_id': normalized_execution_id,
            'phase': 'feedback',
            'error': f'monitor request failed: {exc}',
        }

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_response": response.text}

    result = {
        "status": "failed",
        "operation_id": normalized_execution_id,
        "response_code": response.status_code,
    }
    if response.ok:
        if isinstance(payload, dict):
            result.update(payload)
            if not str(payload.get("status") or "").strip():
                result.update(
                    {
                        "status": "failed",
                        "phase": "feedback_contract",
                        "error": "PDA feedback response is missing a terminal or pending status",
                    }
                )
        else:
            result.update(
                {
                    "phase": "feedback_contract",
                    "error": "PDA feedback response must be a JSON object with status",
                    "response": payload,
                }
            )
    else:
        result["error"] = payload.get("error") if isinstance(payload, dict) else response.text
    return result


def _poll_policy_execution(
    dispatch_payload: Dict[str, Any],
    initial_result: Dict[str, Any],
    *,
    overall_deadline_sec: Optional[float],
) -> Dict[str, Any]:
    operation_id = str(initial_result.get('operation_id') or '').strip()
    if not operation_id:
        return {
            **dispatch_payload,
            **initial_result,
            'status': 'failed',
            'phase': 'dispatch_contract',
            'error': 'pending policy execution response did not include operation_id',
        }

    try:
        requested_deadline = float(overall_deadline_sec) if overall_deadline_sec is not None else PCF_POLICY_OVERALL_DEADLINE_SEC
    except (TypeError, ValueError):
        requested_deadline = PCF_POLICY_OVERALL_DEADLINE_SEC
    deadline_seconds = max(0.01, requested_deadline) if math.isfinite(requested_deadline) else PCF_POLICY_OVERALL_DEADLINE_SEC
    deadline = time.monotonic() + deadline_seconds
    latest_result = dict(initial_result)
    while True:
        status = str(latest_result.get('status') or '').strip().lower()
        if status in {'success', 'applied', 'completed', 'succeeded', 'failed', 'error', 'cancelled', 'canceled', 'timeout', 'timed_out'}:
            return {**dispatch_payload, **latest_result}
        if time.monotonic() >= deadline:
            return {
                **dispatch_payload,
                **latest_result,
                'status': 'failed',
                'status_code': 504,
                'phase': 'pda_deadline',
                'operation_id': operation_id,
                'message': 'PDA deadline reached before a terminal policy-execution result.',
                'error': (
                    f'PDA execution did not reach a terminal status within {deadline_seconds:.1f}s '
                    f'(operation_id={operation_id})'
                ),
            }
        time.sleep(min(PCF_POLICY_POLL_INTERVAL_SEC, max(0.0, deadline - time.monotonic())))
        latest_result = _fetch_network_feedback(operation_id)


def get_network_feedback(execution_id: str) -> str:
    """Query feedback for a policy operation from the monitoring side."""
    return json.dumps(_fetch_network_feedback(execution_id), ensure_ascii=False)
