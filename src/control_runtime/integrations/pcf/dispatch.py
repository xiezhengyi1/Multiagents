from __future__ import annotations

import json
import re
from typing import Any

from ..testbed import TestbedClient, TestbedConfig

SM_POLICY_TYPE = "SmPolicyDecision"


def _mapping(policy_json: Any) -> dict[str, Any]:
    payload = json.loads(policy_json) if isinstance(policy_json, str) else policy_json
    if not isinstance(payload, dict):
        raise TypeError("policy payload must be a JSON object")
    details = payload.get("policy_details", payload)
    if not isinstance(details, dict):
        raise TypeError("policy_details must be a JSON object")
    return details


def _mbps(value: object) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(Kbps|Mbps|Gbps)\s*", str(value))
    if match is None:
        raise ValueError(f"unsupported bandwidth value: {value!r}")
    amount = float(match.group(1))
    return amount * {"Kbps": 0.001, "Mbps": 1.0, "Gbps": 1000.0}[match.group(2)]


def build_dispatch_envelope(
    policy_type: str,
    policy_json: Any,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    if policy_type != SM_POLICY_TYPE:
        raise ValueError("the live testbed supports SmPolicyDecision only")
    details = _mapping(policy_json)
    if not snapshot_id:
        raise ValueError("snapshot_id is required")
    return {
        "request_id": request_id or "",
        "session_id": session_id or "",
        "snapshot_id": snapshot_id,
        "policy_type": policy_type,
        "policy_details": details,
    }


def dispatch_policy_to_pcf_request(
    policy_type: str,
    policy_json: Any,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    snapshot_id: str | None = None,
    overall_deadline_sec: float | None = None,
) -> dict[str, Any]:
    if overall_deadline_sec is not None:
        raise ValueError("the live action gateway owns the action deadline")
    envelope = build_dispatch_envelope(
        policy_type,
        policy_json,
        request_id=request_id,
        session_id=session_id,
        snapshot_id=snapshot_id,
    )
    details = envelope["policy_details"]
    supi = str(details["supi"])
    flow_id = str(details["flow_id"])
    qos_decs = details["qosDecs"]
    if not isinstance(qos_decs, dict) or len(qos_decs) != 1:
        raise ValueError("one QoS decision is required")
    qos = next(iter(qos_decs.values()))
    client = TestbedClient.create(TestbedConfig.from_env())
    snapshot = client.snapshot(envelope["snapshot_id"])
    native_flow = next(
        flow
        for flow in snapshot["flows"]
        if flow["id"] == flow_id and flow["supi"] == supi
    )
    return client.dispatch_policy(
        {
            "run_id": client.config.run_id,
            "snapshot_id": envelope["snapshot_id"],
            "action_type": "sm_policy_update",
            "target": {"supi": supi, "flow_id": flow_id},
            "parameters": {
                "five_qi": native_flow["session"]["five_qi"],
                "gbr_ul_mbps": _mbps(qos["gbrUl"]),
                "mbr_ul_mbps": _mbps(qos["maxbrUl"]),
                "gbr_dl_mbps": _mbps(qos["gbrDl"]),
                "mbr_dl_mbps": _mbps(qos["maxbrDl"]),
            },
        }
    )


def dispatch_policy_to_pcf(policy_type: str, policy_json: str) -> str:
    details = _mapping(policy_json)
    result = dispatch_policy_to_pcf_request(
        policy_type,
        details,
        snapshot_id=str(details["snapshot_id"]),
    )
    return json.dumps(result, ensure_ascii=False)


__all__ = [
    "SM_POLICY_TYPE",
    "build_dispatch_envelope",
    "dispatch_policy_to_pcf",
    "dispatch_policy_to_pcf_request",
]
