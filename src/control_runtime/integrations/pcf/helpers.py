from __future__ import annotations

import json
from typing import Any, Dict, Optional

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


