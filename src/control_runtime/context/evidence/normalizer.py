from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_app_id(app_id: Any) -> str:
    value = str(app_id or "").strip()
    if not value:
        return ""
    if value.startswith("app-"):
        return value
    return value


def build_slice_snssai(slice_code: str) -> Optional[Dict[str, Any]]:
    code = str(slice_code or "").strip()
    if len(code) < 8:
        return None
    try:
        sst = int(code[:2], 16)
    except ValueError:
        return None
    return {"sst": sst, "sd": code[2:8]}
