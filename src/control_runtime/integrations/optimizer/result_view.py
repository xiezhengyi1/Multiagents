"""JSON-safe views of joint-optimizer results shared by agent tool adapters."""

from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python


def serialize_optimizer_result(result: Any) -> dict[str, Any]:
    """Return the optimizer result as the canonical JSON mapping."""
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    elif isinstance(result, dict):
        payload = to_jsonable_python(result)
    elif hasattr(result, "__dict__"):
        payload = to_jsonable_python(vars(result))
    else:
        payload = to_jsonable_python(result)
    if not isinstance(payload, dict):
        raise TypeError("optimizer result payload requires a mapping payload")
    return payload


def summarize_optimizer_result(result: Any) -> dict[str, Any]:
    """Expose the stable, compact optimizer evidence used by agent prompts."""
    payload = serialize_optimizer_result(result)
    qos_plan = payload.get("qos_plan") or {}
    qos_meta = qos_plan.get("meta") if isinstance(qos_plan, dict) else {}
    summary: dict[str, Any] = {
        "status": _normalize_optimizer_status(payload),
        "objective_value": (
            payload.get("objective_value")
            if payload.get("objective_value") is not None
            else (qos_meta.get("objective_value") if isinstance(qos_meta, dict) else None)
        ),
        "objective_breakdown": (
            payload.get("objective_breakdown")
            if isinstance(payload.get("objective_breakdown"), dict)
            else (qos_meta.get("breakdown") if isinstance(qos_meta, dict) else {})
        ),
        "infeasible_reasons": payload.get("infeasible_reasons") or [],
    }
    if isinstance(qos_meta, dict):
        summary["qos_meta_status"] = qos_meta.get("status")
        if isinstance(qos_meta.get("infeasibility_details"), list) and qos_meta.get("infeasibility_details"):
            summary["qos_infeasibility_details"] = qos_meta.get("infeasibility_details")

    flow_assignments: list[dict[str, Any]] = []
    target_apps = qos_plan.get("target_apps") if isinstance(qos_plan, dict) else []
    if isinstance(target_apps, list):
        for app_payload in target_apps:
            if not isinstance(app_payload, dict):
                continue
            for flow in app_payload.get("flows") or []:
                if not isinstance(flow, dict):
                    continue
                flow_assignments.append(_summarize_qos_flow(flow))
    if not flow_assignments:
        target_app = qos_plan.get("target_app") if isinstance(qos_plan, dict) else {}
        if isinstance(target_app, dict) and target_app.get("flows"):
            flow_assignments = [
                _summarize_qos_flow(flow)
                for flow in target_app["flows"]
                if isinstance(flow, dict)
            ]
    if flow_assignments:
        summary["qos_flow_assignments"] = flow_assignments

    mobility_plan = payload.get("mobility_plan") or {}
    if isinstance(mobility_plan, dict) and mobility_plan:
        summary["mobility_plan_present"] = True
        summary["mobility_association_id"] = mobility_plan.get("association_id")

    cross_verdicts = payload.get("cross_domain_verdicts") or []
    if cross_verdicts:
        summary["cross_domain_verdicts"] = cross_verdicts
    return summary


def _normalize_optimizer_status(payload: dict[str, Any]) -> str:
    infeasible_reasons = payload.get("infeasible_reasons") or []
    if infeasible_reasons:
        return "rejected"
    raw_status = str(payload.get("status") or "").strip()
    if raw_status:
        lowered = raw_status.lower()
        if lowered == "incomplete_context":
            return "incomplete_context"
        if lowered in {"rejected", "approved"}:
            return lowered
    qos_plan = payload.get("qos_plan") if isinstance(payload.get("qos_plan"), dict) else {}
    qos_meta = qos_plan.get("meta") if isinstance(qos_plan, dict) else {}
    qos_status = str((qos_meta or {}).get("status") or "").strip().lower()
    if "infeasible" in qos_status:
        return "rejected"
    if qos_status:
        return qos_status
    return raw_status or "unknown"


def _summarize_qos_flow(flow: dict[str, Any]) -> dict[str, Any]:
    allocation = flow.get("allocation") or {}
    telemetry = flow.get("telemetry") or {}
    sla = flow.get("sla") or {}
    return {
        "flow_id": flow.get("id"),
        "new_slice": allocation.get("current_slice_snssai"),
        "bw_ul": allocation.get("allocated_bandwidth_ul"),
        "bw_dl": allocation.get("allocated_bandwidth_dl"),
        "lat": telemetry.get("latency") or sla.get("latency"),
    }


__all__ = ["serialize_optimizer_result", "summarize_optimizer_result"]
