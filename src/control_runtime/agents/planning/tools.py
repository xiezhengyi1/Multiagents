"""OSA request-scoped tool factory with explicit QoS / mobility tool partitions."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Optional

from langchain.tools import tool

from shared.tools.wrapper_think import tool_with_reason
from ...context.evidence import EvidenceFormatter
from ...domain.collaboration import PlanningRequest
from ...integrations.scenario.network_status import get_network_status_summary
from ...integrations.storage import get_ue_context_by_supi

from .planning_validation import json_friendly as _json_friendly


def _pick_first(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _normalize_optimizer_status(payload: Dict[str, Any]) -> str:
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


def _summarize_optimizer_result(result: Any) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    elif isinstance(result, dict):
        payload = _json_friendly(result)
    elif hasattr(result, "__dict__"):
        payload = _json_friendly(vars(result))
    else:
        payload = _json_friendly(result)
    if not isinstance(payload, dict):
        raise TypeError("optimizer result summary requires a mapping payload")
    qos_plan = payload.get("qos_plan") or {}
    qos_meta = qos_plan.get("meta") if isinstance(qos_plan, dict) else {}
    summary: Dict[str, Any] = {
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

    flow_assignments: List[Dict[str, Any]] = []
    target_apps = qos_plan.get("target_apps") if isinstance(qos_plan, dict) else []
    if isinstance(target_apps, list):
        for app_payload in target_apps:
            if not isinstance(app_payload, dict):
                continue
            for flow in app_payload.get("flows") or []:
                if not isinstance(flow, dict):
                    continue
                flow_assignments.append(
                    {
                        "flow_id": flow.get("id"),
                        "new_slice": ((flow.get("allocation") or {}).get("current_slice_snssai")),
                        "bw_ul": ((flow.get("allocation") or {}).get("allocated_bandwidth_ul")),
                        "bw_dl": ((flow.get("allocation") or {}).get("allocated_bandwidth_dl")),
                        "lat": ((flow.get("telemetry") or {}).get("latency") or (flow.get("sla") or {}).get("latency")),
                    }
                )
    if not flow_assignments:
        target_app = qos_plan.get("target_app") if isinstance(qos_plan, dict) else {}
        if isinstance(target_app, dict) and target_app.get("flows"):
            flow_assignments = [
                {
                    "flow_id": flow.get("id"),
                    "new_slice": ((flow.get("allocation") or {}).get("current_slice_snssai")),
                    "bw_ul": ((flow.get("allocation") or {}).get("allocated_bandwidth_ul")),
                    "bw_dl": ((flow.get("allocation") or {}).get("allocated_bandwidth_dl")),
                    "lat": ((flow.get("telemetry") or {}).get("latency") or (flow.get("sla") or {}).get("latency")),
                }
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


def _serialize_optimizer_result(result: Any) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    elif isinstance(result, dict):
        payload = _json_friendly(result)
    elif hasattr(result, "__dict__"):
        payload = _json_friendly(vars(result))
    else:
        payload = _json_friendly(result)
    if not isinstance(payload, dict):
        raise TypeError("optimizer result payload requires a mapping payload")
    return payload


def build_request_tools(
    planning_request: PlanningRequest,
    *,
    cached_tool_evidence: Optional[Dict[str, Any]] = None,
) -> tuple[List[Any], Dict[str, Any]]:
    from ...integrations.optimizer import run_joint_control_optimizer as run_optimizer
    _results_cache: Dict[str, Any] = {
        str(key): copy.deepcopy(value)
        for key, value in (cached_tool_evidence or {}).items()
        if value not in (None, "", [], {})
    }
    active_domains = {
        str(item.value if hasattr(item, "value") else item or "").strip().lower()
        for item in (planning_request.context.shared_context.main_intent.requested_domains or [])
        if str(item.value if hasattr(item, "value") else item or "").strip()
    }

    def _require_target_supi(candidate: Optional[str]) -> str:
        target = str(candidate or "").strip() or str(planning_request.context.shared_context.main_intent.supi or "").strip()
        if not target:
            target = next(
                (
                    str(flow.supi or "").strip()
                    for flow in (planning_request.grounding_decision.flows or [])
                    if str(flow.supi or "").strip()
                ),
                "",
            )
        if not target:
            raise ValueError("inspect_mobility_ue_policies requires a SUPI in the tool args or planning request")
        return target

    def _require_ue_context(target: str) -> Dict[str, Any]:
        snapshot_id = str(planning_request.context.snapshot_id or "").strip()
        if not snapshot_id:
            raise ValueError("inspect_mobility_ue_policies requires a bound snapshot_id")
        ue_ctx = get_ue_context_by_supi(target, snapshot_id=snapshot_id)
        if not ue_ctx:
            raise LookupError(f"No UE context found for {target}")
        trimmed: Dict[str, Any] = {}
        for key in (
            "supi",
            "accessMobilityContext",
            "amPolicyContext",
            "mobilitySummary",
            "servingNfContext",
        ):
            if key in ue_ctx:
                trimmed[key] = ue_ctx[key]
        if not trimmed:
            raise RuntimeError(f"UE context for {target} contains no policy-relevant mobility fields")
        return trimmed

    @tool_with_reason
    def preview_qos_optimizer(
        objective_profile: str = "balanced",
        optimization_template: str = "joint_balanced",
        qos_relaxation_ratio: float = 0.2,
        slice_kpi_source: str = "qos",
        qos_feasibility_mode: str = "auto",
    ) -> str:
        """Run the joint optimizer for executable planning evidence and return result plus summary."""
        cached_preview = _results_cache.get("latest_optimizer_preview")
        if isinstance(cached_preview, dict) and cached_preview:
            # A retry must use the already-grounded optimizer result rather
            # than rerun an expensive solve after a response-schema failure.
            return json.dumps(
                {"summary": _summarize_optimizer_result(cached_preview)},
                ensure_ascii=False,
            )
        request = EvidenceFormatter.for_optimizer(
            planning_request,
            profile_name=str(objective_profile or "balanced").strip().lower(),
            template_name=str(optimization_template or "joint_balanced").strip().lower(),
            qos_relaxation_ratio=qos_relaxation_ratio,
            slice_kpi_source=slice_kpi_source,
            qos_feasibility_mode=qos_feasibility_mode,
        )
        result = run_optimizer(request)
        full_payload = _serialize_optimizer_result(result)
        _results_cache["latest_optimizer_preview"] = full_payload
        summary = _summarize_optimizer_result(full_payload)
        current_slice_by_flow_id = {
            str(flow.flow_id or "").strip(): str(flow.current_slice_snssai or "").strip()
            for flow in (planning_request.grounding_decision.flows or [])
            if str(flow.flow_id or "").strip() and str(flow.current_slice_snssai or "").strip()
        }
        for assignment in summary.get("qos_flow_assignments") or []:
            if not isinstance(assignment, dict):
                continue
            flow_id = str(assignment.get("flow_id") or "").strip()
            current_slice = current_slice_by_flow_id.get(flow_id)
            if current_slice:
                assignment["current_slice"] = current_slice
                assignment["slice_changed"] = (
                    str(assignment.get("new_slice") or "").strip() != current_slice
                )
        return json.dumps(
            {"summary": summary},
            ensure_ascii=False,
        )

    @tool_with_reason
    def fetch_qos_network_status(service_type_id: Optional[int] = None) -> str:
        """Fetch QoS-domain network slice utilization and capacity summary."""
        return get_network_status_summary(
            flow_type_id=service_type_id,
            snapshot_id=planning_request.context.snapshot_id,
        )

    @tool_with_reason
    def inspect_mobility_ue_policies(supi: Optional[str] = None) -> str:
        """Inspect current UE mobility-relevant policy state and access-mobility context."""
        target = _require_target_supi(supi)
        trimmed = _require_ue_context(target)
        return json.dumps(_json_friendly(trimmed), ensure_ascii=False)

    request_tools: List[Any] = []
    if "qos" in active_domains:
        request_tools.extend([preview_qos_optimizer, fetch_qos_network_status])
    if "mobility" in active_domains:
        request_tools.append(inspect_mobility_ue_policies)
    if not request_tools:
        request_tools = [preview_qos_optimizer, fetch_qos_network_status, inspect_mobility_ue_policies]
    return request_tools, _results_cache


__all__ = ["build_request_tools", "_serialize_optimizer_result", "_summarize_optimizer_result"]

