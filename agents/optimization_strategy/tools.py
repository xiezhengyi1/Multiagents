"""OSA request-scoped tool factory."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from langchain.tools import tool

from agents.tools.wrapper_think import tool_with_reason

from agents.tools.db_tool import get_ue_context_by_supi
from agents.tools.network_status import get_network_status_summary
from domain.collaboration import PlanningRequest

from .policy_normalizer import json_friendly as _json_friendly
from .request_builder import build_joint_optimizer_request


def _pick_first(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


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
        "status": payload.get("status"),
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

    target_app = qos_plan.get("target_app") if isinstance(qos_plan, dict) else {}
    if isinstance(target_app, dict) and target_app.get("flows"):
        summary["qos_flow_assignments"] = [
            {
                "flow_id": _pick_first(flow, ("flow_id", "Flow ID")),
                "new_slice": _pick_first(flow, ("new_slice", "New Slice")),
                "bw_ul": _pick_first(flow, ("bw_ul", "Act BW UL", "Req BW UL")),
                "bw_dl": _pick_first(flow, ("bw_dl", "Act BW DL", "Req BW DL")),
                "lat": _pick_first(flow, ("lat", "Latency", "Req Lat")),
            }
            for flow in target_app["flows"]
            if isinstance(flow, dict)
        ]

    mobility_plan = payload.get("mobility_plan") or {}
    if isinstance(mobility_plan, dict) and mobility_plan:
        summary["mobility_plan_present"] = True
        summary["mobility_association_id"] = mobility_plan.get("association_id")

    cross_verdicts = payload.get("cross_domain_verdicts") or []
    if cross_verdicts:
        summary["cross_domain_verdicts"] = cross_verdicts
    return summary


def build_request_tools(planning_request: PlanningRequest) -> List[Any]:
    from agents.tools.optimizer import run_joint_control_optimizer as run_optimizer

    @tool_with_reason
    def preview_optimizer(
        objective_profile: str = "balanced",
        optimization_template: str = "joint_balanced",
    ) -> str:
        """Run the joint optimizer with a specific profile or template and return a summary."""
        request = build_joint_optimizer_request(
            planning_request,
            profile_name=str(objective_profile or "balanced").strip().lower(),
            template_name=str(optimization_template or "joint_balanced").strip().lower(),
        )
        result = run_optimizer(request)
        return json.dumps(_summarize_optimizer_result(result), ensure_ascii=False)

    @tool_with_reason
    def fetch_network_status(service_type_id: Optional[int] = None) -> str:
        """Fetch current network slice utilization and capacity summary."""
        return get_network_status_summary(flow_type_id=service_type_id)

    @tool_with_reason
    def inspect_ue_policies(supi: Optional[str] = None) -> str:
        """Inspect current UE AM/SM policies and mobility context."""
        target = str(supi or "").strip() or str(planning_request.operation_intent.supi or "").strip()
        if not target:
            raise ValueError("inspect_ue_policies requires a SUPI")
        ue_ctx = get_ue_context_by_supi(target)
        if not ue_ctx:
            raise ValueError(f"No UE context found for {target}")

        trimmed: Dict[str, Any] = {}
        for key in (
            "supi",
            "smPolicyData",
            "pccRules",
            "qosDecs",
            "urspRules",
            "accessMobilityContext",
            "amPolicyContext",
            "mobilitySummary",
            "servingNfContext",
        ):
            if key in ue_ctx:
                trimmed[key] = ue_ctx[key]
        if not trimmed:
            raise ValueError(f"UE context for {target} contains no policy-relevant fields")
        return json.dumps(_json_friendly(trimmed), ensure_ascii=False)

    return [preview_optimizer, fetch_network_status, inspect_ue_policies]


__all__ = ["build_request_tools", "_summarize_optimizer_result"]
