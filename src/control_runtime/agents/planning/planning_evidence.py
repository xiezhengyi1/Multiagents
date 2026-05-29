from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...domain.collaboration import PlanningRequest
from .policy_normalizer import normalize_app_id as _normalize_app_id


def build_slice_snssai(slice_code: str) -> Optional[Dict[str, Any]]:
    code = str(slice_code or "").strip()
    if len(code) < 8:
        return None
    try:
        sst = int(code[:2], 16)
    except ValueError:
        return None
    return {"sst": sst, "sd": code[2:8]}


class PlanningEvidenceBuilder:
    def build_planning_evidence(self, planning_request: PlanningRequest) -> Dict[str, Any]:
        operation_intent = planning_request.operation_intent
        flows: List[Dict[str, Any]] = []
        for flow in operation_intent.flows:
            flows.append(
                {
                    "flow_id": str(flow.flow_id or "").strip(),
                    "app_id": _normalize_app_id(flow.app_id),
                    "name": str(flow.name or "").strip(),
                    "priority": flow.priority,
                    "service_type_id": flow.service_type_id,
                }
            )
        qos_objectives = [
            objective.model_dump(mode="json")
            for objective in operation_intent.qos_target_envelopes
        ]
        return {
            "requested_domains": list(planning_request.context.active_domains or []),
            "main_retry_scope": str(planning_request.context.main_retry_scope or "").strip(),
            "objective_profile": dict(planning_request.context.objective_profile or {}),
            "required_evidence": list(planning_request.context.required_evidence or []),
            "forbidden_assumptions": list(planning_request.context.forbidden_assumptions or []),
            "revision_requests": list(planning_request.context.revision_requests or []),
            "unified_constraints": dict(planning_request.context.unified_constraints or {}),
            "flows": flows,
            "qos_target_envelopes": qos_objectives,
        }


__all__ = ["PlanningEvidenceBuilder", "build_slice_snssai"]
