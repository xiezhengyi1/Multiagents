"""OSA request-scoped tool factory with explicit QoS / mobility tool partitions."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from langchain.tools import tool

from shared.tools.wrapper_think import tool_with_reason
from ...integrations.optimizer.contracts import OptimizerInput
from ...integrations.optimizer.request_factory import build_optimizer_request
from ...integrations.optimizer.result_view import (
    serialize_optimizer_result,
    summarize_optimizer_result,
)
from .contracts import PlanningRequest
from ...integrations.scenario.network_status import get_network_status_summary
from ...integrations.storage import get_ue_context_by_supi
from .planning_validation import json_friendly


def _optimizer_input_from_planning_request(planning_request: PlanningRequest) -> OptimizerInput:
    main_intent = planning_request.context.shared_context.main_intent
    candidates = [main_intent.scope.supi]
    candidates.extend(
        target.supi
        for stage in main_intent.control_semantics.stages
        for target in stage.targets
    )
    candidates.extend(flow.supi for flow in planning_request.grounding_decision.flows)
    target_supis = list(
        dict.fromkeys(
            normalized
            for candidate in candidates
            if (normalized := str(candidate or "").strip())
        )
    )
    return OptimizerInput(
        session_id=planning_request.context.session_id,
        snapshot_id=planning_request.context.snapshot_id,
        target_supis=target_supis,
        requested_domains=main_intent.requested_domains,
        grounding_decision=planning_request.grounding_decision.model_dump(mode="json"),
        objective_profile=main_intent.objective_profile,
        raw_intent=main_intent.raw_input,
    )


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
        target = str(candidate or "").strip() or (
            planning_request.context.shared_context.main_intent.scope.supi or ""
        )
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

    @tool_with_reason(side_effect="read", result_format="json")
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
                {"summary": summarize_optimizer_result(cached_preview)},
                ensure_ascii=False,
            )
        request = build_optimizer_request(
            _optimizer_input_from_planning_request(planning_request),
            profile_name=str(objective_profile or "balanced").strip().lower(),
            template_name=str(optimization_template or "joint_balanced").strip().lower(),
            qos_relaxation_ratio=qos_relaxation_ratio,
            slice_kpi_source=slice_kpi_source,
            qos_feasibility_mode=qos_feasibility_mode,
        )
        result = run_optimizer(request)
        full_payload = serialize_optimizer_result(result)
        _results_cache["latest_optimizer_preview"] = full_payload
        summary = summarize_optimizer_result(full_payload)
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

    @tool_with_reason(side_effect="read", result_format="json")
    def fetch_qos_network_status(service_type_id: Optional[int] = None) -> str:
        """Fetch QoS-domain network slice utilization and capacity summary."""
        return get_network_status_summary(
            flow_type_id=service_type_id,
            snapshot_id=planning_request.context.snapshot_id,
        )

    @tool_with_reason(side_effect="read", result_format="json")
    def inspect_mobility_ue_policies(supi: Optional[str] = None) -> str:
        """Inspect current UE mobility-relevant policy state and access-mobility context."""
        target = _require_target_supi(supi)
        trimmed = _require_ue_context(target)
        return json.dumps(json_friendly(trimmed), ensure_ascii=False)

    request_tools: List[Any] = []
    if "qos" in active_domains:
        request_tools.extend([preview_qos_optimizer, fetch_qos_network_status])
    if "mobility" in active_domains:
        request_tools.append(inspect_mobility_ue_policies)
    if not request_tools:
        request_tools = [preview_qos_optimizer, fetch_qos_network_status, inspect_mobility_ue_policies]
    return request_tools, _results_cache


__all__ = ["build_request_tools"]

