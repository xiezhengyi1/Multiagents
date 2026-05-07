from __future__ import annotations

from ...domain.collaboration import PlanningRequest
from ...domain.control_plane import (
    ControlDomain,
    JointOptimizationRequest,
    ObjectiveProfile,
    OptimizationProblemConfig,
    OptimizationTemplate,
)
from ...integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi


def _collect_target_supis(planning_request: PlanningRequest) -> list[str]:
    operation_intent = planning_request.operation_intent
    target_supis: list[str] = []

    def _append(candidate: str) -> None:
        supi = str(candidate or "").strip()
        if supi and supi not in target_supis:
            target_supis.append(supi)

    _append(operation_intent.supi)
    for flow in operation_intent.flows or []:
        _append(flow.supi)
    return target_supis


def build_joint_optimizer_request(
    planning_request: PlanningRequest,
    *,
    profile_name: str | None = None,
    template_name: str | None = None,
    qos_relaxation_ratio: float | None = None,
    slice_kpi_source: str | None = None,
) -> JointOptimizationRequest:
    operation_intent = planning_request.operation_intent
    snapshot_id = str(planning_request.context.snapshot_id or "").strip()
    if not snapshot_id:
        raise ValueError("optimizer request requires a bound snapshot_id")
    snapshot = get_snapshot_data_by_id(snapshot_id) or {}
    if not snapshot:
        raise LookupError(f"optimizer request snapshot not found: snapshot_id={snapshot_id}")
    target_supis = _collect_target_supis(planning_request)

    requested_domains: list[ControlDomain] = []
    for item in planning_request.context.active_domains or []:
        normalized = str(item or "").strip().lower()
        if normalized in {"both", "all"}:
            requested_domains = [ControlDomain.QOS, ControlDomain.MOBILITY]
            break
        requested_domains.append(ControlDomain(normalized))

    ue_contexts = {supi: (get_ue_context_by_supi(supi, snapshot_id=snapshot_id) or {}) for supi in target_supis}

    objective_profile_payload = dict(planning_request.context.objective_profile or {"profile_name": "balanced"})
    if profile_name is not None:
        objective_profile_payload["profile_name"] = str(profile_name or "").strip().lower() or "balanced"

    problem_config = OptimizationProblemConfig()
    if template_name is not None:
        normalized_template = str(template_name or "").strip().lower()
        if not normalized_template:
            raise ValueError("template_name must not be empty when provided")
        problem_config = OptimizationProblemConfig(template=OptimizationTemplate(normalized_template))
    if qos_relaxation_ratio is not None:
        problem_config.qos_relaxation_ratio = float(qos_relaxation_ratio)
    if slice_kpi_source is not None:
        normalized_source = str(slice_kpi_source or "").strip().lower()
        if normalized_source not in {"qos", "telemetry"}:
            raise ValueError("slice_kpi_source must be either 'qos' or 'telemetry'")
        problem_config.slice_kpi_source = normalized_source

    return JointOptimizationRequest(
        session_id=planning_request.context.session_id,
        snapshot_id=snapshot_id,
        target_ues=target_supis,
        requested_domains=requested_domains or [ControlDomain.QOS],
        operation_intent=operation_intent.model_dump(mode="json"),
        traffic_state={
            "apps": snapshot.get("apps", []),
            "slices": snapshot.get("slices", []),
            "nodes": snapshot.get("nodes", []),
        },
        resource_state={
            "slices": snapshot.get("slices", []),
            "nodes": snapshot.get("nodes", []),
        },
        mobility_state={
            supi: ctx.get("accessMobilityContext", {})
            for supi, ctx in ue_contexts.items()
            if ctx
        },
        policy_state={
            supi: ctx
            for supi, ctx in ue_contexts.items()
            if ctx
        },
        objective_profile=ObjectiveProfile.model_validate(objective_profile_payload),
        problem_config=problem_config,
    )


__all__ = ["build_joint_optimizer_request"]
