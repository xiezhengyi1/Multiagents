from __future__ import annotations

from typing import Any, Literal

from ..storage import get_snapshot_data_by_id, get_ue_context_by_supi
from .contracts import (
    COUPLING_CONSTRAINTS,
    MOBILITY_CONSTRAINTS,
    MOBILITY_DECISION_VARIABLES,
    SESSION_CONSTRAINTS,
    SESSION_DECISION_VARIABLES,
    JointOptimizationRequest,
    OptimizationProblemConfig,
    OptimizationTemplate,
    OptimizerInput,
)


def build_optimizer_request(
    optimizer_input: OptimizerInput,
    *,
    template_name: Literal[
        "qos_first",
        "mobility_first",
        "joint_balanced",
        "stability_first",
        "congestion_relief",
    ],
    qos_relaxation_ratio: float,
    slice_kpi_source: Literal["qos", "telemetry"],
    qos_feasibility_mode: Literal["auto", "hard", "soft"],
) -> JointOptimizationRequest:
    snapshot_id = optimizer_input.scope.snapshot_id
    snapshot = get_snapshot_data_by_id(snapshot_id)
    target_supis = list(optimizer_input.target_supis)

    requested_domains = list(optimizer_input.requested_domains)
    ue_contexts = {
        supi: get_ue_context_by_supi(supi, snapshot_id=snapshot_id)
        for supi in target_supis
    }

    template = OptimizationTemplate(template_name)
    normalized_feasibility_mode = _resolve_qos_feasibility_mode(
        optimizer_input.raw_intent,
        requested_mode=qos_feasibility_mode,
    )
    problem_config = OptimizationProblemConfig(
        template=template,
        solver_mode="incremental",
        active_objectives=[
            "sla_violation_cost",
            "mobility_risk_cost",
            "control_churn_cost",
            "resource_pressure_cost",
            "fairness_cost",
        ],
        active_constraints=[
            *SESSION_CONSTRAINTS,
            *MOBILITY_CONSTRAINTS,
            *COUPLING_CONSTRAINTS,
        ],
        decision_variables=[
            *SESSION_DECISION_VARIABLES,
            *MOBILITY_DECISION_VARIABLES,
        ],
        qos_relaxation_ratio=float(qos_relaxation_ratio),
        slice_kpi_source=slice_kpi_source,
        qos_feasibility_mode=normalized_feasibility_mode,
        enable_sla_constraints=normalized_feasibility_mode == "hard",
    )

    optimizer_grounding_view = _build_optimizer_grounding_view(
        optimizer_input.grounding_decision,
        snapshot,
    )
    _apply_qos_operation_constraints(optimizer_grounding_view)
    if _slice_migration_is_blocked_by_iea(optimizer_grounding_view):
        raise ValueError("slice migration is not authorized by subscription evidence")

    return JointOptimizationRequest(
        scope=optimizer_input.scope,
        target_ues=target_supis,
        requested_domains=requested_domains,
        grounding_decision=optimizer_grounding_view,
        traffic_state={
            "flows": [
                flow for flow in snapshot["flows"] if flow["supi"] in target_supis
            ],
            "sessions": [
                session
                for session in snapshot["sessions"]
                if session["supi"] in target_supis
            ],
        },
        resource_state={
            "slice_states": snapshot["slice_states"],
            "ran_nodes": snapshot["ran_nodes"],
            "core_nodes": snapshot["core_nodes"],
            "containers": snapshot["containers"],
        },
        mobility_state={
            supi: ctx["access_mobility_context"]
            for supi, ctx in ue_contexts.items()
        },
        policy_state={supi: ctx for supi, ctx in ue_contexts.items()},
        objective_profile=optimizer_input.objective_profile,
        problem_config=problem_config,
    )


def _build_optimizer_grounding_view(
    grounding_decision_payload: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Bind grounded flow identities to authoritative live snapshot values."""
    payload = dict(grounding_decision_payload)
    flow_catalog = _snapshot_flow_catalog_by_id(snapshot)
    canonical_flows: list[dict[str, Any]] = []
    for flow_payload in payload["flows"]:
        if not isinstance(flow_payload, dict):
            raise TypeError("grounding flows must be objects")
        flow_id = str(flow_payload["flow_id"])
        native = flow_catalog[flow_id]
        canonical_flows.append(
            {
                **flow_payload,
                "flow_id": native["id"],
                "name": native["name"],
                "supi": native["supi"],
                "app_id": native["app_id"],
                "app_name": native["app_name"],
                "sla": native["sla"],
                "allocation": native["allocation"],
                "traffic": native["traffic"],
                "telemetry": native["telemetry"],
                "session": native["session"],
                "ran": native["ran"],
            }
        )
    payload["flows"] = canonical_flows
    return payload


def _apply_qos_operation_constraints(optimizer_grounding_view: dict[str, Any]) -> None:
    constraints = _derive_qos_operation_constraints(optimizer_grounding_view)
    if constraints:
        optimizer_grounding_view["qos_operation_constraints"] = constraints
    by_flow_id = {item["flow_id"]: item for item in constraints if item.get("flow_id")}
    for flow in optimizer_grounding_view.get("flows") or []:
        if not isinstance(flow, dict):
            continue
        flow_id = str(flow.get("flow_id") or "").strip()
        constraint = by_flow_id.get(flow_id)
        if not constraint:
            continue
        flow["require_slice_change"] = bool(constraint.get("require_slice_change"))
        flow["excluded_slice_snssais"] = list(constraint.get("excluded_slice_snssais") or [])
        flow["target_slice_preference"] = str(constraint.get("target_slice_preference") or "").strip()
        flow["no_op_allowed"] = bool(constraint.get("no_op_allowed", True))


def _derive_qos_operation_constraints(
    grounding_decision_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project grounded operation constraints into the optimizer's input view."""
    flow_by_id = {
        str(flow.get("flow_id") or "").strip(): flow
        for flow in (grounding_decision_payload.get("flows") or [])
        if isinstance(flow, dict) and str(flow.get("flow_id") or "").strip()
    }
    constraints: list[dict[str, Any]] = []
    for raw_constraint in grounding_decision_payload.get("qos_operation_constraints") or []:
        if not isinstance(raw_constraint, dict):
            continue
        flow_id = str(raw_constraint.get("flow_id") or "").strip()
        if not flow_id or flow_id not in flow_by_id:
            continue
        flow = flow_by_id[flow_id]
        source_slice = str(
            raw_constraint.get("source_slice_snssai")
            or flow.get("current_slice_snssai")
            or ""
        ).strip()
        excluded = [
            str(item or "").strip()
            for item in (raw_constraint.get("excluded_slice_snssais") or [])
            if str(item or "").strip()
        ]
        require_slice_change = bool(raw_constraint.get("require_slice_change", False))
        if require_slice_change and source_slice and source_slice not in excluded:
            excluded.append(source_slice)
        constraints.append(
            {
                "flow_id": flow_id,
                "app_id": str(raw_constraint.get("app_id") or flow.get("app_id") or "").strip(),
                "operation_type": str(raw_constraint.get("operation_type") or "").strip()
                or "qos_reallocation",
                "require_slice_change": require_slice_change,
                "source_slice_snssai": source_slice or None,
                "excluded_slice_snssais": excluded,
                "target_slice_preference": str(raw_constraint.get("target_slice_preference") or "").strip(),
                "no_op_allowed": bool(
                    raw_constraint.get("no_op_allowed", not require_slice_change)
                ),
                "rationale": list(raw_constraint.get("rationale") or []),
            }
        )
    return constraints


def _slice_migration_is_blocked_by_iea(optimizer_grounding_view: dict[str, Any]) -> bool:
    authorization = optimizer_grounding_view.get("slice_migration_authorization")
    if not isinstance(authorization, dict):
        return False
    return str(authorization.get("decision") or "").strip() in {
        "blocked_by_subscription_entitlement",
        "blocked_requires_subscription_provisioning",
        "evidence_missing",
    }


def _resolve_qos_feasibility_mode(
    raw_intent: str,
    *,
    requested_mode: Literal["auto", "hard", "soft"],
) -> Literal["hard", "soft"]:
    if requested_mode == "hard":
        return "hard"
    if requested_mode == "soft":
        return "soft"
    if requested_mode != "auto":
        raise ValueError("qos_feasibility_mode must be 'auto', 'hard', or 'soft'")

    raw_text = raw_intent.strip().lower()
    if not raw_text:
        return "soft"

    hard_sla_patterns = (
        "时延必须",
        "延迟必须",
        "抖动必须",
        "丢包必须",
        "带宽必须",
        "必须不超过",
        "不得超过",
        "不能超过",
        "不超过",
        "硬约束",
        "latency must",
        "jitter must",
        "loss must",
        "bandwidth must",
        "no more than",
        "at most",
        "hard constraint",
    )
    softening_markers = ("尽量", "尽可能", "最好", "优先", "prefer", "best effort", "best-effort")
    if any(pattern in raw_text for pattern in hard_sla_patterns) and not any(
        marker in raw_text for marker in softening_markers
    ):
        return "hard"

    soft_markers = (
        "尽量",
        "尽可能",
        "一些",
        "调稳",
        "调优",
        "优化",
        "优先",
        "看看",
        "best effort",
        "best-effort",
        "prefer",
        "priority",
        "optimize",
        "improve",
    )
    if any(marker in raw_text for marker in soft_markers):
        return "soft"

    return "soft"


def _snapshot_flow_catalog_by_id(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {flow["id"]: flow for flow in snapshot["flows"]}
