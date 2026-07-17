from __future__ import annotations

from typing import Any, Dict, List

from ...domain.collaboration import PlanningContext, PlanningRequest
from ...domain.control_plane import (
    ControlDomain,
    JointOptimizationRequest,
    ObjectiveProfile,
    OptimizationProblemConfig,
    OptimizationTemplate,
)
from ...domain.policy_plan import GroundingDecision
from ..projectors import ProjectorRegistry
from .normalizer import normalize_app_id


class EvidenceFormatter:
    """Unified evidence formatting for agent, optimizer, and LLM consumers."""

    @classmethod
    def for_iea(
        cls,
        *,
        user_input: str,
        supi: str,
        main_directives: Dict[str, Any],
        catalog_payload: Dict[str, Any],
        catalog_evidence_observed: bool = False,
        semantic_candidates: List[Dict[str, Any]],
        am_context_payload: Dict[str, Any] | None = None,
        am_policy_candidates: List[Dict[str, Any]] | None = None,
        subscription_payload: Dict[str, Any] | None = None,
    ) -> Any:
        # Keep the mature grounding builder as the implementation source while
        # moving the public construction boundary into context.evidence.
        from .grounding import IntentEvidenceBuilder

        return IntentEvidenceBuilder().build_intent_evidence(
            user_input=user_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            catalog_evidence_observed=catalog_evidence_observed,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
            subscription_payload=subscription_payload,
        )

    @classmethod
    def for_osa(
        cls,
        *,
        grounding_decision: GroundingDecision,
        planning_context: PlanningContext,
    ) -> Dict[str, Any]:
        operation_constraints = _qos_operation_constraints_for_flows(
            grounding_decision.model_dump(mode="json"),
            planning_context,
        )
        return (
            {"derived_operation_constraints": operation_constraints}
            if operation_constraints
            else {}
        )

    @classmethod
    def for_optimizer(
        cls,
        planning_request: PlanningRequest,
        *,
        profile_name: str | None = None,
        template_name: str | None = None,
        qos_relaxation_ratio: float | None = None,
        slice_kpi_source: str | None = None,
        qos_feasibility_mode: str | None = None,
    ) -> JointOptimizationRequest:
        from ...integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi

        grounding_decision = planning_request.grounding_decision
        main_intent = planning_request.context.shared_context.main_intent
        snapshot_id = str(planning_request.context.snapshot_id or "").strip()
        if not snapshot_id:
            raise ValueError("optimizer request requires a bound snapshot_id")
        snapshot = get_snapshot_data_by_id(snapshot_id) or {}
        if not snapshot:
            raise LookupError(f"optimizer request snapshot not found: snapshot_id={snapshot_id}")
        target_supis = _collect_target_supis(planning_request)

        requested_domains: list[ControlDomain] = []
        requested_domains = list(main_intent.requested_domains or [])
        if not requested_domains:
            raise ValueError("optimizer request requires non-empty Main requested_domains")

        ue_contexts = {
            supi: (get_ue_context_by_supi(supi, snapshot_id=snapshot_id) or {})
            for supi in target_supis
        }

        objective_profile_payload = main_intent.objective_profile.model_dump(mode="json")
        if not objective_profile_payload:
            raise ValueError("optimizer request requires an explicit objective_profile")
        if profile_name is not None:
            normalized_profile = str(profile_name or "").strip().lower()
            if not normalized_profile:
                raise ValueError("profile_name must not be empty when provided")
            objective_profile_payload["profile_name"] = normalized_profile

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
        normalized_feasibility_mode = _resolve_qos_feasibility_mode(
            planning_request,
            requested_mode=qos_feasibility_mode,
        )
        problem_config.qos_feasibility_mode = normalized_feasibility_mode
        problem_config.enable_sla_constraints = normalized_feasibility_mode == "hard"

        optimizer_grounding_view = _build_optimizer_grounding_view(
            grounding_decision.model_dump(mode="json"),
            snapshot,
        )
        _apply_qos_operation_constraints(
            optimizer_grounding_view,
            planning_request.context,
        )
        if _slice_migration_is_blocked_by_iea(optimizer_grounding_view):
            _apply_entitlement_limited_qos_delivery(optimizer_grounding_view)
            # Hybrid mode pins the serving S-NSSAI while leaving the QoS
            # envelope adjustable. This is a code-owned execution boundary;
            # the LLM still decides whether the resulting best-effort plan is
            # useful enough to deliver.
            problem_config.solver_mode = "hybrid"

        return JointOptimizationRequest(
            session_id=planning_request.context.session_id,
            snapshot_id=snapshot_id,
            target_ues=target_supis,
            requested_domains=requested_domains,
            grounding_decision=optimizer_grounding_view,
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
            policy_state={supi: ctx for supi, ctx in ue_contexts.items() if ctx},
            objective_profile=ObjectiveProfile.model_validate(objective_profile_payload),
            problem_config=problem_config,
        )

    @classmethod
    def for_llm(cls, *, model: Any, role: str = "") -> dict[str, Any]:
        projector = ProjectorRegistry.for_instance(model)
        return projector.project(model)


def _collect_target_supis(planning_request: PlanningRequest) -> list[str]:
    grounding_decision = planning_request.grounding_decision
    main_intent = planning_request.context.shared_context.main_intent
    target_supis: list[str] = []

    def _append(candidate: str) -> None:
        supi = str(candidate or "").strip()
        if supi and supi not in target_supis:
            target_supis.append(supi)

    _append(main_intent.supi)
    for stage in main_intent.control_semantics.stages or []:
        for target in stage.targets or []:
            _append(target.supi)
    for flow in grounding_decision.flows or []:
        _append(flow.supi)
    return target_supis


def _build_optimizer_grounding_view(
    grounding_decision_payload: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the typed IEA result with snapshot defaults for the optimizer.

    This is a private runtime view, not an agent artifact: Main's scope remains
    in shared context and the original GroundingDecision is never mutated.
    """
    payload = dict(grounding_decision_payload or {})
    flow_catalog = _snapshot_flow_catalog_by_id(snapshot)
    canonical_flows: List[Dict[str, Any]] = []
    for flow_payload in payload.get("flows") or []:
        if not isinstance(flow_payload, dict):
            canonical_flows.append(flow_payload)
            continue
        flow = dict(flow_payload)
        flow_id = str(flow.get("flow_id") or flow.get("id") or "").strip()
        catalog_entry = flow_catalog.get(flow_id)
        if catalog_entry is not None:
            _apply_snapshot_flow_defaults(flow, catalog_entry)
        canonical_flows.append(flow)
    payload["flows"] = canonical_flows
    return payload


def _qos_operation_constraints_for_flows(
    grounding_decision_payload: Dict[str, Any],
    planning_context: PlanningContext,
) -> List[Dict[str, Any]]:
    flow_by_id = {
        str(flow.get("flow_id") or "").strip(): flow
        for flow in (grounding_decision_payload.get("flows") or [])
        if isinstance(flow, dict) and str(flow.get("flow_id") or "").strip()
    }
    constraints: List[Dict[str, Any]] = []
    for raw_constraint in grounding_decision_payload.get("qos_operation_constraints") or []:
        if not isinstance(raw_constraint, dict):
            continue
        constraint = _normalize_qos_operation_constraint(raw_constraint, flow_by_id)
        if constraint:
            constraints.append(constraint)

    # Main's global migration cue remains available in shared context for
    # routing and audit, but it is not a per-flow execution constraint. IEA
    # alone decides whether an explicitly grounded flow requires a serving
    # S-NSSAI change; synthesizing such a constraint here made same-slice QoS
    # tuning incorrectly require AM policy output.
    return constraints


def _normalize_qos_operation_constraint(
    raw_constraint: Dict[str, Any],
    flow_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    flow_id = str(raw_constraint.get("flow_id") or "").strip()
    if not flow_id or flow_id not in flow_by_id:
        return None
    flow = flow_by_id[flow_id]
    source_slice = str(raw_constraint.get("source_slice_snssai") or flow.get("current_slice_snssai") or "").strip()
    excluded = [
        str(item or "").strip()
        for item in (raw_constraint.get("excluded_slice_snssais") or [])
        if str(item or "").strip()
    ]
    require_slice_change = bool(raw_constraint.get("require_slice_change", False))
    no_op_allowed = bool(raw_constraint.get("no_op_allowed", not require_slice_change))
    if require_slice_change and source_slice and source_slice not in excluded:
        excluded.append(source_slice)
    return {
        "flow_id": flow_id,
        "app_id": normalize_app_id(raw_constraint.get("app_id") or flow.get("app_id")),
        "operation_type": str(raw_constraint.get("operation_type") or "").strip() or "qos_reallocation",
        "require_slice_change": require_slice_change,
        "source_slice_snssai": source_slice or None,
        "excluded_slice_snssais": excluded,
        "target_slice_preference": str(raw_constraint.get("target_slice_preference") or "").strip(),
        "no_op_allowed": no_op_allowed,
        "rationale": list(raw_constraint.get("rationale") or []),
    }


def _apply_qos_operation_constraints(
    optimizer_grounding_view: Dict[str, Any],
    planning_context: PlanningContext,
) -> None:
    constraints = _qos_operation_constraints_for_flows(optimizer_grounding_view, planning_context)
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


def _slice_migration_is_blocked_by_iea(optimizer_grounding_view: Dict[str, Any]) -> bool:
    authorization = optimizer_grounding_view.get("slice_migration_authorization")
    if not isinstance(authorization, dict):
        return False
    return str(authorization.get("decision") or "").strip() in {
        "blocked_by_subscription_entitlement",
        "blocked_requires_subscription_provisioning",
        "evidence_missing",
    }


def _apply_entitlement_limited_qos_delivery(optimizer_grounding_view: Dict[str, Any]) -> None:
    """Turn a blocked migration into an optimizer-safe QoS-only fallback.

    Subscription evidence is an execution boundary: the optimizer may tune
    bandwidth/QoS on the serving slice but cannot select a new S-NSSAI.
    """
    optimizer_grounding_view["entitlement_limited_best_effort"] = True
    optimizer_grounding_view["preserve_current_slice"] = True
    for constraint in optimizer_grounding_view.get("qos_operation_constraints") or []:
        if not isinstance(constraint, dict):
            continue
        constraint["operation_type"] = "qos_reallocation"
        constraint["require_slice_change"] = False
        constraint["excluded_slice_snssais"] = []
        constraint["target_slice_preference"] = ""
        constraint["no_op_allowed"] = True
        rationale = list(constraint.get("rationale") or [])
        constraint["rationale"] = list(
            dict.fromkeys(
                [*rationale, "Subscription boundary: preserve current slice and optimize QoS best effort"]
            )
        )
    for flow in optimizer_grounding_view.get("flows") or []:
        if not isinstance(flow, dict):
            continue
        flow["require_slice_change"] = False
        flow["excluded_slice_snssais"] = []
        flow["target_slice_preference"] = ""
        flow["no_op_allowed"] = True
        flow["preserve_current_slice"] = True


def _resolve_qos_feasibility_mode(
    planning_request: PlanningRequest,
    *,
    requested_mode: str | None,
) -> str:
    normalized = str(requested_mode or "auto").strip().lower()
    if normalized in {"hard", "strict"}:
        return "hard"
    if normalized in {"soft", "best_effort", "best-effort"}:
        return "soft"
    if normalized not in {"", "auto"}:
        raise ValueError("qos_feasibility_mode must be 'auto', 'hard', or 'soft'")

    raw_text = str(
        planning_request.context.shared_context.main_intent.raw_input
    ).strip().lower()
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


def _snapshot_flow_catalog_by_id(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for app in snapshot.get("apps") or []:
        if not isinstance(app, dict):
            continue
        for flow in app.get("flows") or []:
            if not isinstance(flow, dict):
                continue
            flow_id = str(flow.get("flow_id") or flow.get("id") or "").strip()
            if flow_id:
                catalog[flow_id] = {"app": app, "flow": flow}
    return catalog


def _apply_snapshot_flow_defaults(flow: Dict[str, Any], catalog_entry: Dict[str, Any]) -> None:
    app = catalog_entry["app"]
    snapshot_flow = catalog_entry["flow"]
    service = snapshot_flow.get("service") if isinstance(snapshot_flow.get("service"), dict) else {}
    sla = snapshot_flow.get("sla") if isinstance(snapshot_flow.get("sla"), dict) else {}
    allocation = snapshot_flow.get("allocation") if isinstance(snapshot_flow.get("allocation"), dict) else {}

    canonical_values = {
        "supi": _coalesce(_first_present(snapshot_flow, "supi"), _first_present(app, "supi")),
        "app_id": _coalesce(_first_present(app, "app_id", "id"), _first_present(snapshot_flow, "app_id")),
        "app_name": _coalesce(_first_present(app, "app_name", "name"), _first_present(snapshot_flow, "app_name")),
        "flow_id": _first_present(snapshot_flow, "flow_id", "id"),
        "name": _first_present(snapshot_flow, "flow_name", "name"),
        "service_type": _coalesce(_first_present(snapshot_flow, "service_type"), _first_present(service, "service_type")),
        "service_type_id": _coalesce(_first_present(snapshot_flow, "service_type_id"), _first_present(service, "service_type_id")),
        "bw_ul": _coalesce(
            _first_present(snapshot_flow, "bw_ul", "bandwidth_ul", "max_br_ul_mbps"),
            _first_present(sla, "bandwidth_ul", "max_br_ul_mbps"),
        ),
        "bw_dl": _coalesce(
            _first_present(snapshot_flow, "bw_dl", "bandwidth_dl", "max_br_dl_mbps"),
            _first_present(sla, "bandwidth_dl", "max_br_dl_mbps"),
        ),
        "gbr_ul": _coalesce(
            _first_present(snapshot_flow, "gbr_ul", "guaranteed_bandwidth_ul", "gbr_ul_mbps"),
            _first_present(sla, "guaranteed_bandwidth_ul", "gbr_ul_mbps"),
        ),
        "gbr_dl": _coalesce(
            _first_present(snapshot_flow, "gbr_dl", "guaranteed_bandwidth_dl", "gbr_dl_mbps"),
            _first_present(sla, "guaranteed_bandwidth_dl", "gbr_dl_mbps"),
        ),
        "lat": _coalesce(
            _first_present(snapshot_flow, "lat", "latency", "latency_ms"),
            _first_present(sla, "latency", "latency_ms"),
        ),
        "loss_req": _coalesce(
            _first_present(snapshot_flow, "loss_req", "loss_rate", "packet_error_rate"),
            _first_present(sla, "loss_rate", "packet_error_rate"),
        ),
        "jitter_req": _coalesce(
            _first_present(snapshot_flow, "jitter_req", "jitter", "jitter_ms"),
            _first_present(sla, "jitter", "jitter_ms"),
        ),
        "priority": _coalesce(_first_present(snapshot_flow, "priority"), _first_present(sla, "priority")),
        "current_slice_snssai": _coalesce(
            _first_present(snapshot_flow, "current_slice_snssai"),
            _first_present(allocation, "current_slice_snssai"),
        ),
        "current_bw_ul": _coalesce(
            _first_present(snapshot_flow, "current_bw_ul"),
            _first_present(allocation, "allocated_bandwidth_ul"),
        ),
        "current_bw_dl": _coalesce(
            _first_present(snapshot_flow, "current_bw_dl"),
            _first_present(allocation, "allocated_bandwidth_dl"),
        ),
    }

    for key in ("supi", "app_id", "app_name", "flow_id", "name"):
        if canonical_values.get(key) is not None:
            flow[key] = canonical_values[key]
    for key, value in canonical_values.items():
        if key in {"supi", "app_id", "app_name", "flow_id", "name"}:
            continue
        if _is_missing(flow.get(key)) and value is not None:
            flow[key] = value


def _first_present(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if not _is_missing(value):
            return value
    return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
