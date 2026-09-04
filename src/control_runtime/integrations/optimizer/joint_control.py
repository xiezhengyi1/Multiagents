from __future__ import annotations

import re
from typing import Any

from ...contracts import ControlDomain, DomainStatus, DomainVerdict, ObjectiveProfile
from .contracts import (
    JointOptimizationRequest,
    JointOptimizationResult,
    OptimizationProblemConfig,
)
from .interface import optimize_network_slices
from .joint_mobility import (
    build_mobility_draft,
    build_mobility_snapshot,
    run_cross_domain_checks,
)
from .joint_problem import resolve_problem_config
from .models import AMPolicyState
from .objective_weights import build_qos_optimizer_weights


def _relax_upper_bound(value: Any, ratio: float) -> Any:
    if value is None:
        return None
    numeric = float(value)
    return round(max(numeric * (1.0 + ratio), 0.0), 6)


def _relax_lower_bound(value: Any, ratio: float) -> Any:
    if value is None:
        return None
    numeric = float(value)
    return round(max(numeric * (1.0 - ratio), 0.0), 6)


def _normalize_snssai_ref(value: Any) -> str | None:
    if isinstance(value, dict):
        nested = value.get("snssai")
        if nested is not None and "sst" not in value and "sd" not in value:
            return _normalize_snssai_ref(nested)
        raw_sst = value.get("sst")
        raw_sd = value.get("sd")
        if raw_sst is None:
            return None
        try:
            sst = int(str(raw_sst).strip())
        except (TypeError, ValueError):
            return None
        sd = str(raw_sd or "").strip().upper()
        if sd and not re.fullmatch(r"[0-9A-F]{6}", sd):
            return None
        return f"{sst:02X}{sd}" if sd else f"{sst:02X}"

    text = str(value or "").strip()
    if not text:
        return None
    compact = re.sub(r"[^0-9A-Fa-f]", "", text).upper()
    if re.fullmatch(r"[0-9A-F]{8}", compact):
        return compact
    parts = [part for part in re.split(r"[^0-9A-Fa-f]+", text) if part]
    if (
        len(parts) >= 2
        and re.fullmatch(r"\d{1,3}", parts[0])
        and re.fullmatch(r"[0-9A-Fa-f]{6}", parts[1])
    ):
        return f"{int(parts[0]):02X}{parts[1].upper()}"
    return None


def _normalize_snssai_refs(values: Any) -> list[str]:
    if isinstance(values, list):
        raw_values = values
    elif values in (None, "", {}):
        raw_values = []
    else:
        raw_values = [values]

    normalized: list[str] = []
    for item in raw_values:
        if (snssai := _normalize_snssai_ref(item)) and snssai not in normalized:
            normalized.append(snssai)
    return normalized


def _build_optimizer_target_app_from_grounding_decision(
    grounding_decision: dict[str, Any],
) -> dict[str, Any]:
    raw_flows = grounding_decision.get("flows")
    if not isinstance(raw_flows, list) or not raw_flows:
        raise ValueError(
            "grounding_decision.flows must be a non-empty list for QoS optimization"
        )

    flow_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_flows:
        if not isinstance(item, dict):
            continue
        flow_id = str(item.get("flow_id") or "").strip()
        if flow_id:
            flow_by_id[flow_id] = dict(item)

    target_flows: list[dict[str, Any]] = []
    for flow_id, flow in flow_by_id.items():
        target_flows.append(dict(flow))

    if not target_flows:
        raise ValueError("grounding decision produced no resolved QoS flows")

    first_flow = target_flows[0]

    return {
        "app_id": first_flow.get("app_id"),
        "name": first_flow.get("app_name"),
        "supi": first_flow.get("supi"),
        "flows": target_flows,
    }


def _run_qos_subproblem(
    request: JointOptimizationRequest,
    *,
    problem_config: OptimizationProblemConfig,
    objective_profile: ObjectiveProfile,
    am_policy_state: AMPolicyState | None = None,
) -> tuple[dict[str, Any], list[str]]:
    target_app = _build_optimizer_target_app_from_grounding_decision(
        request.grounding_decision,
    )
    if "slice_assignment" not in problem_config.decision_variables:
        return {}, ["QoS optimization disabled by problem configuration"]

    optimizer_weights = build_qos_optimizer_weights(objective_profile)
    result = optimize_network_slices(
        target_app,
        *optimizer_weights.solver_arguments(),
        mode=problem_config.solver_mode,
        return_json=True,
        am_policy_state=am_policy_state,
        mobility_risk_weight=objective_profile.mobility_risk_cost,
        debug_context={
            "episode_id": request.scope.episode_id,
            "snapshot_id": request.scope.snapshot_id,
            "requested_domains": [domain.value for domain in request.requested_domains],
            "target_ues": list(request.target_ues or []),
            "policy_state": request.policy_state,
            "slice_kpi_source": problem_config.slice_kpi_source,
            "qos_relaxation_ratio": problem_config.qos_relaxation_ratio,
            "qos_feasibility_mode": problem_config.qos_feasibility_mode,
            "enable_sla_constraints": problem_config.enable_sla_constraints,
        },
    )
    if not isinstance(result, dict):
        return {}, ["QoS optimizer returned non-dict result"]
    if result.get("error"):
        return {}, [str(result["error"])]
    return result, []


def _collect_unassigned_requested_flow_errors(qos_plan: dict[str, Any]) -> list[str]:
    if not isinstance(qos_plan, dict):
        return []
    meta = qos_plan.get("meta") if isinstance(qos_plan.get("meta"), dict) else {}
    details = meta.get("infeasibility_details") if isinstance(meta, dict) else []
    if not isinstance(details, list):
        return []

    errors: list[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        flow_id = str(item.get("flow_id") or "").strip()
        app_name = str(item.get("app_name") or "").strip()
        candidate_slices = (
            item.get("candidate_slices")
            if isinstance(item.get("candidate_slices"), list)
            else []
        )
        slice_summaries: list[str] = []
        for candidate in candidate_slices:
            if not isinstance(candidate, dict):
                continue
            labels: list[str] = []
            for violation in candidate.get("violations") or []:
                if not isinstance(violation, dict):
                    continue
                constraint = str(violation.get("constraint") or "").strip()
                if "required_max" in violation:
                    labels.append(
                        f"{constraint}: actual={violation.get('actual')} > required_max={violation.get('required_max')}"
                    )
                elif "required_min" in violation:
                    labels.append(
                        f"{constraint}: actual={violation.get('actual')} < required_min={violation.get('required_min')}"
                    )
            for violation in candidate.get("node_violations") or []:
                if not isinstance(violation, dict):
                    continue
                labels.append(
                    f"{violation.get('constraint')}@{violation.get('node')}: actual={violation.get('actual')} < required_min={violation.get('required_min')}"
                )
            if labels:
                slice_summaries.append(
                    f"{candidate.get('snssai')}: " + ", ".join(labels)
                )
        summary = (
            "; ".join(slice_summaries)
            if slice_summaries
            else str(item.get("summary") or "no feasible slice assignment")
        )
        errors.append(
            f"target flow {flow_id or '<unknown>'} in app {app_name or '<unknown>'} has no grounded slice assignment: {summary}"
        )
    return errors


def _build_am_policy_state_from_request(
    request: JointOptimizationRequest,
) -> AMPolicyState | None:
    """关键步骤：从 JointOptimizationRequest 中提取当前 AM 策略状态用于 MILP 联合优化。"""
    policy_state = request.policy_state or {}
    target_supi = str((request.target_ues or [""])[0] or "").strip()

    # policy_state 结构为 {supi: ue_context}，需要先取出 UE 级别的数据
    ue_ctx = (
        policy_state.get(target_supi) or policy_state if target_supi else policy_state
    )
    am_ctx = (
        ue_ctx.get("amPolicyContext")
        or ue_ctx.get("am_policy")
        or ue_ctx.get("accessMobilityContext")
        or {}
    )
    if not am_ctx:
        return None

    mobility_summary = (
        ue_ctx.get("mobilitySummary") or ue_ctx.get("mobility_summary") or {}
    )
    raw_risk = mobility_summary.get("mobilityRiskScore")
    if raw_risk is None:
        raw_risk = mobility_summary.get("mobility_risk_score")
    try:
        mobility_risk_score = max(0.0, min(1.0, float(raw_risk or 0.0)))
    except (TypeError, ValueError):
        mobility_risk_score = 0.0

    return AMPolicyState(
        old_allowed_snssais=_normalize_snssai_refs(
            am_ctx.get("allowed_snssais") or am_ctx.get("allowedSnssais") or []
        ),
        old_target_snssais=_normalize_snssai_refs(
            am_ctx.get("target_snssais") or am_ctx.get("targetSnssais") or []
        ),
        old_rfsp=am_ctx.get("rfsp") or am_ctx.get("rfspIndex") or 1,
        old_triggers=am_ctx.get("triggers")
        or am_ctx.get("policyAssociationRequest", {}).get("triggers")
        or [],
        old_ue_ambr_ul=am_ctx.get("ue_ambr_ul") or 0.0,
        old_ue_ambr_dl=am_ctx.get("ue_ambr_dl") or 0.0,
        mobility_risk_score=mobility_risk_score,
    )


def run_joint_control_optimizer(
    request: JointOptimizationRequest,
) -> JointOptimizationResult:
    requested_domains = request.requested_domains or [ControlDomain.QOS]
    target_supi = str((request.target_ues or [""])[0] or "").strip()
    infeasible_reasons: list[str] = []
    qos_plan: dict[str, Any] = {}
    mobility_plan: dict[str, Any] = {}
    am_plan: dict[str, Any] = {}
    verdicts: list[DomainVerdict] = []
    problem_config, objective_profile = resolve_problem_config(request)

    # 关键步骤：当请求包含 MOBILITY 域时，尝试构建 AM 旧状态以启用联合优化
    am_policy_state: AMPolicyState | None = None
    if ControlDomain.MOBILITY in requested_domains:
        am_policy_state = _build_am_policy_state_from_request(request)

    if ControlDomain.QOS in requested_domains:
        qos_plan, qos_errors = _run_qos_subproblem(
            request,
            problem_config=problem_config,
            objective_profile=objective_profile,
            am_policy_state=am_policy_state,
        )
        qos_errors.extend(_collect_unassigned_requested_flow_errors(qos_plan))
        infeasible_reasons.extend(qos_errors)
        if qos_errors:
            verdicts.append(
                DomainVerdict(
                    domain=ControlDomain.QOS,
                    status=DomainStatus.REJECTED,
                    summary="QoS optimizer returned infeasible result",
                    hard_conflicts=[],
                    soft_conflicts=[],
                    infeasible_reasons=qos_errors,
                    metrics={},
                )
            )
        else:
            verdicts.append(
                DomainVerdict(
                    domain=ControlDomain.QOS,
                    status=DomainStatus.APPROVED,
                    summary="QoS optimizer produced a feasible plan",
                    hard_conflicts=[],
                    soft_conflicts=[],
                    infeasible_reasons=[],
                    metrics=qos_plan.get("meta", {}),
                )
            )
            # 关键步骤：从 QoS 求解结果中提取 AM 最优解
            breakdown = qos_plan.get("meta", {}).get("breakdown") or {}
            if isinstance(breakdown.get("am_solution"), dict):
                am_plan = breakdown.pop("am_solution")

    if ControlDomain.MOBILITY in requested_domains:
        if not target_supi:
            infeasible_reasons.append("mobility optimization requires a target SUPI")
            verdicts.append(
                DomainVerdict(
                    domain=ControlDomain.MOBILITY,
                    status=DomainStatus.INCOMPLETE_CONTEXT,
                    summary="Missing target SUPI for mobility planning",
                    hard_conflicts=[],
                    soft_conflicts=[],
                    infeasible_reasons=["missing target SUPI"],
                    metrics={},
                )
            )
        else:
            snapshot = build_mobility_snapshot(request, target_supi)
            if snapshot.missing_fields:
                infeasible_reasons.extend(
                    f"missing mobility context field: {item}"
                    for item in snapshot.missing_fields
                )
                verdicts.append(
                    DomainVerdict(
                        domain=ControlDomain.MOBILITY,
                        status=DomainStatus.INCOMPLETE_CONTEXT,
                        summary="Mobility context is incomplete",
                        hard_conflicts=[],
                        soft_conflicts=[],
                        infeasible_reasons=[
                            f"missing mobility context field: {item}"
                            for item in snapshot.missing_fields
                        ],
                        metrics={},
                    )
                )
            else:
                draft = build_mobility_draft(
                    request, target_supi, snapshot, qos_plan, am_plan=am_plan
                )
                mobility_plan = {
                    "association_id": draft.association_id,
                    "request": draft.request.model_dump(mode="json"),
                    "policy": draft.policy.model_dump(mode="json"),
                    "rationale": draft.rationale,
                    "trigger_event": draft.trigger_event,
                    "expected_benefits": draft.expected_benefits,
                }
                if "cross_domain_consistency" in problem_config.active_constraints:
                    verdicts.extend(
                        run_cross_domain_checks(
                            snapshot,
                            qos_plan,
                            draft,
                            problem_config=problem_config,
                        )
                    )

    overall_status = DomainStatus.APPROVED
    if infeasible_reasons:
        overall_status = DomainStatus.REJECTED
    if any(item.status == DomainStatus.INCOMPLETE_CONTEXT for item in verdicts):
        overall_status = DomainStatus.INCOMPLETE_CONTEXT

    return JointOptimizationResult(
        status=overall_status,
        qos_plan=qos_plan,
        mobility_plan=mobility_plan,
        am_plan=am_plan,
        cross_domain_verdicts=verdicts,
        objective_breakdown={
            "profile_name": objective_profile.profile_name,
            "template": problem_config.template.value,
            "solver_mode": problem_config.solver_mode,
            "active_objectives": list(problem_config.active_objectives),
            "active_constraints": list(problem_config.active_constraints),
            "decision_variables": list(problem_config.decision_variables),
            "grouped_decision_variables": problem_config.grouped_decision_variables(),
            "grouped_constraints": problem_config.grouped_constraints(),
            "qos_optimizer_weights": build_qos_optimizer_weights(
                objective_profile
            ).to_payload(),
            "mobility_risk_cost": objective_profile.mobility_risk_cost,
            "session_cost": (qos_plan.get("meta", {}).get("breakdown") or {}).get(
                "session_cost", 0.0
            ),
            "mobility_cost": (qos_plan.get("meta", {}).get("breakdown") or {}).get(
                "mobility_cost", 0.0
            ),
            "coupling_cost": (qos_plan.get("meta", {}).get("breakdown") or {}).get(
                "coupling_cost", 0.0
            ),
        },
        infeasible_reasons=infeasible_reasons,
    )
