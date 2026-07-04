from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ...domain.control_plane import (
    ControlDomain,
    DomainStatus,
    DomainVerdict,
    JointOptimizationRequest,
    JointOptimizationResult,
    ObjectiveProfile,
    OptimizationProblemConfig,
)

from .models import AMPolicyState
from .interface import optimize_network_slices
from .joint_mobility import build_mobility_draft, build_mobility_snapshot, run_cross_domain_checks
from .joint_problem import resolve_problem_config


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


def _normalize_snssai_ref(value: Any) -> Optional[str]:
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
    if len(parts) >= 2 and re.fullmatch(r"\d{1,3}", parts[0]) and re.fullmatch(r"[0-9A-Fa-f]{6}", parts[1]):
        return f"{int(parts[0]):02X}{parts[1].upper()}"
    return None


def _normalize_snssai_refs(values: Any) -> List[str]:
    if isinstance(values, list):
        raw_values = values
    elif values in (None, "", {}):
        raw_values = []
    else:
        raw_values = [values]

    normalized: List[str] = []
    for item in raw_values:
        if (snssai := _normalize_snssai_ref(item)) and snssai not in normalized:
            normalized.append(snssai)
    return normalized


def _resolve_target_app_payload(request: JointOptimizationRequest) -> Optional[Dict[str, Any]]:
    operation_intent = request.operation_intent or {}
    traffic_state = request.traffic_state or {}
    if isinstance(operation_intent, dict) and operation_intent.get("flows"):
        return {
            "app_id": operation_intent.get("app_id"),
            "name": operation_intent.get("app_name"),
            "supi": operation_intent.get("supi"),
            "flows": operation_intent.get("flows"),
        }

    apps = traffic_state.get("apps")
    if not isinstance(apps, list):
        return None

    target_supi = str((request.target_ues or [""])[0] or "").strip()
    target_app_id = str(operation_intent.get("app_id") or "").strip()
    for app in apps:
        if not isinstance(app, dict):
            continue
        if target_app_id and str(app.get("id") or "").strip() == target_app_id:
            return app
        if target_supi and str(app.get("supi") or "").strip() == target_supi:
            return app
    return apps[0] if apps else None


def _build_optimizer_target_app_from_operation_intent(
    operation_intent: Dict[str, Any],
    *,
    qos_relaxation_ratio: float,
) -> Dict[str, Any]:
    raw_flows = operation_intent.get("flows")
    if not isinstance(raw_flows, list) or not raw_flows:
        raise ValueError("operation_intent.flows must be a non-empty list for QoS optimization")

    flow_by_id: Dict[str, Dict[str, Any]] = {}
    for item in raw_flows:
        if not isinstance(item, dict):
            continue
        flow_id = str(item.get("flow_id") or "").strip()
        if flow_id:
            flow_by_id[flow_id] = dict(item)

    envelopes = operation_intent.get("qos_target_envelopes")
    if not isinstance(envelopes, list) or not envelopes:
        raise ValueError("QoS optimization requires non-empty operation_intent.qos_target_envelopes")

    target_flows: List[Dict[str, Any]] = []
    matched_flow_ids: set[str] = set()
    for envelope in envelopes:
        if not isinstance(envelope, dict):
            continue
        flow_id = str(envelope.get("flow_id") or "").strip()
        if not flow_id:
            continue
        base_flow = flow_by_id.get(flow_id)
        if base_flow is None:
            raise ValueError(f"QoS target envelope references unknown flow_id={flow_id}")
        matched_flow_ids.add(flow_id)
        flow_payload = dict(base_flow)
        if envelope.get("strictest_priority") is not None:
            flow_payload["priority"] = envelope.get("strictest_priority")
        if envelope.get("strictest_latency_ms") is not None:
            flow_payload["lat"] = _relax_upper_bound(envelope.get("strictest_latency_ms"), qos_relaxation_ratio)
        if envelope.get("strictest_jitter_ms") is not None:
            flow_payload["jitter_req"] = _relax_upper_bound(envelope.get("strictest_jitter_ms"), qos_relaxation_ratio)
        if envelope.get("strictest_packet_error_rate") is not None:
            flow_payload["loss_req"] = _relax_upper_bound(envelope.get("strictest_packet_error_rate"), qos_relaxation_ratio)
        if envelope.get("strictest_max_br_ul_mbps") is not None:
            flow_payload["bw_ul"] = _relax_lower_bound(envelope.get("strictest_max_br_ul_mbps"), qos_relaxation_ratio)
        if envelope.get("strictest_max_br_dl_mbps") is not None:
            flow_payload["bw_dl"] = _relax_lower_bound(envelope.get("strictest_max_br_dl_mbps"), qos_relaxation_ratio)
        if envelope.get("strictest_gbr_ul_mbps") is not None:
            flow_payload["gbr_ul"] = _relax_lower_bound(envelope.get("strictest_gbr_ul_mbps"), qos_relaxation_ratio)
        if envelope.get("strictest_gbr_dl_mbps") is not None:
            flow_payload["gbr_dl"] = _relax_lower_bound(envelope.get("strictest_gbr_dl_mbps"), qos_relaxation_ratio)
        target_flows.append(flow_payload)

    if not target_flows:
        raise ValueError("QoS target envelopes produced no optimizer target flows")

    for item in raw_flows:
        if not isinstance(item, dict):
            continue
        flow_id = str(item.get("flow_id") or "").strip()
        if flow_id and flow_id not in matched_flow_ids:
            target_flows.append(dict(item))

    return {
        "app_id": operation_intent.get("app_id"),
        "name": operation_intent.get("app_name"),
        "supi": operation_intent.get("supi"),
        "flows": target_flows,
    }


def _run_qos_subproblem(
    request: JointOptimizationRequest,
    *,
    problem_config: OptimizationProblemConfig,
    objective_profile: ObjectiveProfile,
    am_policy_state: Optional[AMPolicyState] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    try:
        target_app = _build_optimizer_target_app_from_operation_intent(
            request.operation_intent or {},
            qos_relaxation_ratio=problem_config.qos_relaxation_ratio,
        )
    except Exception as exc:
        return {}, [str(exc)]
    if target_app is None:
        target_app = _resolve_target_app_payload(request)
    if target_app is None:
        return {}, ["missing target app payload for QoS optimization"]

    if "slice_assignment" not in problem_config.decision_variables:
        return {}, ["QoS optimization disabled by problem configuration"]

    legacy_weights = objective_profile.to_legacy_qos_weights()
    result = optimize_network_slices(
        target_app,
        legacy_weights["w1"],
        legacy_weights["w2"],
        legacy_weights["w3"],
        legacy_weights["w4"],
        mode=problem_config.solver_mode,
        return_json=True,
        am_policy_state=am_policy_state,
        mobility_risk_weight=objective_profile.mobility_risk_cost,
        debug_context={
            "session_id": request.session_id,
            "snapshot_id": request.snapshot_id,
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


def _collect_unassigned_requested_flow_errors(qos_plan: Dict[str, Any]) -> List[str]:
    if not isinstance(qos_plan, dict):
        return []
    meta = qos_plan.get("meta") if isinstance(qos_plan.get("meta"), dict) else {}
    details = meta.get("infeasibility_details") if isinstance(meta, dict) else []
    if not isinstance(details, list):
        return []

    errors: List[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        flow_id = str(item.get("flow_id") or "").strip()
        app_name = str(item.get("app_name") or "").strip()
        candidate_slices = item.get("candidate_slices") if isinstance(item.get("candidate_slices"), list) else []
        slice_summaries: List[str] = []
        for candidate in candidate_slices:
            if not isinstance(candidate, dict):
                continue
            labels: List[str] = []
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
                slice_summaries.append(f"{candidate.get('snssai')}: " + ", ".join(labels))
        summary = "; ".join(slice_summaries) if slice_summaries else str(item.get("summary") or "no feasible slice assignment")
        errors.append(
            f"target flow {flow_id or '<unknown>'} in app {app_name or '<unknown>'} has no grounded slice assignment: {summary}"
        )
    return errors


def _build_am_policy_state_from_request(request: JointOptimizationRequest) -> Optional[AMPolicyState]:
    """关键步骤：从 JointOptimizationRequest 中提取当前 AM 策略状态用于 MILP 联合优化。"""
    policy_state = request.policy_state or {}
    target_supi = str((request.target_ues or [""])[0] or "").strip()

    # policy_state 结构为 {supi: ue_context}，需要先取出 UE 级别的数据
    ue_ctx = policy_state.get(target_supi) or policy_state if target_supi else policy_state
    am_ctx = (
        ue_ctx.get("amPolicyContext")
        or ue_ctx.get("am_policy")
        or ue_ctx.get("accessMobilityContext")
        or {}
    )
    if not am_ctx:
        return None

    mobility_summary = (
        ue_ctx.get("mobilitySummary")
        or ue_ctx.get("mobility_summary")
        or {}
    )
    raw_risk = mobility_summary.get("mobilityRiskScore")
    if raw_risk is None:
        raw_risk = mobility_summary.get("mobility_risk_score")
    try:
        mobility_risk_score = max(0.0, min(1.0, float(raw_risk or 0.0)))
    except (TypeError, ValueError):
        mobility_risk_score = 0.0

    return AMPolicyState(
        old_allowed_snssais=_normalize_snssai_refs(am_ctx.get("allowed_snssais") or am_ctx.get("allowedSnssais") or []),
        old_target_snssais=_normalize_snssai_refs(am_ctx.get("target_snssais") or am_ctx.get("targetSnssais") or []),
        old_rfsp=am_ctx.get("rfsp") or am_ctx.get("rfspIndex") or 1,
        old_triggers=am_ctx.get("triggers") or am_ctx.get("policyAssociationRequest", {}).get("triggers") or [],
        old_ue_ambr_ul=am_ctx.get("ue_ambr_ul") or 0.0,
        old_ue_ambr_dl=am_ctx.get("ue_ambr_dl") or 0.0,
        mobility_risk_score=mobility_risk_score,
    )


def run_joint_control_optimizer(request: JointOptimizationRequest) -> JointOptimizationResult:
    requested_domains = request.requested_domains or [ControlDomain.QOS]
    target_supi = str((request.target_ues or [""])[0] or "").strip()
    infeasible_reasons: List[str] = []
    qos_plan: Dict[str, Any] = {}
    mobility_plan: Dict[str, Any] = {}
    am_plan: Dict[str, Any] = {}
    verdicts: List[DomainVerdict] = []
    problem_config, objective_profile = resolve_problem_config(request)

    # 关键步骤：当请求包含 MOBILITY 域时，尝试构建 AM 旧状态以启用联合优化
    am_policy_state: Optional[AMPolicyState] = None
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
                    infeasible_reasons=qos_errors,
                )
            )
        else:
            verdicts.append(
                DomainVerdict(
                    domain=ControlDomain.QOS,
                    status=DomainStatus.APPROVED,
                    summary="QoS optimizer produced a feasible plan",
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
                    infeasible_reasons=["missing target SUPI"],
                )
            )
        else:
            snapshot = build_mobility_snapshot(request, target_supi)
            if snapshot.missing_fields:
                infeasible_reasons.extend(f"missing mobility context field: {item}" for item in snapshot.missing_fields)
                verdicts.append(
                    DomainVerdict(
                        domain=ControlDomain.MOBILITY,
                        status=DomainStatus.INCOMPLETE_CONTEXT,
                        summary="Mobility context is incomplete",
                        infeasible_reasons=[f"missing mobility context field: {item}" for item in snapshot.missing_fields],
                    )
                )
            else:
                draft = build_mobility_draft(request, target_supi, snapshot, qos_plan, am_plan=am_plan)
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
            "legacy_qos_weights": objective_profile.to_legacy_qos_weights(),
            "mobility_risk_cost": objective_profile.mobility_risk_cost,
            "session_cost": (qos_plan.get("meta", {}).get("breakdown") or {}).get("session_cost", 0.0),
            "mobility_cost": (qos_plan.get("meta", {}).get("breakdown") or {}).get("mobility_cost", 0.0),
            "coupling_cost": (qos_plan.get("meta", {}).get("breakdown") or {}).get("coupling_cost", 0.0),
        },
        infeasible_reasons=infeasible_reasons,
    )
