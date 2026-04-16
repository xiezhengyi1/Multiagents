from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from domain.control_plane import (
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


def _resolve_target_app_payload(request: JointOptimizationRequest) -> Optional[Dict[str, Any]]:
    operation_intent = request.operation_intent or {}
    traffic_state = request.traffic_state or {}
    if isinstance(operation_intent, dict) and operation_intent.get("flows"):
        return {
            "app_id": operation_intent.get("app_id"),
            "name": operation_intent.get("app_name") or operation_intent.get("app_name"),
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
        if target_app_id and str(app.get("app_id") or "").strip() == target_app_id:
            return app
        if target_supi and str(app.get("supi") or "").strip() == target_supi:
            return app
    return apps[0] if apps else None


def _run_qos_subproblem(
    request: JointOptimizationRequest,
    *,
    problem_config: OptimizationProblemConfig,
    objective_profile: ObjectiveProfile,
    am_policy_state: Optional[AMPolicyState] = None,
) -> Tuple[Dict[str, Any], List[str]]:
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
    )
    if not isinstance(result, dict):
        return {}, ["QoS optimizer returned non-dict result"]
    if result.get("error"):
        return {}, [str(result["error"])]
    return result, []


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

    return AMPolicyState(
        old_allowed_snssais=am_ctx.get("allowed_snssais") or am_ctx.get("allowedSnssais") or [],
        old_target_snssais=am_ctx.get("target_snssais") or am_ctx.get("targetSnssais") or [],
        old_rfsp=am_ctx.get("rfsp") or am_ctx.get("rfspIndex") or 1,
        old_triggers=am_ctx.get("triggers") or am_ctx.get("policyAssociationRequest", {}).get("triggers") or [],
        old_ue_ambr_ul=am_ctx.get("ue_ambr_ul") or 0.0,
        old_ue_ambr_dl=am_ctx.get("ue_ambr_dl") or 0.0,
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
            "legacy_qos_weights": objective_profile.to_legacy_qos_weights(),
            "mobility_risk_cost": objective_profile.mobility_risk_cost,
        },
        infeasible_reasons=infeasible_reasons,
    )


def run_joint_control_optimizer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    request = JointOptimizationRequest.model_validate(payload)
    result = run_joint_control_optimizer(request)
    return result.model_dump(mode="json")


def run_joint_control_optimizer_json(payload_json: str) -> str:
    payload = json.loads(payload_json)
    return json.dumps(run_joint_control_optimizer_payload(payload), ensure_ascii=False)

