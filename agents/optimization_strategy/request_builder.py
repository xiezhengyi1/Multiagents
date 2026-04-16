from __future__ import annotations

from agents.tools.db_tool import get_latest_snapshot_data, get_ue_context_by_supi
from domain.collaboration import PlanningRequest
from domain.control_plane import (
    ControlDomain,
    JointOptimizationRequest,
    ObjectiveProfile,
    OptimizationProblemConfig,
    OptimizationTemplate,
)


def build_revision_prompt_fragment(planning_request: PlanningRequest) -> str:
    revision_requests = list(planning_request.context.revision_requests or [])
    unified_constraints = dict(planning_request.context.unified_constraints or {})
    if not revision_requests and not unified_constraints:
        return ""

    lines: list[str] = []
    if revision_requests:
        lines.append("Mediator revision requests:")
        for item in revision_requests:
            if not isinstance(item, dict):
                continue
            target_domain = str(item.get("target_domain") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if target_domain or reason:
                lines.append(f"- {target_domain}: {reason}")
            for constraint in item.get("hard_constraints") or []:
                text = str(constraint or "").strip()
                if text:
                    lines.append(f"  constraint: {text}")

    hard_constraints = [
        str(item).strip()
        for item in unified_constraints.get("hard_constraints") or []
        if str(item).strip()
    ]
    if hard_constraints:
        lines.append("Locked hard constraints:")
        for item in hard_constraints:
            lines.append(f"- {item}")

    return "\n".join(lines).strip()


def build_joint_optimizer_request(
    planning_request: PlanningRequest,
    *,
    profile_name: str | None = None,
    template_name: str | None = None,
) -> JointOptimizationRequest:
    operation_intent = planning_request.operation_intent
    snapshot = get_latest_snapshot_data() or {}
    target_supi = str(operation_intent.supi or "").strip()

    requested_domains: list[ControlDomain] = []
    for item in planning_request.context.active_domains or []:
        normalized = str(item or "").strip().lower()
        if normalized in {"both", "all"}:
            requested_domains = [ControlDomain.QOS, ControlDomain.MOBILITY]
            break
        requested_domains.append(ControlDomain(normalized))

    ue_context = get_ue_context_by_supi(target_supi) or {}
    revision_fragment = build_revision_prompt_fragment(planning_request)
    prompt_parts = [str(planning_request.context.main_agent_guidance or "").strip()]
    if revision_fragment:
        prompt_parts.append(revision_fragment)

    objective_profile_payload = dict(planning_request.context.objective_profile or {"profile_name": "balanced"})
    if profile_name is not None:
        objective_profile_payload["profile_name"] = str(profile_name or "").strip().lower() or "balanced"

    problem_config = OptimizationProblemConfig()
    if template_name is not None:
        normalized_template = str(template_name or "").strip().lower()
        if not normalized_template:
            raise ValueError("template_name must not be empty when provided")
        problem_config = OptimizationProblemConfig(template=OptimizationTemplate(normalized_template))

    return JointOptimizationRequest(
        session_id=planning_request.context.session_id,
        snapshot_id=planning_request.context.snapshot_id or str(snapshot.get("snapshot_id") or ""),
        target_ues=[target_supi] if target_supi else [],
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
        mobility_state={target_supi: ue_context.get("accessMobilityContext", {})} if target_supi else {},
        policy_state={target_supi: ue_context} if target_supi else {},
        objective_profile=ObjectiveProfile.model_validate(objective_profile_payload),
        problem_config=problem_config,
        prompt_injection="\n".join(part for part in prompt_parts if part).strip(),
    )


__all__ = ["build_joint_optimizer_request", "build_revision_prompt_fragment"]
