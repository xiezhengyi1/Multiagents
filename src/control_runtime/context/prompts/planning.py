from __future__ import annotations

import json
from typing import Any

from .engine import PromptEngine
from .knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL
from .retry import strip_retry_feedback


def render_planning_system_prompt() -> str:
    return PromptEngine().render(
        "planning/system.j2",
        osa_knowledge_search_skill=OSA_KNOWLEDGE_SEARCH_SKILL,
    )


def build_planning_user_prompt(
    *,
    normalized_user_intent: dict[str, Any],
    coordination_context: dict[str, Any],
    planning_evidence: dict[str, Any],
    available_tool_names: list[str],
) -> str:
    return PromptEngine().render(
        "planning/user.j2",
        normalized_user_intent=normalized_user_intent,
        coordination_context=coordination_context,
        planning_evidence=planning_evidence,
        tool_policy=_render_tool_manifest(available_tool_names),
    )


def _render_tool_manifest(available_tool_names: list[str]) -> str:
    lines = ["Callable tools in this round:"]
    if available_tool_names:
        lines.extend(f"- `{name}`" for name in available_tool_names)
    else:
        lines.append("- <none>")
    return "\n".join(lines)


def build_planning_validation_retry_prompt(
    *,
    base_prompt: str,
    issues: list[str],
    cached_planning_evidence: dict[str, Any] | None = None,
) -> str:
    """Build OSA repair feedback; shared by explicit and single-agent planning."""
    joined = " | ".join(issues).lower()
    if "bare array" in joined or "non-object" in joined:
        correction = (
            "Return one OsaAdvisorOutput JSON object, never a bare policy array or item. "
            "Include planning_status, rationale, evidence arrays, sm_policies, am_policy, ursp_policies, and partial_policies."
        )
    elif "serv_area_res is runtime-owned" in joined or "servareares" in joined:
        correction = (
            "Omit am_policy.serv_area_res entirely. The compiler preserves validated mobility context; "
            "am_policy may contain only its advisor-owned triggers, rfsp, S-NSSAI, and UE-AMBR fields."
        )
    elif (
        "authorized slice migration requires main to activate mobility" in joined
        or "migration without mobility-active planning" in joined
        or "am_policy outside mobility-active planning" in joined
    ):
        correction = (
            "Do not repair a serving-slice migration in a QoS-only Main stage. "
            "Return needs_upstream_reground with no executable policies and request QoS plus mobility activation."
        )
    elif "authorized slice migration requires an am policy" in joined or "mobility-active planning requires am_policy" in joined:
        correction = (
            "For an authorized migration with QoS and mobility active, emit grounded SM and AM policies; "
            "otherwise return needs_upstream_reground."
        )
    elif "extra inputs are not permitted" in joined:
        correction = (
            "Emit only OsaAdvisorOutput fields. SmPolicySpec accepts flow_id, app_id, priority, latency, error-rate, "
            "bandwidth, optional GBR/jitter, and flow_description; code owns PCF policy IDs and details."
        )
    elif (
        "target slice migration" in joined
        or "entitlement-limited qos delivery" in joined
        or "optimizer-selected slice is absent from subscription entitlement evidence" in joined
        or "unauthorized slice migration" in joined
    ):
        correction = (
            "IEA blocks target migration, not all QoS. If the optimizer preserves an entitled serving slice, return partial_plan "
            "with only its SM policy. Never emit AM or URSP; if no entitled optimizer assignment exists, emit only a partial descriptor."
        )
    elif "gbr_" in joined and "max_br" in joined:
        correction = "Keep GBR at or below MBR. If optimizer evidence cannot support it, return partial_plan with conflicts."
    elif "target_latency_ms" in joined and (">= 1.0" in joined or "sub-1ms" in joined):
        correction = "target_latency_ms must be at least 1.0; return partial_plan when the runtime floor is infeasible."
    elif "exceeded max_calls_per_tool" in joined or "get_knowledge_by_key exceeded" in joined:
        correction = (
            "Do not use knowledge tools to repair optimizer binding errors. Reuse visible optimizer evidence and preserve its flow_id/app_id pairs."
        )
    elif "outside groundingdecision flows" in joined:
        correction = "Preserve exact GroundingDecision flow_id/app_id pairs; do not substitute a semantically similar flow."
    elif "inspect_mobility_ue_policies" in joined or "not callable" in joined or "unknown tool" in joined:
        correction = "Use only tools callable in this stage. A QoS-only round must not call mobility tools or emit AM policy."
    elif "did not converge" in joined or "max iterations" in joined:
        correction = "Stop calling tools once evidence is sufficient; return partial_plan or needs_upstream_reground when it is not."
    elif "infeasible" in joined or "incomplete_context" in joined:
        correction = "Optimizer evidence is infeasible or incomplete: do not return executable_plan."
    else:
        correction = "Fix every validation issue below and return a corrected OsaAdvisorOutput:\n- " + "\n- ".join(issues)

    cached_block = ""
    if cached_planning_evidence:
        cached_block = (
            "\n\nCached tool evidence from the failed attempt:\n"
            f"{json.dumps(cached_planning_evidence, ensure_ascii=False, default=str)}\n\n"
            "Reuse this evidence before calling another tool; knowledge retrieval does not repair contract mismatches."
        )
    return (
        f"{strip_retry_feedback(base_prompt)}\n\n"
        f"Retry feedback:\n{correction}"
        f"{cached_block}"
    )
