from __future__ import annotations

import json
from typing import Any, Dict

from .engine import PromptEngine
from .knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL


OSA_DYNAMIC_RULES = """
Dynamic planning rules for this round:
- Treat `main_retry_scope`, `revision_requests`, and `unified_constraints` from the user prompt as binding guidance.
- Prefer the smallest executable policy set that satisfies the current round objective.
- If optimizer/runtime evidence is insufficient or infeasible, return partial_plan or needs_upstream_reground instead of guessing.
"""


def _render_system_prompt() -> str:
    return PromptEngine().render(
        "planning/system.j2",
        osa_knowledge_search_skill=OSA_KNOWLEDGE_SEARCH_SKILL,
    )


OSA_SYSTEM_PROMPT = _render_system_prompt()
OSA_CORE_PROMPT = OSA_SYSTEM_PROMPT


_QOS_EXAMPLE = """
Example — QoS-only executable plan:
{
  "planning_status": "executable_plan",
  "rationale": "Optimizer preview feasible, qos_flow_assignment confirmed for remote_drive_video_1 on slice-1-000001 with latency 10ms.",
  "missing_evidence": [],
  "blocked_targets": [],
  "upstream_requests": [],
  "planner_conflicts": [],
  "sm_policies": [
    {
      "flow_id": "remote_drive_video_1",
      "app_id": "Remote_Drive",
      "priority": 1,
      "target_latency_ms": 10.0,
      "packet_error_rate": 0.00001,
      "max_br_ul_mbps": 50.0,
      "max_br_dl_mbps": 100.0,
      "gbr_ul_mbps": 25.0,
      "gbr_dl_mbps": 50.0,
      "target_jitter_ms": 5.0,
      "flow_description": "URLLC video stream for remote driving control"
    }
  ],
  "am_policy": null,
  "ursp_policies": [],
  "partial_policies": []
}"""


_MOBILITY_EXAMPLE = """
Example — Mobility-only executable plan:
{
  "planning_status": "executable_plan",
  "rationale": "UE current RFSP 1, target 2 provides higher priority; allowed_snssais validated against optimizer coupling output.",
  "missing_evidence": [],
  "blocked_targets": [],
  "upstream_requests": [],
  "planner_conflicts": [],
  "sm_policies": [],
  "am_policy": {
    "triggers": ["RFSP_CH"],
    "rfsp": 2,
    "allowed_snssais": [
      {"sst": 1, "sd": "000001"},
      {"sst": 1, "sd": "000002"}
    ],
    "target_snssais": [
      {"sst": 1, "sd": "000002"}
    ],
    "ue_ambr_ul_mbps": 100.0,
    "ue_ambr_dl_mbps": 200.0
  },
  "ursp_policies": [],
  "partial_policies": []
}"""


_INFEASIBLE_EXAMPLE = """
Example — Infeasible optimizer, must NOT return executable_plan:
{
  "planning_status": "partial_plan",
  "rationale": "Optimizer returned infeasible: slice-1-000001 has no remaining capacity for the target latency requirement.",
  "missing_evidence": [],
  "blocked_targets": ["remote_drive_video_1"],
  "upstream_requests": ["Request capacity reassessment on slice-1-000001 or alternative slice suggestion"],
  "planner_conflicts": ["Target latency 10ms cannot be satisfied on current slices under existing load"],
  "sm_policies": [],
  "am_policy": null,
  "ursp_policies": [],
  "partial_policies": [
    {
      "policy_type": "SmPolicyDecision",
      "policy_id": "partial-sm-remote_drive_video_1",
      "flow_id": "remote_drive_video_1",
      "app_id": "Remote_Drive",
      "target_type": "flow",
      "blocked_reason": "No feasible slice for target latency 10ms"
    }
  ]
}"""


_OUTPUT_FORMAT_RULES = """
Output format (CRITICAL — violations will be rejected):
- Top-level MUST be one JSON object with planning_status, rationale, sm_policies, am_policy, ursp_policies, and partial_policies.
- `rationale` MUST be a string, never an object or array. Put structured details in missing_evidence, blocked_targets, upstream_requests, or planner_conflicts.
- `sm_policies` and `ursp_policies` MUST be JSON arrays. Use `[]` when empty, never `null`.
- `am_policy` may be `null` when mobility is not requested; otherwise it must be an object. Never emit empty `am_policy: {}`.
- Never return a bare array `[...]` as the top-level value.
- Never return a bare policy object like `{\"flow_id\":...}` outside the sm_policies array.
- Never add top-level keys outside the OsaAdvisorOutput schema.
- Never emit `planning_metadata`.
- Omit optional sections or set them to `null`/`[]`; never emit empty objects like `{}` for optional policy fields.

SmPolicySpec fields (flow-scoped, inside sm_policies array):
  Required: flow_id, app_id, priority (1-15), target_latency_ms (>=1.0), packet_error_rate (0.0-1.0), max_br_ul_mbps (>=0), max_br_dl_mbps (>=0)
  Optional: gbr_ul_mbps, gbr_dl_mbps, target_jitter_ms, flow_description

AmPolicySpec fields (inside am_policy):
  Required: triggers (non-empty array), rfsp (>=1), allowed_snssais (non-empty array), target_snssais (non-empty array, subset of allowed_snssais)
  Optional: ue_ambr_ul_mbps, ue_ambr_dl_mbps, serv_area_res

Partial policy items (inside partial_policies array):
  Partial policy items MUST include: policy_type, policy_id.
  For QoS partials, use policy_type="SmPolicyDecision" and a stable policy_id such as "partial-sm-<flow_id>".
  Include flow_id, app_id, target_type, blocked_reason, and any grounded policy_details when available.

Knowledge tools:
- Use `get_knowledge_by_key` for exact 3GPP object names (e.g. \"29.512:QosData\").
- Use `search_semantic_knowledge` only when a required output field still cannot be grounded from runtime evidence.
- Do not query knowledge tools for local schema names like OsaAdvisorOutput, SmPolicySpec, AmPolicySpec, local slice labels, or optimizer-selected S-NSSAI values.

Return raw JSON only, no markdown fence, no prose outside the JSON object.
"""


def build_advisor_user_prompt(
    *,
    normalized_user_intent: Dict[str, Any],
    coordination_context: Dict[str, Any],
    planning_evidence: Dict[str, Any],
    available_tool_names: list[str] | None = None,
) -> str:
    return PromptEngine().render(
        "planning/user.j2",
        normalized_user_intent=normalized_user_intent,
        coordination_context=coordination_context,
        planning_evidence=planning_evidence,
        tool_policy=_render_round_tool_policy(available_tool_names),
        dynamic_rules=OSA_DYNAMIC_RULES.strip(),
        output_format_rules=_OUTPUT_FORMAT_RULES.strip(),
    )


def _render_round_tool_policy(available_tool_names: list[str] | None) -> str:
    if available_tool_names is None:
        return "Callable tools in this round: use the tools registered by the runtime for this request."
    available = [
        str(name or "").strip()
        for name in available_tool_names
        if str(name or "").strip()
    ]
    available = list(dict.fromkeys(available))
    available_set = set(available)
    lines = ["Callable tools in this round:"]
    if available:
        lines.extend(f"- `{name}`" for name in available)
    else:
        lines.append("- <none>")
    request_scoped_tools = [
        "preview_qos_optimizer",
        "fetch_qos_network_status",
        "inspect_mobility_ue_policies",
    ]
    blocked = [name for name in request_scoped_tools if name not in available_set]
    if blocked:
        lines.append("Tools not callable in this round:")
        lines.extend(f"- {name} is not callable in this round." for name in blocked)
    return "\n".join(lines)


def build_validation_retry_prompt(
    *,
    base_prompt: str,
    issues: list[str],
    cached_planning_evidence: Dict[str, Any] | None = None,
) -> str:
    import re

    cleaned = re.sub(
        r'\n\nRetry feedback \(attempt \d+\).*$',
        '',
        base_prompt,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r'\n\nYour previous attempt failed validation.*$',
        '',
        cleaned,
        flags=re.DOTALL,
    )

    joined = " | ".join(issues).lower()

    if "bare array" in joined or "non-object" in joined:
        correction = (
            "Your previous answer used the wrong top-level shape. "
            "The top-level JSON must be an OsaAdvisorOutput object (with planning_status, sm_policies array, am_policy, etc.), "
            "never a bare array or a bare policy item.\n\n"
            + _QOS_EXAMPLE
        )
    elif "extra inputs are not permitted" in joined:
        correction = (
            "Your previous answer included fields not in the OsaAdvisorOutput schema. "
            "Remove every unsupported top-level field. "
            "Unsupported fields may be nested, not only top-level. "
            "Inside sm_policies, remove policy_id, policy_type, target_type, policy_details, and any other field outside SmPolicySpec. "
            "SmPolicySpec fields are only: flow_id, app_id, priority, target_latency_ms, packet_error_rate, "
            "max_br_ul_mbps, max_br_dl_mbps, optional gbr_ul_mbps, gbr_dl_mbps, target_jitter_ms, flow_description. "
            "Only emit the defined OsaAdvisorOutput fields.\n\n"
            + _QOS_EXAMPLE
        )
    elif "partial_policies" in joined or "rationale" in joined or "valid string" in joined:
        correction = (
            "Your previous output had an OsaAdvisorOutput schema error. "
            "The top-level rationale must be a JSON string, not an object or array. "
            "Each partial_policies item must include policy_type and policy_id; "
            "for QoS partials use policy_type=\"SmPolicyDecision\" and policy_id=\"partial-sm-<flow_id>\".\n\n"
            + _INFEASIBLE_EXAMPLE
        )
    elif ("gbr_" in joined and "max_br" in joined) or "must not exceed" in joined:
        correction = (
            "Your previous SM policy used invalid QoS bitrate bounds. "
            "For every SmPolicySpec, gbr_ul_mbps must be <= max_br_ul_mbps and "
            "gbr_dl_mbps must be <= max_br_dl_mbps. "
            "Use optimizer/evidence values only; if the optimizer cannot support the requested GBR under the MBR, "
            "return partial_plan with planner_conflicts instead of an executable_plan.\n\n"
            + _QOS_EXAMPLE
        )
    elif "exceeded max_calls_per_tool" in joined or "get_knowledge_by_key exceeded" in joined:
        correction = (
            "You burned your tool call quota — most likely on knowledge tools that cannot fix the issue. "
            "Knowledge tools (get_knowledge_by_key, search_semantic_knowledge) will NOT resolve "
            "contract validation errors about flow_id, app_id, optimizer assignments, local slice labels, or target-stable preservation. "
            "Those errors are caused by your output not matching the optimizer evidence — not by missing 3GPP knowledge. "
            "Fix: align your sm_policies exactly with the optimizer preview. "
            "Never add a flow_id that is absent from the optimizer QoS assignment. "
            "On target-stable retries, preserve the exact flow_ids and app_ids from the upstream request. "
            "If the optimizer preview is incomplete, return partial_plan with planner_conflicts.\n\n"
            + _QOS_EXAMPLE
        )
    elif "inspect_mobility_ue_policies" in joined or "not callable" in joined or "unknown tool" in joined:
        correction = (
            "You tried to use a tool that is not callable for this QoS-only round. "
            "For slice migration wording without explicit AM/RFSP/allowed-NSSAI/service-area/handover requirements, "
            "do not call inspect_mobility_ue_policies and do not output am_policy. "
            "Use the cached optimizer/network evidence if present; otherwise call only callable QoS tools. "
            "Never ask mobility or knowledge tools to validate local S-NSSAI labels.\n\n"
            + _QOS_EXAMPLE
        )
    elif "did not converge" in joined or "max iterations" in joined:
        correction = (
            "You ran out of iterations without returning a final JSON. "
            "Only the latest visible result for each tool may be retained in context; do not call tools again to recover earlier evidence. "
            "Stop calling tools once you have enough visible evidence. "
            "If evidence is incomplete, return partial_plan or needs_upstream_reground instead of continuing to call tools.\n\n"
            + _INFEASIBLE_EXAMPLE
        )
    elif "infeasible" in joined:
        correction = (
            "The optimizer preview was infeasible or incomplete. "
            "Do NOT return planning_status=\"executable_plan\". "
            "Return partial_plan or needs_upstream_reground and explain the blocking reason.\n\n"
            + _INFEASIBLE_EXAMPLE
        )
    else:
        contract_lines = "\n- ".join(issues)
        correction = (
            "Your previous output failed contract validation:\n"
            f"- {contract_lines}\n\n"
            "Fix each issue and return a corrected OsaAdvisorOutput.\n\n"
            "Important: knowledge tools (get_knowledge_by_key, search_semantic_knowledge) will NOT fix "
            "contract validation errors — those errors are about your output not matching the optimizer/runtime evidence, "
            "not about missing 3GPP facts. Do not burn knowledge tool quota on this retry. "
            "Use only callable domain tools if you need fresh evidence; for QoS-only slice migration that means "
            "preview_qos_optimizer and, optionally, fetch_qos_network_status.\n\n"
            + _QOS_EXAMPLE
        )

    cached_block = ""
    if cached_planning_evidence:
        cached_block = (
            "\n\nCached tool evidence from the failed attempt:\n"
            f"{json.dumps(cached_planning_evidence, ensure_ascii=False, default=str)}\n\n"
            "This evidence already contains optimizer results and network status. "
            "Do NOT call any tool again on this retry unless a required field is genuinely absent from the cached evidence. "
            "Knowledge tools (get_knowledge_by_key, search_semantic_knowledge) will NOT help with contract validation errors. "
            "First try to return the final OsaAdvisorOutput from this cached evidence directly."
        )

    return (
        f"{cleaned}\n\n"
        f"Retry feedback:\n{correction}"
        f"{cached_block}"
    )


__all__ = [
    "OSA_CORE_PROMPT",
    "OSA_DYNAMIC_RULES",
    "OSA_SYSTEM_PROMPT",
    "build_advisor_user_prompt",
    "build_validation_retry_prompt",
]

