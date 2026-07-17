from __future__ import annotations

import json
from typing import Any, Dict

from .engine import PromptEngine
from .knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL


OSA_DYNAMIC_RULES = """
Dynamic planning rules for this round:
- Treat `retry_scope`, `revision_requests`, and `unified_constraints` from the user prompt as binding guidance.
- Prefer the smallest executable policy set that satisfies the current round objective.
- For soft QoS tuning, use best-effort optimizer evidence before declaring the plan blocked.
- If runtime evidence is insufficient or a hard optimizer preview is infeasible, return partial_plan or needs_upstream_reground instead of guessing.
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


_BLOCKED_MIGRATION_EXAMPLE = """
Example — IEA-blocked migration, entitlement-limited best-effort delivery:
{
  "planning_status": "partial_plan",
  "rationale": "Read-only entitlement evidence blocks the requested target migration; the optimizer preserved the catalog-confirmed entitled serving slice and produced the best available QoS policy.",
  "missing_evidence": ["authorized target S-NSSAI"],
  "blocked_targets": ["flow-9649 target slice migration"],
  "upstream_requests": [],
  "planner_conflicts": ["Requested slice change is blocked by subscription entitlement; current-slice QoS tuning is being delivered."],
  "sm_policies": [
    {
      "flow_id": "flow-9649",
      "app_id": "app-remote-drive",
      "priority": 3,
      "target_latency_ms": 10.0,
      "packet_error_rate": 0.001,
      "max_br_ul_mbps": 20.0,
      "max_br_dl_mbps": 80.0,
      "gbr_ul_mbps": 10.0,
      "gbr_dl_mbps": 40.0,
      "target_jitter_ms": 5.0
    }
  ],
  "am_policy": null,
  "ursp_policies": [],
  "partial_policies": []
}"""


_UNENTITLED_PREVIEW_BLOCK_EXAMPLE = """
Example — blocked migration and optimizer selected an unentitled target:
{
  "planning_status": "partial_plan",
  "rationale": "IEA forbids target-slice migration and the only optimizer preview selects an unentitled target, so no executable policy is emitted.",
  "missing_evidence": [],
  "blocked_targets": ["flow-9649 target slice migration"],
  "upstream_requests": [],
  "planner_conflicts": ["Optimizer did not provide a current-slice assignment allowed by the subscription evidence."],
  "sm_policies": [],
  "am_policy": null,
  "ursp_policies": [],
  "partial_policies": [{"policy_type": "SmPolicyDecision", "policy_id": "partial-sm-flow-9649", "flow_id": "flow-9649", "app_id": "app-remote-drive", "target_type": "flow", "blocked_reason": "Selected target slice is not entitled."}]
}"""


OSA_OUTPUT_FORMAT_RULES = """
Output contract (violations are rejected):
- Return raw JSON: exactly one OsaAdvisorOutput object; never markdown or `planning_metadata`. Never add top-level keys outside the OsaAdvisorOutput schema.
- Required root fields: planning_status, rationale, missing_evidence, blocked_targets, upstream_requests, planner_conflicts, sm_policies, am_policy, ursp_policies, partial_policies.
- `rationale` MUST be a string. Every collection field (`sm_policies`, `ursp_policies`, `partial_policies`, blocker lists) is `[]` when empty, never `null`; `am_policy` is `null` only when mobility is inactive. Never emit `{}` as an optional policy value.
- `missing_evidence`, `blocked_targets`, `upstream_requests`, and `planner_conflicts` are `List[str]`: every item is one concise string, never an object with reason/request metadata.
- Never return a bare policy object like `{\"flow_id\":...}` outside the sm_policies array.
- Every SmPolicySpec has an exact GroundingDecision flow_id/app_id, priority 1-15, target_latency_ms >= 1.0, packet_error_rate 0-1, max_br_ul_mbps, and max_br_dl_mbps. Optional GBR cannot exceed maxBR.
- Optimizer output is evidence, not an SmPolicySpec. Never copy `new_slice`, `current_slice`, `slice_snssai`, `jitter_ms`, or other allocation metadata into sm_policies. The compiler retains the optimizer assignment and generates the executable policy_id; `policy_id` belongs only in partial_policies. When jitter is needed in an SM policy, use the exact key `target_jitter_ms`.
- Every AmPolicySpec has ONLY triggers, rfsp, allowed_snssais, target_snssais, optional ue_ambr_ul_mbps, optional ue_ambr_dl_mbps, and optional serv_area_res. SUPI/session/policy IDs are code-owned and must not appear inside am_policy. It has non-empty triggers, rfsp >= 1, non-empty allowed_snssais, and target_snssais contained in allowed_snssais.
- Every UrspPolicySpec uses target_type `flow` or `app`, app_id, relat_precedence >= 1, and non-empty route_sel_param_sets; a flow target also needs flow_id and traffic_desc.
- Partial policy items MUST include: policy_type, policy_id, and grounded flow_id/app_id/target_type plus blocked_reason when available.

Return raw JSON only, no markdown fence, no prose outside the JSON object.
"""


def render_round_tool_policy(available_tool_names: list[str] | None) -> str:
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
    elif "am_policy.supi" in joined or ("ursp_policies" in joined and "valid list" in joined):
        correction = (
            "Repair the exact OSA schema shape. am_policy must not contain supi, session_id, snapshot_id, policy_id, or policy_type; "
            "those are code-owned. ursp_policies is [] when unused, never null. "
            "Keep every required top-level field and return one OsaAdvisorOutput object.\n\n"
            + _MOBILITY_EXAMPLE
        )
    elif "extra inputs are not permitted" in joined:
        correction = (
            "Your previous answer included fields not in the OsaAdvisorOutput schema. "
            "Remove every unsupported top-level field. "
            "Unsupported fields may be nested, not only top-level. "
            "Inside sm_policies, remove policy_id, policy_type, target_type, policy_details, slice_snssai, new_slice, jitter_ms, and any other field outside SmPolicySpec. "
            "SmPolicySpec fields are only: flow_id, app_id, priority, target_latency_ms, packet_error_rate, "
            "max_br_ul_mbps, max_br_dl_mbps, optional gbr_ul_mbps, gbr_dl_mbps, target_jitter_ms, flow_description. "
            "Only emit the defined OsaAdvisorOutput fields.\n\n"
            + _QOS_EXAMPLE
        )
    elif (
        "target slice migration" in joined
        or "entitlement-limited qos delivery" in joined
        or "blocked slice migration may deliver" in joined
        or "blocked_targets.0" in joined
        or "upstream_requests.0" in joined
    ):
        correction = (
            "IEA has blocked only the target S-NSSAI migration, not the QoS objective. "
            "If the optimizer preview preserves the current authorized slice, return planning_status=\"partial_plan\" and deliver that optimizer-backed SM policy. "
            "If the preview selects an unentitled slice, do not emit any SM policy from it: return partial_plan with one non-executable partial_policies descriptor instead. "
            "If the preview is Infeasible or has no grounded assignment, likewise emit no SM policy and return only that non-executable partial descriptor. "
            "Do not emit AM or URSP policies, because they could bypass the subscription boundary. "
            "For blocked_by_subscription_entitlement, do not request subscription provisioning. Keep blocker arrays as plain strings, never objects.\n\n"
            + _BLOCKED_MIGRATION_EXAMPLE
            + "\n\n"
            + _UNENTITLED_PREVIEW_BLOCK_EXAMPLE
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
    elif "target_latency_ms" in joined and (">= 1.0" in joined or "greater than or equal to 1" in joined or "sub-1ms" in joined):
        correction = (
            "Your previous SM policy used an invalid latency target. "
            "Never emit sub-1ms executable SM policy targets: target_latency_ms must be >= 1.0. "
            "Do not keep tightening latency across retries below the runtime floor or below optimizer evidence. "
            "If the requested hard latency is below 1ms, or the optimizer cannot support a target at or above 1ms, "
            "return planning_status=\"partial_plan\" with blocked_targets and planner_conflicts. "
            "Only return executable_plan when the optimizer/runtime evidence supports target_latency_ms >= 1.0.\n\n"
            + _INFEASIBLE_EXAMPLE
        )
    elif "exceeded max_calls_per_tool" in joined or "get_knowledge_by_key exceeded" in joined:
        correction = (
            "You burned your tool call quota — most likely on knowledge tools that cannot fix the issue. "
            "Knowledge tools (get_knowledge_by_key, search_semantic_knowledge) will NOT resolve "
            "contract validation errors about flow_id, app_id, optimizer assignments, local slice labels, or target-stable preservation. "
            "Those errors are caused by your output not matching the optimizer evidence — not by missing 3GPP knowledge. "
            "Fix: align your sm_policies exactly with the optimizer preview. "
            "Never add a flow_id that is absent from GroundingDecision.flows or from the optimizer QoS assignment. "
            "On target-stable retries, preserve the exact flow_ids and app_ids from the upstream request. "
            "If the optimizer preview is incomplete, return partial_plan with planner_conflicts.\n\n"
            + _QOS_EXAMPLE
        )
    elif "outside operationintent flows" in joined:
        correction = (
            "Your previous SM policy drifted to a flow outside GroundingDecision.flows. "
            "You must preserve the exact flow_id/app_id from GroundingDecision.flows. "
            "Do not substitute a semantically similar optimizer flow. "
            "If the optimizer preview does not contain an assignment for the GroundingDecision flow, return partial_plan with blocked_targets and planner_conflicts.\n\n"
            + _INFEASIBLE_EXAMPLE
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
    "OSA_OUTPUT_FORMAT_RULES",
    "OSA_SYSTEM_PROMPT",
    "build_validation_retry_prompt",
    "render_round_tool_policy",
]

