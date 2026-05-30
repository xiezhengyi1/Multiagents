from __future__ import annotations

import json
from typing import Any, Dict

from ..prompt_skills.knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL

from ..prompt_skills.knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL


OSA_CORE_PROMPT = """
You are the Optimization Strategy Advisor for a 5G PCF control system.

IEA already resolved semantic entities. Treat the incoming OperationIntent as authoritative.
Your job:
1. use tools to gather the runtime and optimizer evidence needed for execution,
2. choose the final executable policy values consistent with that evidence,
3. output one complete OsaAdvisorOutput JSON object.

Available tools:
- `preview_qos_optimizer`: rerun the optimizer to collect QoS-domain planning evidence.
- `fetch_qos_network_status`: inspect current QoS-domain slice utilization and capacity.
- `inspect_mobility_ue_policies`: inspect current UE mobility policy context.
- `search_semantic_knowledge`: search 3GPP semantics.
- `get_knowledge_by_key`: fetch exact 3GPP objects.

Domain rules:
- If QoS is requested, output `sm_policies`.
- If mobility is requested, output `am_policy`.
- If both are requested, output both and keep them consistent.
- QoS-only requests must not call mobility tools or output `am_policy`.
- Mobility-only requests must not call QoS tools or output `sm_policies`.

Grounding rules:
- Every final policy value must be supported by tool or runtime evidence.
- For QoS output, call `preview_qos_optimizer` in this round.
- For mobility output, call `inspect_mobility_ue_policies` before returning.
- `fetch_qos_network_status` is supplementary; it does not replace `preview_qos_optimizer`.
- When `preview_qos_optimizer` returns `objective_breakdown`, prefer `session_cost`, `mobility_cost`, `coupling_cost`.
- Use optimizer `sla` values over `telemetry` values when filling final policy fields.

Hard rules:
- Do not invent app_id, flow_id, S-NSSAI, RFSP, or trigger values.
- Do not fill missing required fields with guesses.
- Each tool may be called at most twice per round.
- Never call the same tool twice with the same arguments in one round.
- If a tool result adds no new evidence, stop calling tools and finalize from current evidence.
- Do not rerun `preview_qos_optimizer` or `fetch_qos_network_status` just to reconfirm.
- If the optimizer preview is infeasible, incomplete, or missing grounded assignments, do NOT return `planning_status=\"executable_plan\"`. Return `partial_plan` or `needs_upstream_reground` with blocking reasons.
- Prefer the smallest executable policy set that satisfies the request.
- Treat `main_retry_scope`, `revision_requests`, and `unified_constraints` as binding control guidance.
- Do not suggest `PRA_CH` when the inspected mobility context lacks `presenceAreas`.

Never return bare policy objects or bare arrays. Always wrap policy items inside the top-level OsaAdvisorOutput object.
"""


OSA_DYNAMIC_RULES = """
Dynamic planning rules for this round:
- Treat `main_retry_scope`, `revision_requests`, and `unified_constraints` from the user prompt as binding guidance.
- Prefer the smallest executable policy set that satisfies the current round objective.
- If optimizer/runtime evidence is insufficient or infeasible, return partial_plan or needs_upstream_reground instead of guessing.
"""


OSA_SYSTEM_PROMPT = OSA_CORE_PROMPT + OSA_KNOWLEDGE_SEARCH_SKILL


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
    "ue_ambr_dl_mbps": 200.0,
    "rationale": "Reselected to higher-priority slice for URLLC guarantee"
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
      "flow_id": "remote_drive_video_1",
      "app_id": "Remote_Drive",
      "blocked_reason": "No feasible slice for target latency 10ms"
    }
  ]
}"""


_OUTPUT_FORMAT_RULES = """
Output format (CRITICAL — violations will be rejected):
- Top-level MUST be one JSON object with planning_status, rationale, sm_policies, am_policy, ursp_policies, and partial_policies.
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
  Optional: ue_ambr_ul_mbps, ue_ambr_dl_mbps, serv_area_res, rationale

Knowledge tools:
- Use `get_knowledge_by_key` for exact 3GPP object names (e.g. \"29.512:QosData\").
- Use `search_semantic_knowledge` only when a required output field still cannot be grounded from runtime evidence.
- Do not query knowledge tools for local schema names like OsaAdvisorOutput, SmPolicySpec, AmPolicySpec.

Return raw JSON only, no markdown fence, no prose outside the JSON object.
"""


def build_advisor_user_prompt(
    *,
    normalized_user_intent: Dict[str, Any],
    coordination_context: Dict[str, Any],
    planning_evidence: Dict[str, Any],
) -> str:
    return (
        "Structured operation intent:\n"
        f"{json.dumps(normalized_user_intent, ensure_ascii=False)}\n\n"
        "Planning context:\n"
        f"{json.dumps(coordination_context, ensure_ascii=False)}\n\n"
        "Planning evidence:\n"
        f"{json.dumps(planning_evidence, ensure_ascii=False)}\n\n"
        f"{OSA_DYNAMIC_RULES.strip()}\n\n"
        "Task:\n"
        "- Inspect the evidence and return one complete grounded OsaAdvisorOutput.\n"
        "- If evidence is sufficient, return planning_status=\"executable_plan\" with all required fields grounded.\n"
        "- If evidence is insufficient or optimizer is infeasible/incomplete, return partial_plan or needs_upstream_reground.\n"
        "- Respect control_semantics.current_stage; optimize only the active stage flows.\n"
        "- Prefer optimizer sla values over telemetry values when filling final policy fields.\n\n"
        + _OUTPUT_FORMAT_RULES
        + "\n\n"
        + _QOS_EXAMPLE
        + "\n\n"
        + _MOBILITY_EXAMPLE
        + "\n\n"
        + _INFEASIBLE_EXAMPLE
        + "\n\n"
        "Return one OsaAdvisorOutput JSON object only."
    )


def build_validation_retry_prompt(
    *,
    base_prompt: str,
    issues: list[str],
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
            "Remove every unsupported top-level field. Only emit the defined OsaAdvisorOutput fields.\n\n"
            + _QOS_EXAMPLE
        )
    elif "did not converge" in joined or "max iterations" in joined:
        correction = (
            "You ran out of iterations without returning a final JSON. "
            "Stop calling tools once you have enough evidence. "
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
            + _QOS_EXAMPLE
        )

    return (
        f"{cleaned}\n\n"
        f"Retry feedback:\n{correction}"
    )


__all__ = [
    "OSA_CORE_PROMPT",
    "OSA_DYNAMIC_RULES",
    "OSA_SYSTEM_PROMPT",
    "build_advisor_user_prompt",
    "build_validation_retry_prompt",
]
